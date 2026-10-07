"""Decision core: the clip's state runs once, every question branch continues from a fork of it.

Derived from Kev (https://github.com/jaredpalmer/kev, kev/model.py, Apache-2.0): PointerHead and the
prefix-cache replica. Modified: the state is a Qwen3.5 video prompt with 3-D multimodal RoPE positions,
and the backbone is the multimodal Qwen3_5Model rather than the text-only model.

Why fork instead of a block-causal mask: Qwen3.5's Gated DeltaNet layers are recurrent and ignore
attention masks, so a packed [state][q1][q2] sequence would let q2 read q1 through the recurrent state.
Each branch instead starts from its own copy of what every layer holds at the end of the state: the
attention layers' keys/values and the DeltaNet layers' conv window and recurrent state.
"""
import copy
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DynamicCache
from transformers.cache_utils import LinearAttentionCacheLayerMixin


@dataclass
class State:
    """One clip's shared prefix, as the Qwen3.5 processor returns it (batch of 1)."""
    input_ids: torch.Tensor                    # [1, Ls]
    mm_token_type_ids: torch.Tensor            # [1, Ls]: 0 text, 2 video
    pixel_values_videos: torch.Tensor | None = None
    video_grid_thw: torch.Tensor | None = None

    def vision_inputs(self):
        return {"mm_token_type_ids": self.mm_token_type_ids, "pixel_values_videos": self.pixel_values_videos,
                "video_grid_thw": self.video_grid_thw}


@dataclass
class Branch:
    """One question's tokens after the state, with its readout offsets within the branch."""
    ids: list[int]
    decide: int                                # offset of the <decide> token
    opts: list[int]                            # offset of each option's </opt> token


@dataclass
class Prefix:
    """The state after one pass: its cache (batch 1, never mutated by branches) and where text positions resume."""
    length: int
    next_pos: int
    cache: DynamicCache


class PointerHead(nn.Module):
    def __init__(self, d, dp=256):
        super().__init__()
        self.q, self.k = nn.Linear(d, dp), nn.Linear(d, dp)
        self.scale = 1 / math.sqrt(dp)

    def forward(self, h_decide, h_opts):       # [d], [K, d] -> logits [K]
        # Unit-RMS inputs: the backbone's final hidden states are not unit scale (Qwen2.5-Omni-3B's gave logit
        # spreads of 40-70 with the raw states, i.e. one-hot softmax at init); normalized, logits start near 0.
        h_decide = F.rms_norm(h_decide, (h_decide.shape[-1],))
        h_opts = F.rms_norm(h_opts, (h_opts.shape[-1],))
        return (self.k(h_opts) @ self.q(h_decide)) * self.scale


def state_positions(vlm, state):
    """[3, 1, Ls] M-RoPE positions of the state, from the model's own get_rope_index."""
    pos, _ = vlm.get_rope_index(state.input_ids, mm_token_type_ids=state.mm_token_type_ids,
                                video_grid_thw=state.video_grid_thw)
    return pos


@torch.no_grad()
def encode_state(vlm, state):
    """Run the state once (vision tower included) and keep every layer's cache. Text after the state takes
    position max(state positions) + 1 on all three axes, as get_rope_index assigns it."""
    pos = state_positions(vlm, state)
    out = vlm(input_ids=state.input_ids, position_ids=pos, past_key_values=DynamicCache(config=vlm.config.text_config),
              use_cache=True, **state.vision_inputs())
    return Prefix(state.input_ids.shape[1], int(pos.max()) + 1, out.past_key_values)


def fork(cache, n):
    """n independent copies of a batch-1 cache as one batch-n cache; the source stays untouched and reusable.
    The cache layers update their tensors in place (static addresses for CUDA graphs), so each linear-attention
    layer's state dicts are copied before reorder_cache replaces the tensors with fresh batch-n ones."""
    replica = copy.copy(cache)
    replica.layers = [copy.copy(layer) for layer in cache.layers]
    for layer in replica.layers:
        if isinstance(layer, LinearAttentionCacheLayerMixin):
            for name in ("conv_states", "recurrent_states", "is_conv_states_initialized",
                         "is_recurrent_states_initialized", "has_previous_state", "conv_kernel_size"):
                setattr(layer, name, dict(getattr(layer, name)))
    device = next(t for layer in cache.layers for t in _tensors(layer)).device
    replica.reorder_cache(torch.zeros(n, dtype=torch.long, device=device))
    return replica


def _tensors(layer):
    if isinstance(layer, LinearAttentionCacheLayerMixin):
        yield from (t for t in [*layer.conv_states.values(), *layer.recurrent_states.values()] if t is not None)
    if getattr(layer, "keys", None) is not None:
        yield layer.keys


@torch.no_grad()
def branch_hidden(vlm, prefix, branches, pad_id=0):
    """Hidden states [len(b.ids), d] of each branch, all branches in one batched pass from a fork of the prefix.
    Branches are right-padded; pads come after every real token and are masked, so they change nothing."""
    n, L = len(branches), max(len(b.ids) for b in branches)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, prefix.length + L), dtype=torch.long)
    att[:, :prefix.length] = 1
    pos = (prefix.next_pos + torch.arange(L)).view(1, 1, L).repeat(3, n, 1)
    for i, b in enumerate(branches):
        ids[i, :len(b.ids)] = torch.tensor(b.ids)
        att[i, prefix.length:prefix.length + len(b.ids)] = 1
    out = vlm.language_model(input_ids=ids, attention_mask=att, position_ids=pos,
                             past_key_values=fork(prefix.cache, n), use_cache=True)
    return [out.last_hidden_state[i, :len(b.ids)] for i, b in enumerate(branches)]


@torch.no_grad()
def row_hidden(vlm, state, branch):
    """Reference (Qev/Valen's form): state + one branch as its own causal row, positions from get_rope_index
    over the whole row. Exact isolation by construction; costs a full state pass per question."""
    ids = torch.cat([state.input_ids, torch.tensor([branch.ids])], 1)
    types = torch.cat([state.mm_token_type_ids, torch.zeros((1, len(branch.ids)), dtype=state.mm_token_type_ids.dtype)], 1)
    out = vlm(input_ids=ids, mm_token_type_ids=types, pixel_values_videos=state.pixel_values_videos,
              video_grid_thw=state.video_grid_thw)
    return out.last_hidden_state[0, state.input_ids.shape[1]:]


def readout(head, h, branch):
    """Option probabilities of one branch from its hidden states."""
    return torch.softmax(head(h[branch.decide], h[branch.opts]), -1)
