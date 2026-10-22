"""Decision core on the Qwen2.5-Omni thinker: one audio-visual clip, many questions, one packed pass.

The thinker's language model is attention-only, so Kev's packed form applies directly
(https://github.com/jaredpalmer/kev, kev/model.py, Apache-2.0; branch mask derived from
branch_mask_batch, modified): the sequence is [clip state][branch 1][branch 2]..., and a block-causal
mask lets each branch see the state and itself only. Positions are TMRoPE: the state's come from the
thinker's own get_rope_index; every branch restarts at max(state positions) + 1 on all three axes,
which is where get_rope_index puts text that follows the clip.
"""
from dataclasses import dataclass, field

import numpy as np
import torch

from .model import Branch, PointerHead, readout  # noqa: F401  (shared with the Qwen3.5 path)

# Delimiters: existing Qwen2.5 special tokens that Omni never uses, so no embedding rows are added.
QUESTION, OPTION, END_OPTION, DECIDE = "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>", "<|fim_pad|>"


@dataclass
class Clip:
    """One clip's state as the Omni processor returns it (batch of 1). Video, audio, or both."""
    input_ids: torch.Tensor                              # [1, Ls]
    mm: dict = field(default_factory=dict)               # processor tensors for the encoders
    use_audio_in_video: bool = False

    @property
    def length(self):
        return self.input_ids.shape[1]

    def to(self, device):
        return Clip(self.input_ids.to(device), {k: v.to(device) if torch.is_tensor(v) else v for k, v in self.mm.items()},
                    self.use_audio_in_video)


def encode_clip(processor, frames=None, fps=None, audio=None, prefix="", suffix=""):
    """Build a Clip from sampled frames ([T, H, W, C] uint8 face crops, T even) at `fps` and/or 16 kHz mono audio.
    The crops are used at their own size (224 px -> 64 tokens per frame pair).
    With both, audio and video are interleaved in 2 s chunks on one timeline (use_audio_in_video)."""
    if frames is None and audio is None:
        raise ValueError("a clip needs frames, audio, or both")
    both = frames is not None and audio is not None
    if frames is not None:
        media = "<|vision_bos|><|VIDEO|><|vision_eos|>"
    else:
        media = "<|audio_bos|><|AUDIO|><|audio_eos|>"
    kwargs = {"text": prefix + media + suffix, "return_tensors": "pt"}
    if frames is not None:
        frames = np.asarray(frames)
        h, w = frames.shape[1:3]
        if h % 28 or w % 28:
            raise ValueError(f"frames must be crops with sides divisible by 28 (patch 14 x merge 2), got {h}x{w}")
        # pin the target size to the crop's own: the processor would otherwise upscale small frames
        kwargs.update(videos=[frames], fps=fps, do_sample_frames=False, cap_pixels_per_frame=True,
                      size={"shortest_edge": h * w, "longest_edge": h * w})
    if audio is not None:
        kwargs.update(audio=[np.asarray(audio, dtype=np.float32)])
    out = processor(**kwargs, use_audio_in_video=both)
    mm = {k: out[k] for k in ("input_features", "feature_attention_mask", "pixel_values_videos", "video_grid_thw")
          if k in out}
    if "video_second_per_grid" in out:
        mm["video_second_per_grid"] = torch.as_tensor(out["video_second_per_grid"], dtype=torch.float)
    return Clip(input_ids=out["input_ids"], mm=mm, use_audio_in_video=both)


def clip_positions(thinker, clip):
    """[3, 1, Ls] TMRoPE positions of the clip state, from the thinker's own get_rope_index."""
    fam = clip.mm.get("feature_attention_mask")
    pos, _ = thinker.get_rope_index(
        clip.input_ids, video_grid_thw=clip.mm.get("video_grid_thw"),
        attention_mask=torch.ones_like(clip.input_ids), use_audio_in_video=clip.use_audio_in_video,
        audio_seqlens=None if fam is None else fam.sum(-1), second_per_grids=clip.mm.get("video_second_per_grid"))
    return pos


def batch_vision_attention(thinker):
    """Under sdpa, transformers runs the ViT's attention as a Python loop: one call per window (64 patches at a
    224 px crop) and per frame pair in the full-attention blocks, ~4k small kernels per forward for 4 clips. When
    every chunk has the same length (any crop whose patch grid the windows tile, e.g. 224 px) this does one batched
    call per layer instead; same math. Other lengths fall back to the loop. In place; returns the thinker."""
    if thinker.visual.config._attn_implementation != "sdpa":
        return thinker
    for block in thinker.visual.blocks:
        attn = block.attn
        loop = attn.forward

        def forward(hidden_states, cu_seqlens, position_embeddings=None, max_seqlen=None, _attn=attn, _loop=loop,
                    **kwargs):
            lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
            if len(set(lengths)) != 1:
                return _loop(hidden_states, cu_seqlens, position_embeddings=position_embeddings,
                             max_seqlen=max_seqlen, **kwargs)
            from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import apply_rotary_pos_emb_vision
            S, H, n, L = hidden_states.shape[0], _attn.num_heads, len(lengths), lengths[0]
            q, k, v = (proj(hidden_states).reshape(S, H, -1) for proj in (_attn.q, _attn.k, _attn.v))
            q, k = apply_rotary_pos_emb_vision(q, k, *position_embeddings)
            q, k, v = (t.reshape(n, L, H, -1).transpose(1, 2) for t in (q, k, v))     # [windows, H, L, D]
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=_attn.scaling)
            return _attn.proj(out.transpose(1, 2).reshape(S, -1))

        attn.forward = forward
    return thinker


def embed_clips(thinker, clips):
    """Input embeddings [Ls_i, d] of each clip's state: token embeddings with the audio and video encoder outputs
    scattered into their placeholder tokens (the thinker's forward, steps 1-2). The encoders run once over all
    clips (audio clips batched, video frames concatenated): fewer, larger kernels than one call per clip."""
    embed = thinker.get_input_embeddings()
    embs = [embed(c.input_ids)[0] for c in clips]
    audio_ids, video_ids = thinker.config.audio_token_id, thinker.config.video_token_id
    a = [i for i, c in enumerate(clips) if "input_features" in c.mm]
    if a:
        out = thinker.get_audio_features(torch.cat([clips[i].mm["input_features"] for i in a]),
                                         torch.cat([clips[i].mm["feature_attention_mask"] for i in a]),
                                         return_dict=True).last_hidden_state
        counts = [int((clips[i].input_ids == audio_ids).sum()) for i in a]
        for i, feats in zip(a, out.split(counts)):
            embs[i] = embs[i].masked_scatter((clips[i].input_ids[0] == audio_ids)[:, None], feats.to(embs[i].dtype))
    v = [i for i, c in enumerate(clips) if "pixel_values_videos" in c.mm]
    if v:
        per_video = thinker.get_video_features(torch.cat([clips[i].mm["pixel_values_videos"] for i in v]),
                                               torch.cat([clips[i].mm["video_grid_thw"] for i in v]),
                                               return_dict=True).pooler_output    # one tensor per video, in order
        for i, feats in zip(v, per_video):
            embs[i] = embs[i].masked_scatter((clips[i].input_ids[0] == video_ids)[:, None], feats.to(embs[i].dtype))
    return embs


def embed_clip(thinker, clip):
    """[1, Ls, d] input embeddings of one clip's state."""
    return embed_clips(thinker, [clip])[0][None]


def block_causal_mask(segs, dtype, device, length=None):
    """Additive [B, 1, L, L] for right-padded rows: attend(i, j) iff j <= i and (seg[j] == 0 or seg[j] == seg[i]).
    Pad keys (seg -1) are masked for every query; pad query rows keep their diagonal so no row is fully masked (from
    Kev's branch_mask_batch). Real tokens never see pads: pads come after them and belong to no segment."""
    L = max(length or 0, max(len(s) for s in segs))
    s = torch.full((len(segs), L), -1, device=device)
    for b, seg in enumerate(segs):
        s[b, :len(seg)] = torch.tensor(seg, device=device)
    allow = torch.ones(L, L, dtype=torch.bool, device=device).tril()[None]
    allow = allow & ((s[:, None, :] == s[:, :, None]) | (s[:, None, :] == 0)) & (s != -1)[:, None, :]
    allow = allow | torch.eye(L, dtype=torch.bool, device=device)[None]
    return torch.zeros(len(segs), L, L, dtype=dtype, device=device).masked_fill(~allow, torch.finfo(dtype).min)[:, None]


def packed_hidden_batch(thinker, clips, branch_lists, encoders_grad=False):
    """One forward pass for several clips: row k is clip k's state followed by its branches (block-causal within
    the row), right-padded to the longest row. -> per clip, the hidden states [len(b.ids), d] of each branch.
    Rows are independent (no attention across rows; pads masked), so a clip's answers don't depend on what else is
    in the batch. encoders_grad=False (frozen encoders) runs them without an autograd graph."""
    with torch.set_grad_enabled(encoders_grad and torch.is_grad_enabled()):
        states = embed_clips(thinker, clips)
    device, dtype, embed = states[0].device, states[0].dtype, thinker.get_input_embeddings()
    rows_emb, rows_pos, segs = [], [], []
    for clip, branches, state in zip(clips, branch_lists, states):
        state_pos = clip_positions(thinker, clip).to(device)[:, 0]                       # [3, Ls]
        start = int(state_pos.max()) + 1
        ids = torch.tensor([t for b in branches for t in b.ids], device=device)
        rows_emb.append(torch.cat([state, embed(ids)], 0))
        rows_pos.append(torch.cat([state_pos] + [(start + torch.arange(len(b.ids), device=device)).expand(3, -1)
                                                 for b in branches], 1))
        segs.append([0] * clip.length + [k for k, b in enumerate(branches, start=1) for _ in b.ids])
    L = max(len(s) for s in segs)
    emb = torch.zeros(len(clips), L, states[0].shape[-1], device=device, dtype=dtype)
    pos = torch.zeros(3, len(clips), L, dtype=torch.long, device=device)
    for k, (e, p) in enumerate(zip(rows_emb, rows_pos)):
        emb[k, :len(e)] = e
        pos[:, k, :p.shape[1]] = p
    mask = block_causal_mask(segs, dtype, device, L)
    h = thinker.model(inputs_embeds=emb, position_ids=pos,
                      attention_mask={t: mask for t in set(thinker.config.text_config.layer_types)}).last_hidden_state
    out = []
    for k, (clip, branches) in enumerate(zip(clips, branch_lists)):
        cursor, row = clip.length, []
        for b in branches:
            row.append(h[k, cursor:cursor + len(b.ids)])
            cursor += len(b.ids)
        out.append(row)
    return out


def packed_hidden(thinker, clip, branches, encoders_grad=False):
    """Hidden states [len(b.ids), d] of every branch of one clip, all scored in one packed pass over its state."""
    return packed_hidden_batch(thinker, [clip], [branches], encoders_grad)[0]


def row_hidden(thinker, clip, branch):
    """Reference: clip + one branch as an ordinary causal sequence through the thinker's own forward
    (its own embedding merge, position computation and mask). Costs a full clip pass per question."""
    captured = {}

    def keep(module, args, out):
        captured["h"] = out.last_hidden_state   # returns None: the output is left unchanged

    hook = thinker.model.register_forward_hook(keep)
    try:
        ids = torch.cat([clip.input_ids, torch.tensor([branch.ids])], 1)
        thinker(input_ids=ids, attention_mask=torch.ones_like(ids), use_audio_in_video=clip.use_audio_in_video, **clip.mm)
    finally:
        hook.remove()
    return captured["h"][0, clip.length:]
