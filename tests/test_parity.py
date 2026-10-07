"""Correctness of the shared video state on a tiny random Qwen3.5 (CPU, fp32, no downloads).

Reference: each question as its own causal row, state included (Qev/Valen's form), with positions from
Qwen's own get_rope_index. The forked path must match it, and a question's answer must not depend on
which other questions are asked (CLAUDE.md correctness tests a and b).
"""
import random

import pytest
import torch
from transformers import Qwen3_5Config, Qwen3_5Model

from facedecide.model import Branch, PointerHead, State, branch_hidden, encode_state, readout, row_hidden

VIDEO, V_START, V_END, IMAGE = 250, 251, 252, 253
Q, OPT, END_OPT, DECIDE = 241, 242, 243, 244
ATOL = 1e-5


def tiny_vlm():
    torch.manual_seed(0)
    config = Qwen3_5Config(
        text_config=dict(vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=8,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=64,
                         linear_num_key_heads=4, linear_num_value_heads=4, linear_key_head_dim=16,
                         linear_value_head_dim=16, linear_conv_kernel_dim=4,
                         rope_parameters=dict(rope_type="default", rope_theta=1e7, partial_rotary_factor=0.25,
                                              mrope_section=[3, 3, 2], mrope_interleaved=True)),
        vision_config=dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64,
                           patch_size=16, temporal_patch_size=2, spatial_merge_size=2, num_position_embeddings=64),
        video_token_id=VIDEO, vision_start_token_id=V_START, vision_end_token_id=V_END, image_token_id=IMAGE)
    model = Qwen3_5Model(config).eval()
    assert model.config.text_config.layer_types.count("linear_attention") == 6   # 3 DeltaNet : 1 attention, as in 0.8B
    return model


def tiny_head():
    torch.manual_seed(1)
    return PointerHead(64, dp=32).eval()



vlm = pytest.fixture(scope="module")(tiny_vlm)
head = pytest.fixture(scope="module")(tiny_head)


def make_state(seed, frames=8, grid=4):
    """A video prompt laid out like the processor's: per frame pair, timestamp text, then
    <vision_start> video tokens <vision_end>. grid x grid patches per frame -> grid^2 / 4 tokens per pair."""
    g = torch.Generator().manual_seed(seed)
    t, per_pair = frames // 2, grid * grid // 4
    ids = [5, 6, 7]
    for i in range(t):
        ids += [10 + i, 20, V_START] + [VIDEO] * per_pair + [V_END]
    ids += [8, 9]
    pixels = torch.randn(t * grid * grid, 3 * 2 * 16 * 16, generator=g)
    return State(input_ids=torch.tensor([ids]), mm_token_type_ids=torch.tensor([[2 if x == VIDEO else 0 for x in ids]]),
                 pixel_values_videos=pixels, video_grid_thw=torch.tensor([[t, grid, grid]]))


def make_branch(rng, n_instr, option_lengths):
    ids, opts = [Q] + [rng.randrange(30, 200) for _ in range(n_instr)], []
    for n in option_lengths:
        ids += [OPT] + [rng.randrange(30, 200) for _ in range(n)] + [END_OPT]
        opts.append(len(ids) - 1)
    ids.append(DECIDE)
    return Branch(ids=ids, decide=len(ids) - 1, opts=opts)


def three_branches():
    rng = random.Random(0)
    return [make_branch(rng, 6, [2, 3, 1, 2, 2, 1, 3]),    # Choice, 7 options
            make_branch(rng, 9, [1, 1]),                   # Noul
            make_branch(rng, 4, [3, 4, 3, 5])]             # Score, 4 levels


branches = pytest.fixture(scope="module")(three_branches)


def test_fork_matches_independent_rows(vlm, head, branches):
    state = make_state(0)
    forked = branch_hidden(vlm, encode_state(vlm, state), branches)
    for b, h in zip(branches, forked):
        ref = row_hidden(vlm, state, b)
        torch.testing.assert_close(h, ref, atol=ATOL, rtol=0)
        torch.testing.assert_close(readout(head, h, b), readout(head, ref, b), atol=ATOL, rtol=0)


def test_answer_independent_of_other_questions(vlm, head, branches):
    prefix = encode_state(vlm, make_state(0))
    alone = [readout(head, branch_hidden(vlm, prefix, [b])[0], b) for b in branches]
    together = [readout(head, h, b) for b, h in zip(branches, branch_hidden(vlm, prefix, branches))]
    order = [2, 0, 1]
    shuffled = branch_hidden(vlm, prefix, [branches[i] for i in order])
    for p_alone, p_together in zip(alone, together):
        torch.testing.assert_close(p_together, p_alone, atol=ATOL, rtol=0)
    for i, h in zip(order, shuffled):
        torch.testing.assert_close(readout(head, h, branches[i]), alone[i], atol=ATOL, rtol=0)


def test_prefix_is_reusable(vlm, branches):
    """Forking must not write into the cached state: a second round of questions sees the same state."""
    prefix = encode_state(vlm, make_state(0))
    first = branch_hidden(vlm, prefix, branches)
    second = branch_hidden(vlm, prefix, branches)
    for a, b in zip(first, second):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_positions_matter(vlm, branches):
    """Guards against a vacuous parity test: an off-by-one text position must break it."""
    state = make_state(0)
    prefix = encode_state(vlm, state)
    prefix.next_pos += 1
    h = branch_hidden(vlm, prefix, branches[:1])[0]
    assert (h - row_hidden(vlm, state, branches[0])).abs().max() > 1e-3


def test_video_reaches_the_answer(vlm, head, branches):
    """Different pixels, same tokens: the answer must change, so the vision path is live."""
    b = branches[0]
    p0 = readout(head, branch_hidden(vlm, encode_state(vlm, make_state(0)), [b])[0], b)
    p1 = readout(head, branch_hidden(vlm, encode_state(vlm, make_state(1)), [b])[0], b)
    assert (p0 - p1).abs().max() > 1e-4


def test_deltanet_state_matters(vlm, branches):
    """Guards against a fork that silently drops the recurrent state: zeroing it must break parity."""
    from transformers.cache_utils import LinearAttentionCacheLayerMixin
    state = make_state(0)
    prefix = encode_state(vlm, state)
    for layer in prefix.cache.layers:
        if isinstance(layer, LinearAttentionCacheLayerMixin):
            layer.recurrent_states = {k: None if v is None else torch.zeros_like(v) for k, v in layer.recurrent_states.items()}
    h = branch_hidden(vlm, prefix, branches[:1])[0]
    assert (h - row_hidden(vlm, state, branches[0])).abs().max() > 1e-3
