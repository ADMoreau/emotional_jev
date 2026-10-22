"""Correctness of the packed Omni path on a tiny random thinker (CPU, fp32).

Clips are built by the real Qwen2.5-Omni-3B processor (downloads ~5 MB of tokenizer/processor files on
first run; no weights), so the audio/video interleaving and TMRoPE layout are the production ones.
Reference: each question as its own causal sequence through the thinker's own forward.
"""
import random

import numpy as np
import pytest
import torch
from transformers import Qwen2_5OmniThinkerConfig, Qwen2_5OmniThinkerForConditionalGeneration

from emotion_decider.omni import Branch, PointerHead, encode_clip, packed_hidden, readout, row_hidden

ATOL = 1e-5


@pytest.fixture(scope="module")
def processor():
    from transformers import AutoProcessor
    try:
        return AutoProcessor.from_pretrained("Qwen/Qwen2.5-Omni-3B")
    except OSError as e:
        pytest.skip(f"Omni processor unavailable: {e}")


def tiny_thinker():
    torch.manual_seed(0)
    config = Qwen2_5OmniThinkerConfig(
        text_config=dict(vocab_size=151936, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2,
                         rope_parameters=dict(rope_type="default", rope_theta=1e6, mrope_section=[2, 3, 3])),
        audio_config=dict(num_mel_bins=128, encoder_layers=2, encoder_attention_heads=2, encoder_ffn_dim=64,
                          d_model=32, output_dim=64, n_window=100, max_source_positions=1500),
        vision_config=dict(depth=2, hidden_size=32, num_heads=2, intermediate_size=64, out_hidden_size=64,
                           patch_size=14, spatial_merge_size=2, temporal_patch_size=2, window_size=112,
                           fullatt_block_indexes=[1]),
        # set explicitly in the real Qwen2.5-Omni-3B thinker_config; not class defaults
        vision_start_token_id=151652, vision_end_token_id=151653, vision_token_id=151654)
    return Qwen2_5OmniThinkerForConditionalGeneration(config).eval()


thinker = pytest.fixture(scope="module")(tiny_thinker)


def make_clip(processor, mode, seed=0, seconds=4, fps=4, side=112):
    """mode: 'av', 'video' or 'audio'. 4 s spans two 2 s TMRoPE chunks, so interleaving is exercised."""
    rng = np.random.default_rng(seed)
    frames = rng.integers(0, 256, (seconds * fps, side, side, 3), dtype=np.uint8) if mode != "audio" else None
    audio = (0.1 * rng.standard_normal(16000 * seconds)).astype(np.float32) if mode != "video" else None
    return encode_clip(processor, frames=frames, fps=fps, audio=audio, prefix="<|im_start|>user\n", suffix="<|im_end|>\n")


def make_branch(rng, n_instr, option_lengths, specials=(151659, 151660, 151661, 151662)):
    q, opt, end_opt, decide = specials
    ids, opts = [q] + [rng.randrange(100, 30000) for _ in range(n_instr)], []
    for n in option_lengths:
        ids += [opt] + [rng.randrange(100, 30000) for _ in range(n)] + [end_opt]
        opts.append(len(ids) - 1)
    ids.append(decide)
    return Branch(ids=ids, decide=len(ids) - 1, opts=opts)


def three_branches():
    rng = random.Random(0)
    return [make_branch(rng, 6, [2, 3, 1, 2, 2, 1]),      # intended, 6 emotions
            make_branch(rng, 8, [1, 2, 1, 1, 2, 3]),      # perceived, 6 emotions
            make_branch(rng, 9, [1, 1])]                  # Noul


@pytest.fixture(scope="module")
def head():
    torch.manual_seed(1)
    return PointerHead(64, dp=32).eval()


@pytest.mark.parametrize("mode", ["av", "video", "audio"])
@torch.no_grad()
def test_packed_matches_independent_rows(processor, thinker, head, mode):
    clip, branches = make_clip(processor, mode), three_branches()
    if mode == "av":   # the layout we expect: both bos tokens, then interleaved video/audio chunks
        ids = clip.input_ids[0].tolist()
        assert ids.count(151656) > 0 and ids.count(151646) > 0
    for b, h in zip(branches, packed_hidden(thinker, clip, branches)):
        ref = row_hidden(thinker, clip, b)
        torch.testing.assert_close(h, ref, atol=ATOL, rtol=0)
        torch.testing.assert_close(readout(head, h, b), readout(head, ref, b), atol=ATOL, rtol=0)


@torch.no_grad()
def test_answer_independent_of_other_questions(processor, thinker, head):
    clip, branches = make_clip(processor, "av"), three_branches()
    alone = [readout(head, packed_hidden(thinker, clip, [b])[0], b) for b in branches]
    order = [2, 0, 1]
    for i, h in zip(order, packed_hidden(thinker, clip, [branches[i] for i in order])):
        torch.testing.assert_close(readout(head, h, branches[i]), alone[i], atol=ATOL, rtol=0)


@torch.no_grad()
def test_each_modality_reaches_the_question(processor, thinker):
    """Changing only the pixels, or only the audio, must change the <decide> state: both encoders are live.
    (Compared on hidden states: a random pointer head's probabilities sit near uniform.)"""
    b = three_branches()[0]
    rng = np.random.default_rng(7)
    h0 = packed_hidden(thinker, make_clip(processor, "av"), [b])[0][b.decide]
    for key in ("pixel_values_videos", "input_features"):
        clip = make_clip(processor, "av")
        clip.mm[key] = clip.mm[key] + torch.as_tensor(rng.standard_normal(clip.mm[key].shape), dtype=clip.mm[key].dtype)
        h = packed_hidden(thinker, clip, [b])[0][b.decide]
        assert (h - h0).abs().max() > 1e-3, key


@torch.no_grad()
def test_mask_matters(processor, thinker):
    """Guards against a vacuous parity test: without the block mask, a later branch must see an earlier one."""
    clip, (b1, b2, _) = make_clip(processor, "av"), three_branches()
    joined = Branch(ids=b1.ids + b2.ids, decide=0, opts=[])
    h = row_hidden(thinker, clip, joined)[len(b1.ids):]
    assert (h - packed_hidden(thinker, clip, [b1, b2])[1]).abs().max() > 1e-3


@torch.no_grad()
def test_batched_clips_match_one_at_a_time(processor, thinker):
    """Several clips (mixed conditions, so different lengths and encoders) in one padded forward pass give the
    same branch hidden states as one clip per pass, in any batch order."""
    from emotion_decider.omni import packed_hidden_batch
    clips = [make_clip(processor, "av", seed=0), make_clip(processor, "voice", seed=1),
             make_clip(processor, "face", seed=2), make_clip(processor, "av", seed=3, seconds=2)]
    bs = three_branches()
    branch_lists = [bs, bs[:1], bs[1:], [bs[2], bs[0]]]
    single = [packed_hidden(thinker, c, b) for c, b in zip(clips, branch_lists)]
    for order in ([0, 1, 2, 3], [3, 1, 0, 2]):
        batched = packed_hidden_batch(thinker, [clips[i] for i in order], [branch_lists[i] for i in order])
        for i, row in zip(order, batched):
            for a, b in zip(row, single[i]):
                torch.testing.assert_close(a, b, atol=ATOL, rtol=0)


@pytest.mark.parametrize("side", [224, 168])   # 224: equal windows (batched path); 168: unequal (loop fallback)
def test_batched_vision_attention_matches_loop(processor, side):
    """batch_vision_attention: same ViT features and same gradients as transformers' per-window loop."""
    from emotion_decider.omni import batch_vision_attention
    loop, batched = tiny_thinker(), batch_vision_attention(tiny_thinker())   # same seed: same weights
    clip = make_clip(processor, "video", side=side)
    grads = []
    for t in (loop, batched):
        t.zero_grad()
        feats = t.get_video_features(clip.mm["pixel_values_videos"], clip.mm["video_grid_thw"],
                                     return_dict=True).pooler_output[0]
        (feats.square().sum()).backward()
        grads.append((feats.detach(), [b.attn.q.weight.grad.clone() for b in t.visual.blocks]))
    torch.testing.assert_close(grads[0][0], grads[1][0], atol=1e-5, rtol=1e-5)
    for g0, g1 in zip(grads[0][1], grads[1][1]):
        torch.testing.assert_close(g0, g1, atol=1e-4, rtol=1e-4)
