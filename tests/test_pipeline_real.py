"""End to end on real CREMA-D data (skipped when data/ is absent): cached face crops + WAV -> Omni processor;
manifest questions -> tokens; one packed pass on a tiny random thinker == one sequence per question."""
import json
from pathlib import Path

import pytest
import torch

from emotion_decider.dataset import load_media
from emotion_decider.omni import encode_clip, packed_hidden, row_hidden
from emotion_decider.questions import delimiter_ids, encode_question

MANIFEST = Path("data/crema-d/derived/manifest_seed0.jsonl")
pytestmark = pytest.mark.skipif(not MANIFEST.exists(), reason="CREMA-D manifest not built")


@pytest.fixture(scope="module")
def processor():
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained("Qwen/Qwen2.5-Omni-3B")


@pytest.fixture(scope="module")
def thinker():
    from test_omni_parity import tiny_thinker
    return tiny_thinker()


def rows(n=3):
    """The first clip with the most questions, in each condition."""
    out = {}
    for line in open(MANIFEST):
        r = json.loads(line)
        if len(r["questions"]) == 4 and r["condition"] not in out:
            out[r["condition"]] = r
        if len(out) == 3:
            return list(out.values())


@pytest.mark.parametrize("row", rows(), ids=lambda r: f"{r['clip_id']}-{r['condition']}")
@torch.no_grad()
def test_real_clip_packed_matches_rows(processor, thinker, row):
    frames, fps, audio = load_media(row["clip_id"], row["condition"])
    clip = encode_clip(processor, frames=frames, fps=fps, audio=audio, prefix="<|im_start|>user\n", suffix="<|im_end|>\n")
    ids = clip.input_ids[0].tolist()
    if frames is not None:
        assert ids.count(151656) == 512                    # 16 frames of 224 px: 8 pairs x 64 tokens
    if audio is not None:
        assert abs(ids.count(151646) - 25 * len(audio) / 16000) <= 2   # 25 audio tokens per second
    delims = delimiter_ids(processor.tokenizer)
    branches = [encode_question(processor.tokenizer, q, delims) for q in row["questions"].values()]
    for b, h in zip(branches, packed_hidden(thinker, clip, branches)):
        torch.testing.assert_close(h, row_hidden(thinker, clip, b), atol=1e-5, rtol=0)
