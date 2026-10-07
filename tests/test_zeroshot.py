"""Zero-shot scorer on a tiny random thinker: packed letter logits == one sequence per question."""
import random

import torch

from facedecide.omni import row_hidden
from facedecide.questions import make_questions
from facedecide.zeroshot import ZeroShot
from test_omni_parity import make_clip, processor, tiny_thinker  # noqa: F401  (processor is a fixture)

ROW = {"clip_id": "x", "intended": "A", "perceived": {
    m: {"distribution": {"anger": 0.6, "disgust": 0.3, "fear": 0.0, "happiness": 0.0, "neutral": 0.1, "sadness": 0.0},
        "ratings": [["A", 80.0], ["D", 55.0], ["N", 90.0]]} for m in ("av", "face", "voice")}}


@torch.no_grad()
def test_zero_shot_packed_matches_rows(processor):  # noqa: F811
    thinker = tiny_thinker()
    zs = ZeroShot(thinker, processor.tokenizer)
    clip = make_clip(processor, "av")
    qs, _ = make_questions(ROW, "av", random.Random(0), p_noul=1.0, p_score=1.0)
    questions = list(qs.values())
    packed = zs.logits(clip, questions)
    assert [len(z) for z in packed] == [6, 6, 2, 4]
    for q, z in zip(questions, packed):
        h = row_hidden(thinker, clip, zs.branch(q))
        ref = thinker.lm_head(h[-1])[zs.letters[: len(z)]]
        torch.testing.assert_close(z, ref.float(), atol=1e-4, rtol=0)
    branch = processor.tokenizer.decode(zs.branch(questions[0]).ids)
    assert branch.rstrip().endswith("<|im_start|>assistant") and "\nA. " in branch and "\nF. " in branch


@torch.no_grad()
def test_zero_shot_batch_matches_single(processor):  # noqa: F811
    thinker = tiny_thinker()
    zs = ZeroShot(thinker, processor.tokenizer)
    clips = [make_clip(processor, "av"), make_clip(processor, "voice", seed=1)]
    qlists = [list(make_questions(ROW, "av", random.Random(s), p_noul=1.0, p_score=1.0)[0].values()) for s in (0, 1)]
    for clip, qs, zs_batch in zip(clips, qlists, zs.logits_batch(clips, qlists)):
        for a, b in zip(zs_batch, zs.logits(clip, qs)):
            torch.testing.assert_close(a, b, atol=1e-4, rtol=0)
