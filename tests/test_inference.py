"""Release loading and Jev-format answers, on a tiny random thinker (CPU)."""
import json

import numpy as np
import pytest
import torch

from emotion_decider import metrics as M
from emotion_decider.decision import DecisionModel
from emotion_decider.inference import answer, format_answer, load
from emotion_decider.omni import encode_clip
from test_omni_parity import tiny_thinker

LORA = {"rank": 4, "alpha": 8, "dropout": 0.0, "vision_targets": ["attn.q", "attn.v"], "audio_targets": ["fc1"]}
QUESTIONS = {
    "intended": {"type": "choice", "instructions": "Which emotion was the actor portraying?",
                 "criteria": {"anger": None, "fear": "worry or alarm", "neutral": None}},
    "angry": {"type": "noul", "instructions": "Would viewers say this person sounds angry?"},
    "how_strong": {"type": "score", "instructions": "How strongly does emotion come through?",
                   "criteria": ["none", "mild", "clear", "intense"]},
}


@pytest.fixture(scope="module")
def processor():
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained("Qwen/Qwen2.5-Omni-3B")


def media(seed=0):
    rng = np.random.default_rng(seed)
    return (rng.integers(0, 256, (8, 112, 112, 3), dtype=np.uint8), 4.0,
            (0.1 * rng.standard_normal(32000)).astype(np.float32))


@pytest.fixture(scope="module")
def saved(processor, tmp_path_factory):
    torch.manual_seed(1)
    m = DecisionModel(tiny_thinker(), processor.tokenizer, lora=LORA, pointer_dim=32)
    for p in m.trainable_parameters():                      # non-trivial adapters and head
        p.data.normal_(0, 0.05)
    path = tmp_path_factory.mktemp("release")
    m.save(path, meta={"backbone": {"name": "Qwen/Qwen2.5-Omni-3B"}})
    (path / "calibration.json").write_text(json.dumps({"intended": 1.5, "noul": 0.8, "intensity": 1.2}))
    return m.eval(), path


def test_loaded_model_matches_and_applies_temperatures(processor, saved):
    m, path = saved
    loaded = load(path, device="cpu", thinker=tiny_thinker(), processor=processor)
    frames, fps, audio = media()
    out = answer(loaded, QUESTIONS, frames=frames, fps=fps, audio=audio)
    clip = encode_clip(processor, frames=frames, fps=fps, audio=audio, prefix=m.prefix, suffix=m.suffix)
    with torch.no_grad():
        ref = m.logits(clip, list(QUESTIONS.values()))
    choice = M.softmax(ref[0].numpy(), 1.5)                  # id "intended" -> its own temperature
    assert [out["answers"]["intended"]["probabilities"][k] for k in ("anger", "fear", "neutral")] == pytest.approx(choice, abs=1e-5)
    assert out["answers"]["angry"]["noul"] == pytest.approx(M.softmax(ref[1].numpy(), 0.8)[0], abs=1e-5)  # noul default
    score = M.softmax(ref[2].numpy(), 1.2)                   # unknown score id -> the intensity temperature
    assert [out["answers"]["how_strong"]["probabilities"][str(i)] for i in range(4)] == pytest.approx(score, abs=1e-5)
    assert out["usage"]["output_tokens"] == 0 and out["usage"]["input_tokens"] > clip.length


def test_each_question_is_isolated_and_modalities_optional(processor, saved):
    _, path = saved
    loaded = load(path, device="cpu", thinker=tiny_thinker(), processor=processor)
    frames, fps, audio = media()
    together = answer(loaded, QUESTIONS, frames=frames, fps=fps, audio=audio)["answers"]
    alone = answer(loaded, {"angry": QUESTIONS["angry"]}, frames=frames, fps=fps, audio=audio)["answers"]
    assert alone["angry"]["noul"] == pytest.approx(together["angry"]["noul"], abs=1e-5)
    for kw in ({"frames": frames, "fps": fps}, {"audio": audio}):   # face only, voice only
        assert set(answer(loaded, QUESTIONS, **kw)["answers"]) == set(QUESTIONS)


def test_answer_formats():
    c = format_answer(QUESTIONS["intended"], [0.7, 0.2, 0.1])
    assert c["choice"] == "anger" and c["confidence"] == pytest.approx((0.7 - 1 / 3) / (2 / 3))
    assert set(format_answer(QUESTIONS["angry"], [0.3, 0.7])) == {"type", "noul"}
    s = format_answer(QUESTIONS["how_strong"], [0.1, 0.2, 0.6, 0.1])
    assert s["score"] == pytest.approx(0.2 + 1.2 + 0.3) and s["legend"]["3"] == "intense"
    mad_unif = np.mean([2, 1, 0, 1])                          # argmax level 2
    assert s["confidence"] == pytest.approx(1 - (0.1 * 2 + 0.2 * 1 + 0.1 * 1) / mad_unif)
    assert format_answer(QUESTIONS["how_strong"], [0.25] * 4)["confidence"] == pytest.approx(0.0, abs=1e-9)


def test_rejects_malformed_questions(processor, saved):
    _, path = saved
    loaded = load(path, device="cpu", thinker=tiny_thinker(), processor=processor)
    frames, fps, audio = media()
    for bad in ({"type": "choice", "instructions": "x", "criteria": {}},
                {"type": "score", "instructions": "x", "criteria": ["only one"]},
                {"type": "rank", "instructions": "x"}):
        with pytest.raises(ValueError):
            answer(loaded, {"q": bad}, frames=frames, fps=fps, audio=audio)


def test_safetensors_release_format_loads_the_same(processor, saved, tmp_path):
    """The packaged release (decision.safetensors, flat keys) gives the same answers as the training checkpoint."""
    import shutil
    from safetensors.torch import save_file
    from emotion_decider.inference import flatten
    _, path = saved
    for f in ("decision.json", "calibration.json"):
        shutil.copy(path / f, tmp_path / f)
    save_file(flatten(torch.load(path / "decision.pt", weights_only=True)), tmp_path / "decision.safetensors")
    frames, fps, audio = media()
    a = answer(load(path, device="cpu", thinker=tiny_thinker(), processor=processor), QUESTIONS, frames=frames, fps=fps, audio=audio)
    b = answer(load(tmp_path, device="cpu", thinker=tiny_thinker(), processor=processor), QUESTIONS, frames=frames, fps=fps, audio=audio)
    assert a == b
