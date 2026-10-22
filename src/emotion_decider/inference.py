"""Load a released EmotionDecider checkpoint and answer Jev-format questions about one clip (no server).

    from emotion_decider.inference import load, answer
    model = load("path/to/EmotionDecider-AV-3B")    # decision.safetensors, decision.json, calibration.json
    out = answer(model, questions, frames=frames, fps=fps, audio=audio)
    # -> {"answers": {id: Answer}, "usage": {"input_tokens": n, "output_tokens": 0}}

frames: [T, H, W, 3] uint8 face crops (T even, H and W multiples of 28; training used 16 frames at 224 px, a square
box about 1.3x the detected face; see faces.py), fps: the rate those frames were sampled at, audio: float32 16 kHz mono
covering the same time span. Either frames or audio may be None. All questions run in one packed prefill and never
see each other.
"""
import json
from pathlib import Path

import numpy as np
import torch

from . import metrics as M
from .decision import DecisionModel
from .omni import encode_clip
from .questions import encode_question, option_keys

# Temperature per question type when the question id is not one of the trained families in calibration.json
# (intended, perceived, noul, intensity).
DEFAULT_FAMILY = {"choice": "intended", "noul": "noul", "score": "intensity"}


def load(path, device=None, dtype=torch.bfloat16, thinker=None, processor=None):
    """path: a directory with decision.safetensors (or decision.pt) + decision.json, and calibration.json there or in
    its parent (a training run's best/). thinker / processor: pre-built ones (tests); by default the pinned backbone is loaded."""
    path = Path(path)
    meta = json.loads((path / "decision.json").read_text())
    if (path / "decision.safetensors").exists():   # the release format: one flat file, groups as key prefixes
        from safetensors.torch import load_file
        state = unflatten(load_file(path / "decision.safetensors"))
    else:                                           # a training run's best/ or final/
        state = torch.load(path / "decision.pt", map_location="cpu", weights_only=True)
    calib = next((p / "calibration.json" for p in (path, path.parent) if (p / "calibration.json").exists()), None)
    temperatures = json.loads(calib.read_text()) if calib else {}
    pointer_dim = state["head"]["q.weight"].shape[0]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if processor is None:
        from transformers import AutoProcessor
        b = meta["backbone"]
        processor = AutoProcessor.from_pretrained(b["name"], revision=b.get("revision"))
    if thinker is None:
        b = meta["backbone"]
        model = DecisionModel.from_pretrained(b["name"], processor.tokenizer, revision=b.get("revision"), dtype=dtype,
                                              device=device, attn=b.get("attn", "sdpa"), lora=meta["lora"],
                                              pointer_dim=pointer_dim)
    else:
        model = DecisionModel(thinker, processor.tokenizer, lora=meta["lora"], pointer_dim=pointer_dim).to(device)
    model.load_state(state)
    model.eval()
    model.processor, model.temperatures = processor, temperatures
    return model


@torch.no_grad()
def answer(model, questions, frames=None, fps=None, audio=None):
    """questions: {id: {"type": "choice" | "noul" | "score", "instructions", "criteria"}} (Jev request schema).
    -> {"answers": {id: Answer}, "usage": {"input_tokens", "output_tokens"}} (Jev response schema)."""
    clip = encode_clip(model.processor, frames=frames, fps=fps, audio=audio, prefix=model.prefix, suffix=model.suffix)
    ids = list(questions)
    qs = [questions[i] for i in ids]
    for i, q in zip(ids, qs):
        _validate(i, q)
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if model.device.type == "cuda" else torch.autocast("cpu", enabled=False)
    with autocast:
        logits = model.logits(clip, qs)
    answers = {}
    for i, q, z in zip(ids, qs, logits):
        family = i if i in model.temperatures else DEFAULT_FAMILY[q["type"]]
        probs = M.softmax(z.float().cpu().numpy(), model.temperatures.get(family, 1.0))
        answers[i] = format_answer(q, probs)
    branch_tokens = sum(len(encode_question(model.tokenizer, q, model.delimiters).ids) for q in qs)
    return {"answers": answers, "usage": {"input_tokens": clip.length + branch_tokens, "output_tokens": 0}}


def flatten(state):
    """DecisionModel.state() -> one flat {group.key: tensor} dict (safetensors holds no nesting)."""
    return {f"{group}.{k}": v.contiguous() for group, d in state.items() for k, v in d.items()}


def unflatten(flat):
    state = {}
    for key, v in flat.items():
        group, k = key.split(".", 1)
        state.setdefault(group, {})[k] = v
    return state


def format_answer(question, probs):
    """probs in option_keys(question) order -> one Jev-format answer."""
    probs = np.asarray(probs, dtype=float)
    keys, n, m = option_keys(question), len(probs), int(np.argmax(probs))
    if question["type"] == "noul":
        return {"type": "noul", "noul": float(probs[0])}
    if question["type"] == "score":
        dist = np.abs(np.arange(n) - m)
        conf = max(0.0, 1.0 - float(probs @ dist) / dist.mean()) if dist.mean() > 0 else 1.0
        return {"type": "score", "score": float(probs @ np.arange(n)),
                "probabilities": {k: float(p) for k, p in zip(keys, probs)}, "confidence": conf,
                "legend": {k: level for k, level in zip(keys, question["criteria"])}}
    return {"type": "choice", "choice": keys[m], "probabilities": {k: float(p) for k, p in zip(keys, probs)},
            "confidence": float((probs[m] - 1 / n) / (1 - 1 / n)) if n > 1 else 1.0}


def _validate(qid, q):
    t = q.get("type")
    if t not in DEFAULT_FAMILY or "instructions" not in q:
        raise ValueError(f"{qid}: needs type choice/noul/score and instructions")
    c = q.get("criteria")
    if t == "choice" and not (isinstance(c, dict) and 1 <= len(c) <= 255):
        raise ValueError(f"{qid}: choice criteria must be a map of 1-255 keys")
    if t == "score" and not (isinstance(c, list) and 2 <= len(c) <= 10):
        raise ValueError(f"{qid}: score criteria must be an ordered list of 2-10 levels")
    if t == "noul" and c is not None and not (isinstance(c, dict) and set(c) <= {"true", "false"}):
        raise ValueError(f"{qid}: noul criteria must be {{'true': ..., 'false': ...}}")
