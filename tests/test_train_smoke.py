"""Training loop smoke tests on real CREMA-D clips with a tiny random thinker (CPU; skipped without data)."""
import json
from pathlib import Path

import pytest
import torch

from emotion_decider.decision import DecisionModel
from emotion_decider.train import load_config, train

pytestmark = pytest.mark.skipif(not Path("data/crema-d/derived/targets.jsonl").exists(), reason="CREMA-D not built")
CONFIG = Path(__file__).resolve().parents[1] / "configs" / "omni_3b.yaml"


@pytest.fixture(scope="module")
def processor():
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained("Qwen/Qwen2.5-Omni-3B")


ENCODER_TARGETS = {"vision_targets": ["attn.q", "attn.k", "attn.v", "attn.proj", "gate_proj", "up_proj", "down_proj",
                                      "mlp.0", "mlp.2"],
                   "audio_targets": ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]}


def tiny_model(processor, seed=0, **lora):
    from test_omni_parity import tiny_thinker
    torch.manual_seed(seed)
    return DecisionModel(tiny_thinker(), processor.tokenizer, lora={"rank": 4, "alpha": 8, "dropout": 0.0, **lora},
                         pointer_dim=32)


def smoke_cfg(num_workers=0, **train):
    cfg = load_config(CONFIG)
    cfg["data"]["num_workers"] = num_workers
    cfg["train"].update({"grad_accum": 2, "log_every": 1, "ckpt_every": 1, "warmup_frac": 0.0, "warmup_min_steps": 0,
                         "dev_probe_clips": 0, **train})
    return cfg


def step_records(out):
    return [r for r in map(json.loads, open(out / "train_log.jsonl")) if "loss" in r]


def test_lora_and_head_are_the_only_trainable_parameters(processor):
    m = tiny_model(processor)
    names = [n for n, p in m.named_parameters() if p.requires_grad]
    assert names and all("lora_" in n or n.startswith("head.") for n in names)
    assert not any("audio_tower" in n or "visual" in n for n in names)


def test_encoder_lora_trains_and_round_trips(processor, tmp_path):
    """Encoder LoRA: every targeted ViT block, the merger and every audio layer get adapters, the training loop
    updates them (lora_B starts at zero), and the saved checkpoint restores them exactly."""
    m = tiny_model(processor, **ENCODER_TARGETS)
    lora_b = {n: p for n, p in m.named_parameters() if "lora_B" in n and ("visual" in n or "audio_tower" in n)}
    assert sum("visual.blocks" in n for n in lora_b) == 2 * 7 and sum("visual.merger" in n for n in lora_b) == 2
    assert sum("audio_tower.layers" in n for n in lora_b) == 2 * 6
    assert all(p.requires_grad and p.dtype == torch.float32 for p in lora_b.values())
    out = tmp_path / "enc"
    cfg = smoke_cfg(lr_lora=3e-3, lr_encoder=3e-3)
    cfg["augment"]["enabled"] = True                                # augmented training clips run end to end too
    train(cfg, 0, out, model=m, processor=processor, max_steps=2, limit_clips=4, eval_limit=1, log=lambda *_: None)
    assert any(p.abs().sum() > 0 for n, p in lora_b.items() if "visual" in n)
    assert any(p.abs().sum() > 0 for n, p in lora_b.items() if "audio_tower" in n)
    fresh = tiny_model(processor, seed=1, **ENCODER_TARGETS)
    fresh.load_state(torch.load(out / "final" / "decision.pt", weights_only=False))
    restored = dict(fresh.named_parameters())
    for n, p in m.named_parameters():
        if p.requires_grad:
            assert torch.equal(p.detach().cpu(), restored[n].detach().cpu()), n
    saved = torch.load(out / "checkpoint.pt", weights_only=False)
    assert len(saved["optimizer"]["param_groups"]) == 3            # LM LoRA, head, encoder LoRA (lr_encoder)
    with pytest.raises(ValueError):   # an LM-only model refuses a checkpoint carrying encoder adapters
        tiny_model(processor).load_state(torch.load(out / "final" / "decision.pt", weights_only=False))


def test_train_resume_and_outputs(processor, tmp_path):
    cfg = smoke_cfg(num_workers=2, dev_probe_clips=2, dev_every=2)   # worker processes and dev probes on
    out = tmp_path / "run"
    logs = []
    train(cfg, 0, out, model=tiny_model(processor), processor=processor, max_steps=2, limit_clips=4, log=logs.append)
    assert (out / "checkpoint.pt").exists() and (out / "final" / "decision.pt").exists()
    res = json.loads((out / "metrics.json").read_text())
    assert set(res["test"]) == {"intended", "perceived", "noul", "intensity"}
    assert set(res["test"]["intended"]) == {"av", "face", "voice"}
    assert set(res["temperatures"]) == set(res["test"])
    preds = [json.loads(l) for l in open(out / "predictions_test.jsonl")]
    assert len(preds) == 4 * 3 * 4     # 4 clips x 3 conditions x 4 question types
    # rerunning with a larger budget resumes from the checkpoint instead of starting over
    logs = []
    train(cfg, 0, out, model=tiny_model(processor), processor=processor, max_steps=3, limit_clips=4, log=logs.append)
    assert any("resumed at step 2" in l for l in logs)
    assert [r["step"] for r in step_records(out)] == [1, 2, 3]
    probes = [r for r in map(json.loads, open(out / "train_log.jsonl")) if "dev_loss" in r]
    assert [r["step"] for r in probes] == [0, 2, 3]       # at start, every 2 steps, and at the end


def test_worker_processes_do_not_change_training(processor, tmp_path):
    """Examples are pure functions of (clip, seed): 0 or 2 DataLoader workers give the same losses."""
    losses = {}
    for workers in (0, 2):
        out = tmp_path / f"w{workers}"
        train(smoke_cfg(num_workers=workers), 0, out, model=tiny_model(processor), processor=processor,
              max_steps=3, limit_clips=6, eval_limit=1, log=lambda *_: None)
        losses[workers] = [r["loss"] for r in step_records(out)]
    assert losses[0] == pytest.approx(losses[2], abs=1e-6)


def test_batching_clips_does_not_change_training(processor, tmp_path):
    """4 clips per optimizer step, 1 or 4 per forward pass: same losses and same final metrics (padding only)."""
    runs = {}
    for per_forward in (1, 4):
        out = tmp_path / f"b{per_forward}"
        res = train(smoke_cfg(grad_accum=4, clips_per_forward=per_forward, eval_batch=per_forward), 0, out,
                    model=tiny_model(processor), processor=processor, max_steps=3, limit_clips=12, eval_limit=3,
                    log=lambda *_: None)
        runs[per_forward] = ([r["loss"] for r in step_records(out)], res["test"]["intended"]["av"]["nll"])
    assert runs[1][0] == pytest.approx(runs[4][0], abs=1e-4)
    assert runs[1][1] == pytest.approx(runs[4][1], abs=1e-4)


def test_loss_decreases_when_overfitting_one_clip(processor, tmp_path, monkeypatch):
    """Training-loop wiring: loss -> gradients -> updates across steps and epochs. Questions are held fixed so
    20 steps suffice. (Under per-epoch reshuffling the tiny model also learns option content, but needs ~90
    steps: P(correct) went 0.17 -> 0.92 in a direct check; too slow for the suite.)"""
    import random
    import emotion_decider.train as T
    fixed = T.make_questions
    monkeypatch.setattr(T, "make_questions", lambda row, cond, rng, **kw: fixed(row, cond, random.Random(0), **kw))
    cfg = smoke_cfg(lr_lora=3e-3, lr_head=3e-2, epochs=20, grad_accum=1, variant="A", p_noul=0.0, p_score=0.0)
    out = tmp_path / "overfit"
    train(cfg, 0, out, model=tiny_model(processor), processor=processor, limit_clips=1, log=lambda *_: None)
    losses = [r["loss"] for r in step_records(out)]
    first, last = sum(losses[:3]) / 3, sum(losses[-3:]) / 3
    assert last < 0.5 * first, losses


def test_best_dev_checkpoint_is_the_one_tested(processor, tmp_path):
    cfg = smoke_cfg(dev_probe_clips=3, dev_every=1, dev_probe_conditions=["av", "voice"], keep_best=True,
                    lr_lora=3e-3, lr_head=3e-2)
    out = tmp_path / "best"
    res = train(cfg, 0, out, model=tiny_model(processor), processor=processor, max_steps=4, limit_clips=6,
                eval_limit=2, log=lambda *_: None)
    probes = [r for r in map(json.loads, open(out / "train_log.jsonl")) if "dev_loss" in r]
    best = min(probes, key=lambda r: r["dev_loss"])
    assert res["selected_step"] == best["step"] and res["total_steps"] == 4
    meta = json.loads((out / "best" / "decision.json").read_text())
    assert meta["step"] == best["step"] and meta["loss"] == pytest.approx(best["dev_loss"])


def test_all_actor_protocol_trains_without_a_test_set(processor, tmp_path):
    """protocol "all" (the release model): every usable clip is train or dev, dev actors are held out whole, and
    the run calibrates on dev and writes no test predictions."""
    import json as _json
    from emotion_decider.train import load_rows, splits
    cfg = smoke_cfg()
    cfg["data"]["protocol"] = "all"
    rows = load_rows(cfg["data"]["root"])
    sp = splits(cfg, 0, rows)
    exclude = set(_json.load(open(Path(CONFIG).parents[0] / "splits" / "crema_d_exclude.json"))["clips"])
    assert sorted(sp["train"] + sp["dev"]) == sorted(c for c in rows if c not in exclude) and sp["test"] == []
    dev_actors = {rows[c]["actor"] for c in sp["dev"]}
    assert len(dev_actors) == cfg["data"]["dev_actors"] and not dev_actors & {rows[c]["actor"] for c in sp["train"]}
    out = tmp_path / "all"
    res = train(cfg, 0, out, model=tiny_model(processor), processor=processor, max_steps=1, limit_clips=4,
                eval_limit=2, log=lambda *_: None)
    assert "test" not in res and "dev" in res and (out / "calibration.json").exists()
    assert not (out / "predictions_test.jsonl").exists()
