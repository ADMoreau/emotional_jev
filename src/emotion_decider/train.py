"""Train, calibrate and evaluate the decision model on CREMA-D.

    python -m emotion_decider.train --config configs/omni_3b.yaml --fold 0 --out runs/omni3b-f0 [--max_steps N]

Splits (always without the exclusion list): protocol "5fold" tests on one actor-disjoint fold and holds out
`dev_actors` of the remaining actors for temperature fitting; protocol "emobox" uses EmoBox fold 1 (random by
clip, not actor-disjoint) with a random `emobox_dev_frac` of its train clips as dev.

Each epoch visits every training clip once. Variant A shows it audio-visually; variant B shows a random condition
(av / face / voice) with that condition's crowd votes as the perceived target. Questions (phrasing, option order,
Noul/Score presence) are redrawn per epoch. Loss: soft-label cross-entropy per question, averaged per clip.
Every `dev_every` steps a fixed probe of dev clips (audio-visual) is scored and its loss logged.

Examples are prepared (media read, processor, questions) in DataLoader worker processes, `data.num_workers` of
them, so preprocessing overlaps the model. Each example is a pure function of its (clip, seed string), so runs
are deterministic and resumable whatever the worker count.

After training: per-question temperatures are fitted on dev (calibration.json), then dev and test are scored
with every question type in every condition (metrics.json, predictions_<split>.jsonl). Checkpoints (adapter +
head + optimizer + position) are written every `ckpt_every` optimizer steps; rerunning the same command resumes.
"""
import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from . import metrics as M
from .crema import EMOTIONS
from .augment import augment_media
from .dataset import load_media
from .omni import encode_clip
from .questions import CONDITIONS, NAMES, make_questions, option_keys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")   # tokenizers + forked workers
REPO = Path(__file__).resolve().parents[2]
CANONICAL = {"intended": NAMES, "perceived": NAMES, "noul": ["true", "false"], "intensity": ["0", "1", "2", "3"]}


# --- data

def load_rows(root):
    return {r["clip_id"]: r for r in map(json.loads, open(Path(root) / "derived" / "targets.jsonl"))}


def splits(cfg, fold, rows):
    """-> {"train": [...], "dev": [...], "test": [...]} clip IDs."""
    sp = REPO / "configs" / "splits"
    exclude = set(json.load(open(sp / "crema_d_exclude.json"))["clips"])
    usable = sorted(c for c in rows if c not in exclude)
    rng = random.Random(f"dev/{cfg['seed']}/{cfg['data']['protocol']}/{fold}")
    if cfg["data"]["protocol"] == "5fold":
        folds = json.load(open(sp / "crema_d_5fold.json"))["folds"]
        rest = sorted(a for k, f in enumerate(folds) if k != fold for a in f)
        dev_actors = set(rng.sample(rest, cfg["data"]["dev_actors"]))
        test_actors = set(folds[fold])
        return {"train": [c for c in usable if rows[c]["actor"] not in test_actors | dev_actors],
                "dev": [c for c in usable if rows[c]["actor"] in dev_actors],
                "test": [c for c in usable if rows[c]["actor"] in test_actors]}
    if cfg["data"]["protocol"] == "emobox":
        eb = json.load(open(sp / "crema_d_emobox_fold1.json"))
        train = [c for c in eb["train"] if c not in exclude]
        dev = set(rng.sample(train, round(len(train) * cfg["data"]["emobox_dev_frac"])))
        return {"train": [c for c in train if c not in dev], "dev": sorted(dev),
                "test": [c for c in eb["test"] if c not in exclude]}
    raise ValueError(cfg["data"]["protocol"])


def spread(clips, n, key):
    """A seeded sample of n clips across the whole split (sorted). Clip IDs sort by actor, so the first n would be
    one or two actors; a sample spans them."""
    return clips if n >= len(clips) else sorted(random.Random(key).sample(clips, n))


def example(fmt, processor, row, condition, rng, root, augment=None, aug_rng=None, **qkw):
    """-> (clip, names, questions, targets) for one clip shown under `condition`. fmt: anything with the clip
    state's text `prefix` and `suffix` (a scorer). augment: the config's augment section (training only), drawn
    from aug_rng so the question draws from `rng` are the same with or without it."""
    frames, fps, audio = load_media(row["clip_id"], condition, root)
    if augment:
        frames, fps, audio = augment_media(frames, fps, audio, aug_rng, augment)
    clip = encode_clip(processor, frames=frames, fps=fps, audio=audio, prefix=fmt.prefix, suffix=fmt.suffix)
    qs, ts = make_questions(row, condition, rng, **qkw)
    names = list(qs)
    return clip, names, [qs[n] for n in names], [ts[n]["probabilities"] for n in names]


class Format:
    def __init__(self, prefix, suffix):
        self.prefix, self.suffix = prefix, suffix


class Examples(torch.utils.data.Dataset):
    """items: (clip_id, seed string, condition or None). None draws the condition from the seeded RNG first
    (variant B; variant A always shows av), then the questions, exactly as a single-process loop would."""

    def __init__(self, items, rows, processor, fmt, root, variant="B", augment=None, **qkw):
        self.items, self.rows, self.processor, self.root, self.variant, self.qkw = items, rows, processor, root, variant, qkw
        self.augment = augment
        self.fmt = Format(fmt.prefix, fmt.suffix)   # not the model: workers must not pickle it

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        clip_id, seed, cond = self.items[i]
        rng = random.Random(seed)
        if cond is None:
            cond = "av" if self.variant == "A" else rng.choice(CONDITIONS)
        aug_rng = random.Random(f"{seed}/augment") if self.augment else None
        clip, names, qs, ts = example(self.fmt, self.processor, self.rows[clip_id], cond, rng, self.root,
                                      augment=self.augment, aug_rng=aug_rng, **self.qkw)
        return {"clip_id": clip_id, "condition": cond, "clip": clip, "names": names, "questions": qs, "targets": ts}


def _worker_init(_):
    torch.set_num_threads(1)


def loader(dataset, num_workers):
    return torch.utils.data.DataLoader(dataset, batch_size=None, shuffle=False, num_workers=num_workers,
                                       prefetch_factor=4 if num_workers else None, worker_init_fn=_worker_init)


def soft_ce(logits, target):
    return -(target * F.log_softmax(logits, -1)).sum()


# --- evaluation (shared with the zero-shot baseline)

@torch.no_grad()
def predict(scorer, processor, rows, clips, root, temperatures=None, conditions=CONDITIONS, num_workers=0,
            log_every=200, quiet=False, batch_size=1):
    """Every question type in each condition, deterministic phrasing/order per (clip, condition), `batch_size`
    examples per forward pass. -> list of {clip, condition, question, keys, logits, probs, target} (probs after
    temperature)."""
    items = [(c, f"eval/{c}/{cond}", cond) for c in clips for cond in conditions]
    data = loader(Examples(items, rows, processor, scorer, root, p_noul=1.0, p_score=1.0), num_workers)
    out, t0, t_wait, t_model, it, done = [], time.time(), 0.0, 0.0, iter(data), 0
    for start in range(0, len(items), batch_size):
        t = time.time()
        batch = [next(it) for _ in range(min(batch_size, len(items) - start))]
        t_wait += time.time() - t; t = time.time()
        logit_lists = scorer.logits_batch([ex["clip"] for ex in batch], [ex["questions"] for ex in batch])
        logit_lists = [[z.float().cpu().numpy() for z in zs] for zs in logit_lists]   # .cpu() syncs: model time is real
        t_model += time.time() - t
        for ex, logits in zip(batch, logit_lists):
            for name, q, z, y in zip(ex["names"], ex["questions"], logits, ex["targets"]):
                temp = (temperatures or {}).get(name, 1.0)
                out.append({"clip": ex["clip_id"], "condition": ex["condition"], "question": name,
                            "keys": option_keys(q), "logits": z.tolist(), "probs": M.softmax(z, temp).tolist(),
                            "target": list(y.values())})
        before, done = done, (start + len(batch)) // len(conditions)
        if log_every and done // log_every > before // log_every:
            print(f"  scored {done}/{len(clips)} clips ({done / (time.time() - t0):.2f} clips/s)", flush=True)
    total = time.time() - t0
    if not quiet:
        print(f"  scored {len(clips)} clips x {len(conditions)} conditions in {total:.0f}s "
              f"({len(clips) / max(total, 1e-9):.2f} clips/s; waiting for data {t_wait:.0f}s, model {t_model:.0f}s)",
              flush=True)
    return out


def fit_temperatures(preds):
    return {name: M.fit_temperature([p["logits"] for p in preds if p["question"] == name],
                                    [p["target"] for p in preds if p["question"] == name])
            for name in sorted({p["question"] for p in preds})}


def canonical(pred):
    """(probs, target) reordered to the question's canonical option order."""
    order = [pred["keys"].index(k) for k in CANONICAL[pred["question"]]]
    return np.array(pred["probs"])[order], np.array(pred["target"])[order]


def summarize(preds, rows):
    """{question: {condition: {metric: value}}}."""
    out = {}
    for name in sorted({p["question"] for p in preds}):
        out[name] = {}
        for cond in CONDITIONS:
            sel = [p for p in preds if p["question"] == name and p["condition"] == cond]
            if not sel:
                continue
            P, T = map(np.array, zip(*map(canonical, sel)))
            m = {"n": len(sel), "nll": M.nll(P, T)}
            if name == "intended":
                y = T.argmax(1)
                m.update(acc=M.accuracy(P, y), uar=M.uar(P, y), f1_macro=M.f1(P, y, "macro"),
                         f1_micro=M.f1(P, y, "micro"), ece=M.ece(P, y), aurc=M.aurc(P, y))
            elif name == "perceived":
                majority = T.argmax(1)
                intended = np.array([NAMES.index(EMOTIONS[rows[p["clip"]]["intended"]]) for p in sel])
                m.update(kl=M.kl(T, P), ece_vs_majority=M.ece(P, majority), agree_majority=M.accuracy(P, majority),
                         acc_vs_intended=M.accuracy(P, intended))
            elif name == "noul":
                m.update(brier=float(((P[:, 0] - T[:, 0]) ** 2).mean()))
            elif name == "intensity":
                lv = np.arange(P.shape[1])
                m.update(kl=M.kl(T, P), score_mae=float(np.abs(P @ lv - T @ lv).mean()))
            out[name][cond] = m
    return out


def probe_metrics(preds):
    """Mean soft-target loss over all questions (T=1) and intended accuracy: the in-training dev check."""
    loss = np.mean([-(np.array(p["target"]) * np.log(M.softmax(np.array(p["logits"])) + M.EPS)).sum() for p in preds])
    intended = [p for p in preds if p["question"] == "intended"]
    acc = np.mean([np.argmax(p["logits"]) == np.argmax(p["target"]) for p in intended]) if intended else float("nan")
    return float(loss), float(acc)


# --- training

def load_config(path, overrides=()):
    cfg = yaml.safe_load(open(path))
    for o in overrides:   # dotted.key=value (value parsed as YAML)
        k, v = o.split("=", 1)
        node = cfg
        *parents, leaf = k.split(".")
        for p in parents:
            node = node[p]
        node[leaf] = yaml.safe_load(v)
    return cfg


def lr_factor(step, total, warmup):
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup))))


def train(cfg, fold, out, model=None, processor=None, max_steps=None, limit_clips=None, eval_limit=None, log=print,
          on_checkpoint=None):
    """limit_clips: first N clips of every split (smoke tests). eval_limit: first N dev and test clips for the
    final calibration and scoring only (short training checks). on_checkpoint: called after every checkpoint
    write (Modal: commit the runs Volume so a preempted job resumes from it)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    root, t = cfg["data"]["root"], cfg["train"]
    workers = cfg["data"].get("num_workers", 0)
    rows = load_rows(root)
    sp = splits(cfg, fold, rows)
    if limit_clips:
        sp = {k: v[:limit_clips] for k, v in sp.items()}
    (out / "config.json").write_text(json.dumps({"config": cfg, "fold": fold, "sizes": {k: len(v) for k, v in sp.items()}}, indent=1))
    if processor is None:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(cfg["backbone"]["name"], revision=cfg["backbone"].get("revision"))
    if model is None:
        from .decision import DecisionModel
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = DecisionModel.from_pretrained(
            cfg["backbone"]["name"], processor.tokenizer, revision=cfg["backbone"].get("revision"),
            dtype=getattr(torch, cfg["backbone"]["dtype"]), device=device, attn=cfg["backbone"].get("attn", "sdpa"),
            lora=cfg["lora"], pointer_dim=cfg["head"]["pointer_dim"],
            gradient_checkpointing=cfg["backbone"].get("gradient_checkpointing", False))
    head_params = list(model.head.parameters())
    other = [(n, p) for n, p in model.named_parameters() if p.requires_grad and all(p is not h for h in head_params)]
    encoder = [p for n, p in other if n.startswith(("thinker.visual.", "thinker.audio_tower."))]   # encoder LoRA
    groups = [{"params": [p for n, p in other if not n.startswith(("thinker.visual.", "thinker.audio_tower."))],
               "lr": t["lr_lora"]}, {"params": head_params, "lr": t["lr_head"]}]
    if encoder:
        groups.append({"params": encoder, "lr": t.get("lr_encoder") or t["lr_lora"]})
    opt = torch.optim.AdamW(groups, weight_decay=t["weight_decay"])
    augment = cfg["augment"] if cfg.get("augment", {}).get("enabled") else None
    base_lrs = [g["lr"] for g in opt.param_groups]
    steps_per_epoch = math.ceil(len(sp["train"]) / t["grad_accum"])
    total = max_steps or steps_per_epoch * t["epochs"]
    warmup = max(1, round(total * t["warmup_frac"]), min(t.get("warmup_min_steps", 0), total // 4))
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if model.device.type == "cuda" else torch.autocast("cpu", enabled=False)
    probe_clips = spread(sp["dev"], t.get("dev_probe_clips", 0), f"probe/{cfg['seed']}/{fold}") if t.get("dev_probe_clips") else []
    per_forward = max(1, t.get("clips_per_forward", 1))   # clips batched into one forward pass (rows; see omni)
    eval_batch = max(1, t.get("eval_batch", 1))

    ckpt, state = out / "checkpoint.pt", {"step": 0, "epoch": 0, "index": 0}
    if ckpt.exists():
        saved = torch.load(ckpt, weights_only=False)
        model.load_state(saved["model"]); opt.load_state_dict(saved["optimizer"]); state = saved["position"]
        log(f"resumed at step {state['step']} (epoch {state['epoch']}, clip {state['index']})")

    def save_checkpoint():
        tmp = ckpt.with_suffix(".tmp")
        torch.save({"model": model.state(), "optimizer": opt.state_dict(), "position": dict(state)}, tmp)
        tmp.replace(ckpt)
        if on_checkpoint:
            on_checkpoint()

    log_f = open(out / "train_log.jsonl", "a")
    probe_conditions = tuple(t.get("dev_probe_conditions", ["av"]))

    def dev_probe():
        """Score the probe; the best probe loss so far is kept in best/ (selected for calibration and test)."""
        model.eval()
        loss, acc = probe_metrics(predict(model, processor, rows, probe_clips, root, conditions=probe_conditions,
                                          num_workers=workers, log_every=0, quiet=True, batch_size=eval_batch))
        model.train()
        best = state.get("best")
        improved = best is None or loss < best["loss"]
        if improved and t.get("keep_best"):
            state["best"] = {"loss": loss, "step": state["step"]}
            model.save(out / "best", meta={"backbone": cfg["backbone"], "fold": fold, **state["best"]})
            save_checkpoint()   # record the new best now: a resume must not overwrite best/ with a worse one
        log_f.write(json.dumps({"step": state["step"], "dev_loss": loss, "dev_intended_acc": acc}) + "\n"); log_f.flush()
        log(f"dev probe at step {state['step']}: loss {loss:.4f}, intended acc {acc:.3f} "
            f"({len(probe_clips)} clips x {'/'.join(probe_conditions)}){' *best*' if improved and t.get('keep_best') else ''}")

    model.train()
    if probe_clips and state["step"] == 0:
        dev_probe()
    t0, running, wait_s = time.time(), [], 0.0
    while state["step"] < total and state["epoch"] < t["epochs"]:
        order = sp["train"][:]
        random.Random(f"order/{cfg['seed']}/{state['epoch']}").shuffle(order)
        items = [(c, f"train/{cfg['seed']}/{state['epoch']}/{c}", None) for c in order[state["index"]:]]
        it = iter(loader(Examples(items, rows, processor, model, root, variant=t["variant"], augment=augment,
                                  p_noul=t["p_noul"], p_score=t["p_score"]), workers))
        while state["index"] < len(order) and state["step"] < total:
            n = min(t["grad_accum"], len(order) - state["index"])
            for start in range(0, n, per_forward):   # n clips per optimizer step, per_forward per forward pass
                tick = time.time()
                batch = [next(it) for _ in range(min(per_forward, n - start))]
                wait_s += time.time() - tick
                with autocast:
                    logit_lists = model.logits_batch([ex["clip"] for ex in batch], [ex["questions"] for ex in batch])
                losses = [sum(soft_ce(z, torch.tensor(list(y.values()), device=z.device))
                              for z, y in zip(logits, ex["targets"])) / len(logits)
                          for ex, logits in zip(batch, logit_lists)]   # per clip: mean over its questions
                (torch.stack(losses).sum() / n).backward()
                running += [float(l) for l in torch.stack(losses).detach().cpu()]   # syncs: data wait vs wall is real
            for g, base in zip(opt.param_groups, base_lrs):
                g["lr"] = base * lr_factor(state["step"], total, warmup)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), t["max_grad_norm"])
            opt.step(); opt.zero_grad(set_to_none=True)
            state["step"] += 1; state["index"] += n
            if state["step"] % t["log_every"] == 0 or state["step"] == total:
                wall = time.time() - t0
                rec = {"step": state["step"], "epoch": state["epoch"], "loss": float(np.mean(running)),
                       "grad_norm": float(grad_norm), "lr": opt.param_groups[0]["lr"],
                       "clips_per_s": len(running) / wall, "data_wait_frac": wait_s / wall}
                log_f.write(json.dumps(rec) + "\n"); log_f.flush()
                log(f"step {rec['step']}/{total} loss {rec['loss']:.4f} gnorm {rec['grad_norm']:.2f} lr {rec['lr']:.2e} "
                    f"{rec['clips_per_s']:.2f} clips/s (waiting for data {100 * rec['data_wait_frac']:.0f}%)")
                running, t0, wait_s = [], time.time(), 0.0
            if probe_clips and t.get("dev_every") and state["step"] % t["dev_every"] == 0:
                dev_probe()
                t0, wait_s = time.time(), 0.0
            if state["step"] % t["ckpt_every"] == 0:
                save_checkpoint()
        del it   # stop the epoch's workers (also when max_steps ends it early)
        if state["index"] >= len(order):
            state["epoch"] += 1; state["index"] = 0
    if probe_clips and not (t.get("dev_every") and state["step"] % t["dev_every"] == 0):
        dev_probe()
    save_checkpoint()
    model.save(out / "final", meta={"backbone": cfg["backbone"], "fold": fold, "step": state["step"]})
    selected = state["step"]
    if t.get("keep_best") and state.get("best"):
        model.load_state(torch.load(out / "best" / "decision.pt", weights_only=False))
        selected = state["best"]["step"]
        log(f"calibrating and testing the best dev-probe checkpoint: step {selected} (loss {state['best']['loss']:.4f})")

    model.eval()
    if eval_limit:
        sp = {**sp, "dev": spread(sp["dev"], eval_limit, f"eval/{cfg['seed']}/{fold}/dev"),
              "test": spread(sp["test"], eval_limit, f"eval/{cfg['seed']}/{fold}/test")}
    log("scoring dev for calibration")
    dev = predict(model, processor, rows, sp["dev"], root, num_workers=workers, batch_size=eval_batch)
    temps = fit_temperatures(dev)
    (out / "calibration.json").write_text(json.dumps(temps, indent=1))
    results = {"temperatures": temps, "selected_step": selected, "total_steps": state["step"]}
    for split, preds in (("dev", dev), ("test", None)):
        if preds is None:
            log(f"scoring {split}")
            preds = predict(model, processor, rows, sp[split], root, temperatures=temps, num_workers=workers,
                            batch_size=eval_batch)
        else:   # dev was scored at T=1; re-apply the fitted temperatures for its report
            for p in preds:
                p["probs"] = M.softmax(np.array(p["logits"]), temps.get(p["question"], 1.0)).tolist()
        with open(out / f"predictions_{split}.jsonl", "w") as f:
            for p in preds:
                f.write(json.dumps(p) + "\n")
        results[split] = summarize(preds, rows)
    (out / "metrics.json").write_text(json.dumps(results, indent=1))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_steps", type=int)
    ap.add_argument("--limit_clips", type=int, help="smoke runs: first N clips of each split")
    ap.add_argument("--eval_limit", type=int, help="first N dev/test clips for the final scoring only")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. train.variant=A")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    torch.manual_seed(cfg["seed"])
    res = train(cfg, a.fold, a.out, max_steps=a.max_steps, limit_clips=a.limit_clips, eval_limit=a.eval_limit)
    print(json.dumps({k: v for k, v in res.items() if k != "dev"}, indent=1))


if __name__ == "__main__":
    main()
