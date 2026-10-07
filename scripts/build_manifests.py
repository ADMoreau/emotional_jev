"""Write Jev-format question manifests for CREMA-D (one row per usable clip x condition).

Row: {clip_id, actor, fold, emobox, condition, questions: {name: {type, instructions, criteria}},
      targets: {name: {probabilities: {key: p}}}}
Excluded clips (configs/splits/crema_d_exclude.json) are left out. Phrasings and option order come from a seeded
RNG per (seed, clip, condition), so a manifest is reproducible; training redraws them per epoch through
facedecide.questions.make_questions. Output is under data/ (it holds labels).

Usage: python scripts/build_manifests.py [--seed 0]
"""
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from facedecide.questions import CONDITIONS, make_questions  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
ROOT = Path("data/crema-d")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    splits = REPO / "configs" / "splits"
    folds = {actor: k for k, f in enumerate(json.load(open(splits / "crema_d_5fold.json"))["folds"]) for actor in f}
    emobox = json.load(open(splits / "crema_d_emobox_fold1.json"))
    emobox = {**{c: "train" for c in emobox["train"]}, **{c: "test" for c in emobox["test"]}}
    exclude = set(json.load(open(splits / "crema_d_exclude.json"))["clips"])
    out, kinds, n = ROOT / "derived" / f"manifest_seed{a.seed}.jsonl", Counter(), 0
    with open(out, "w") as f:
        for line in open(ROOT / "derived" / "targets.jsonl"):
            row = json.loads(line)
            if row["clip_id"] in exclude:
                continue
            for cond in CONDITIONS:
                rng = random.Random(f"{a.seed}/{row['clip_id']}/{cond}")
                qs, ts = make_questions(row, cond, rng)
                kinds.update(qs.keys())
                f.write(json.dumps({"clip_id": row["clip_id"], "actor": row["actor"], "fold": folds[row["actor"]],
                                    "emobox": emobox[row["clip_id"]], "condition": cond,
                                    "questions": qs, "targets": ts}) + "\n")
                n += 1
    print(f"{n} rows -> {out}; questions: {dict(kinds)} ({sum(kinds.values()) / n:.2f} per row)")


if __name__ == "__main__":
    main()
