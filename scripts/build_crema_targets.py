"""Build CREMA-D targets and splits.

Writes (all under data/, gitignored, except the ID-only split files under configs/splits/):
  data/crema-d/derived/targets.jsonl   one row per clip: actor, sentence, intended label, acted level, and per
                                       modality (av / face / voice) the vote counts, vote distribution and
                                       per-rater [emotion, intensity] pairs
  configs/splits/crema_d_5fold.json    5 actor-disjoint folds (actor IDs only)
  configs/splits/crema_d_emobox_fold1.json   EmoBox fold 1 (clip IDs only), if --emobox_dir is given

Usage: python scripts/build_crema_targets.py [--root data/crema-d] [--emobox_dir DIR]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from emotion_decider.crema import actor_folds, load_ratings, parse_clip, vote_distribution  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/crema-d")
    ap.add_argument("--emobox_dir", help="directory holding EmoBox's cremad_{train,test}_fold_1.jsonl")
    a = ap.parse_args()
    root = Path(a.root)

    ratings = load_ratings(root / "finishedResponses.csv", root / "finishedEmoResponses.csv")
    clips = sorted(p.stem for p in (root / "VideoFlash").glob("*.flv"))
    if set(clips) != set(ratings):
        raise SystemExit(f"clip/rating mismatch: {len(set(clips) ^ set(ratings))} clips differ")
    out = root / "derived"
    out.mkdir(exist_ok=True)
    with open(out / "targets.jsonl", "w") as f:
        for c in clips:
            row = {"clip_id": c, **parse_clip(c), "perceived": {}}
            for m, cell in ratings[c].items():
                row["perceived"][m] = {"votes": cell["votes"], "n": sum(cell["votes"].values()),
                                       "distribution": vote_distribution(cell["votes"]), "ratings": cell["ratings"]}
            f.write(json.dumps(row) + "\n")
    print(f"targets: {len(clips)} clips -> {out / 'targets.jsonl'}")

    splits = REPO / "configs" / "splits"
    splits.mkdir(parents=True, exist_ok=True)
    folds = actor_folds(root / "VideoDemographics.csv", k=5, seed=0)
    json.dump({"source": "emotion_decider.crema.actor_folds(k=5, seed=0)", "unit": "actor", "folds": folds},
              open(splits / "crema_d_5fold.json", "w"), indent=1)
    print("folds (actors):", [len(f) for f in folds], "clips:",
          [sum(parse_clip(c)["actor"] in set(f) for c in clips) for f in folds])

    if a.emobox_dir:
        def ids(split):
            path = Path(a.emobox_dir) / f"cremad_{split}_fold_1.jsonl"
            return sorted("_".join(json.loads(l)["key"].split("-")[1:]) for l in open(path))
        train, test = ids("train"), ids("test")
        shared = {parse_clip(c)["actor"] for c in train} & {parse_clip(c)["actor"] for c in test}
        json.dump({"source": "https://github.com/emo-box/EmoBox data/cremad/fold_1 (used by arXiv 2603.11095)",
                   "note": f"random 75/25 by clip; NOT actor-disjoint ({len(shared)} actors in both)",
                   "unit": "clip", "train": train, "test": test},
                  open(splits / "crema_d_emobox_fold1.json", "w"), indent=1)
        print(f"emobox: train {len(train)}, test {len(test)}, actors in both: {len(shared)}")


if __name__ == "__main__":
    main()
