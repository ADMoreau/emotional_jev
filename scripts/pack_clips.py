"""Pack each clip's face crops, WAV and meta.json into one file: data/crema-d/derived/packed/<clip>.npz.

One read per clip instead of 18 (16 JPEGs, meta.json, WAV); first reads on a Modal Volume are slow. The bytes are
stored unchanged, so a packed clip decodes to exactly what the separate files do. Idempotent (existing files are
kept); writes are atomic.

Usage: python scripts/pack_clips.py [--workers 8]
"""
import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from facedecide.dataset import pack  # noqa: E402

ROOT = Path("data/crema-d")
OUT = ROOT / "derived" / "packed"


def work(clip):
    dst = OUT / f"{clip}.npz"
    if dst.exists():
        return 0
    tmp = OUT / f".{clip}.tmp.npz"
    np.savez(tmp, **pack(clip, ROOT))
    os.replace(tmp, dst)
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    clips = sorted(p.name for p in (ROOT / "derived" / "faces").iterdir() if not p.name.startswith("."))
    with Pool(a.workers) as pool:
        written = sum(pool.imap_unordered(work, clips, chunksize=32))
    print(f"packed {written} new of {len(clips)} clips -> {OUT}")


if __name__ == "__main__":
    main()
