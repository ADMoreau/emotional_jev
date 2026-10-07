"""Audit CREMA-D media after face extraction: find clips whose distributed media can't match their labels.

Checks per clip: audio RMS (silent / near-silent WAV), audio vs video duration, black sampled frames, face
detection status and minimum confidence. Writes data/crema-d/derived/audit.jsonl (one row per clip) and
configs/splits/crema_d_exclude.json (IDs + reason; committed, IDs only).

Also uses data/crema-d/derived/audio_offsets.jsonl (scripts/audio_offsets.py): where the WAV sits on the video
timeline and how well it matches the FLV's own audio track.

Exclusion rule: silent audio; no face in any frame; more than half the sampled frames black; the WAV doesn't
match its video (correlation < 0.5); the WAV starts before the video (> 0.1 s: frames can't cover its start);
or the WAV is byte-identical to another clip's (one recording under two labels; both excluded). Clips with a
few black frames stay in (flagged); the model sees what the clip contains.

Usage: nice -n 19 python scripts/audit_crema.py
"""
import hashlib
import json
import wave
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("data/crema-d")
FACES = ROOT / "derived" / "faces"
REPO = Path(__file__).resolve().parents[1]
SILENT_RMS, QUIET_RMS, BLACK_MEAN = 1.0, 50.0, 10.0


def audit(clip):
    w = wave.open(str(ROOT / "AudioWAV" / f"{clip}.wav"))
    a = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float64)
    meta = json.loads((FACES / clip / "meta.json").read_text())
    means = [float(np.asarray(Image.open(FACES / clip / f"{i:02d}.jpg")).mean()) for i in range(len(meta["indices"]))]
    probs = [p for p in meta["probs"] if p is not None]
    return {"clip": clip, "audio_s": len(a) / w.getframerate(), "video_s": meta["duration"],
            "audio_rms": float(np.sqrt((a ** 2).mean())) if len(a) else 0.0,
            "black_frames": [i for i, m in enumerate(means) if m < BLACK_MEAN],
            "face_status": meta["status"], "min_face_prob": min(probs) if probs else None}


def reasons(r, offsets, duplicate_wavs):
    out = []
    o = offsets.get(r["clip"])
    if o is not None and o["offset_s"] is not None and o["corr"] >= 0.5 and o["offset_s"] - 0.052 < -0.1:
        out.append("audio starts before the video")
    if o is not None and o["corr"] < 0.5 and r["audio_rms"] >= SILENT_RMS:
        out.append(f"WAV doesn't match the video (corr {o['corr']:.2f})")
    if r["clip"] in duplicate_wavs:
        out.append("WAV identical to " + ", ".join(sorted(duplicate_wavs[r["clip"]])))
    if r["audio_rms"] < SILENT_RMS:
        out.append("silent audio")
    if r["face_status"] == "no_face":
        out.append("no face in any frame")
    if len(r["black_frames"]) > 8:
        out.append("most sampled frames black")
    return out


def main():
    clips = sorted(p.name for p in FACES.iterdir() if not p.name.startswith("."))
    rows = [audit(c) for c in clips]
    with open(ROOT / "derived" / "audit.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    offsets = {}
    path = ROOT / "derived" / "audio_offsets.jsonl"
    if path.exists():
        offsets = {r["clip"]: r for r in map(json.loads, open(path))}
    by_hash = defaultdict(list)
    for c in clips:
        by_hash[hashlib.sha256((ROOT / "AudioWAV" / f"{c}.wav").read_bytes()).hexdigest()].append(c)
    duplicate_wavs = {c: set(g) - {c} for g in by_hash.values() if len(g) > 1 for c in g}
    exclude = {r["clip"]: why for r in rows if (why := reasons(r, offsets, duplicate_wavs))}
    json.dump({"rule": "silent audio; no face in any frame; > 8 of 16 sampled frames black; WAV doesn't match "
                       "the video (corr < 0.5); WAV starts > 0.1 s before the video; WAV byte-identical to "
                       "another clip's",
               "clips": exclude}, open(REPO / "configs" / "splits" / "crema_d_exclude.json", "w"), indent=1)

    rms = np.array([r["audio_rms"] for r in rows])
    dd = np.array([r["audio_s"] - r["video_s"] for r in rows])
    print(f"clips audited: {len(rows)}")
    print(f"audio rms: silent(<{SILENT_RMS}) {int((rms < SILENT_RMS).sum())}, quiet(<{QUIET_RMS}) {int((rms < QUIET_RMS).sum())}, "
          f"median {np.median(rms):.0f}")
    print(f"audio - video duration (s): median {np.median(dd):+.3f}, |diff| > 0.25 s: {int((np.abs(dd) > 0.25).sum())}, "
          f"min {dd.min():+.2f}, max {dd.max():+.2f}")
    print(f"clips with black sampled frames: {sum(bool(r['black_frames']) for r in rows)}",
          [(r["clip"], r["black_frames"]) for r in rows if r["black_frames"]][:10])
    lo = sorted((r["min_face_prob"], r["clip"]) for r in rows if r["min_face_prob"] is not None)[:5]
    print("lowest face confidence:", [(c, round(p, 3)) for p, c in lo])
    print(f"excluded: {len(exclude)}")
    for c, why in exclude.items():
        print(f"  {c}: {'; '.join(why)}")


if __name__ == "__main__":
    main()
