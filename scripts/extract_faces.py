"""Extract 16 face crops per CREMA-D clip (MTCNN), resumable, gentle on a shared machine.

Writes data/crema-d/derived/faces/<clip>/00.jpg ... 15.jpg and meta.json (frame indices, the frame window the
WAV covers, effective fps of the sampled sequence, raw boxes and detection probabilities, the crop track).
Frames are sampled only from the part of the video the WAV covers (scripts/audio_offsets.py), so audio and
frames span the same time. Clips already done are skipped.
Each worker is single-threaded; run under `nice -n 19` so other jobs keep priority.

Usage: nice -n 19 python scripts/extract_faces.py [--workers 4] [--limit N] [--clips ID ...]
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ROOT = Path("data/crema-d")
OUT = ROOT / "derived" / "faces"
K, SIZE = 16, 224
MP3_DELAY = 0.052   # median WAV offset against the FLV's MP3 track (encoder priming): true offset ~0
OFFSETS = ROOT / "derived" / "audio_offsets.jsonl"   # scripts/audio_offsets.py; optional
_offsets = None


def audio_window(clip, fps, n_frames):
    """Frame range [start, stop) covered by the clip's WAV, so frames and audio span the same time (TMRoPE puts
    both on one timeline from t=0). Without offsets, or where the WAV doesn't match the video, the whole clip."""
    global _offsets
    if _offsets is None:
        _offsets = {}
        if OFFSETS.exists():
            for line in open(OFFSETS):
                r = json.loads(line)
                _offsets[r["clip"]] = r
    r = _offsets.get(clip)
    if r is None or r["offset_s"] is None or r["corr"] < 0.5:
        return 0, n_frames, 0.0
    t0 = r["offset_s"] - MP3_DELAY
    t0 = 0.0 if abs(t0) < 0.06 else t0          # within encoder-delay jitter: on the timeline
    start = max(0, round(t0 * fps))
    stop = min(n_frames, round((t0 + r["wav_s"]) * fps))
    return (start, stop, t0) if stop - start >= 2 else (0, n_frames, 0.0)
_mtcnn = {}


def detector(min_face):
    """MTCNN at a minimum face size. 80 px skips the pyramid's smallest scales (CREMA-D faces are ~100+ px):
    2.3x faster than 40 px, crops within ~4 px. Clips with a missed frame are retried at 40 px."""
    if min_face not in _mtcnn:
        import torch
        from facenet_pytorch import MTCNN
        torch.set_num_threads(1)
        _mtcnn[min_face] = MTCNN(keep_all=True, device="cpu", min_face_size=min_face)
    return _mtcnn[min_face]


def process(clip):
    from PIL import Image
    from emotion_decider.faces import crop, decode, fill_missing, sample_indices, square_track
    done = OUT / clip / "meta.json"
    if done.exists():
        return clip, "skip"
    frames, fps = decode(ROOT / "VideoFlash" / f"{clip}.flv")
    start, stop, t0 = audio_window(clip, fps, len(frames))
    idx = sample_indices(len(frames), K, start, stop)
    picked = frames[idx]
    images = [Image.fromarray(f) for f in picked]
    boxes, probs = detector(80).detect(images)
    if any(b is None for b in boxes):
        boxes, probs = detector(40).detect(images)
    raw, conf = [], []
    for b, p in zip(boxes, probs):
        if b is None:
            raw.append([float("nan")] * 4); conf.append(None)
        else:   # several faces: keep the largest (the actor; CREMA-D has one person per clip)
            j = max(range(len(b)), key=lambda i: (b[i][2] - b[i][0]) * (b[i][3] - b[i][1]))
            raw.append([float(v) for v in b[j]]); conf.append(float(p[j]))
    filled = fill_missing(raw)
    status = "ok" if all(c is not None for c in conf) else "partial"
    if filled is None:   # no face in any frame: centre crop of the frame's short side, flagged
        h, w = picked.shape[1:3]
        side = min(h, w)
        track = [[w / 2, h / 2, side]] * K
        status = "no_face"
    else:
        track = square_track(filled).tolist()
    tmp = OUT / f".{clip}.tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    for i, (f, (cx, cy, side)) in enumerate(zip(picked, track)):
        Image.fromarray(crop(f, cx, cy, side, SIZE)).save(tmp / f"{i:02d}.jpg", quality=95)
    meta = {"clip": clip, "n_frames": len(frames), "src_fps": fps, "indices": idx.tolist(),
            "window": [start, stop], "audio_offset_s": t0,
            "sample_fps": fps * K / (stop - start), "duration": len(frames) / fps, "size": SIZE,
            "boxes": raw, "probs": conf, "track": track, "status": status}
    (tmp / "meta.json").write_text(json.dumps(meta))
    tmp.rename(OUT / clip)   # atomic: a clip dir exists only when complete
    return clip, status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--clips", nargs="*")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    clips = a.clips or sorted(p.stem for p in (ROOT / "VideoFlash").glob("*.flv"))
    clips = clips[: a.limit] if a.limit else clips
    counts, t0 = {}, time.time()
    with Pool(a.workers) as pool:
        for n, (clip, status) in enumerate(pool.imap_unordered(process, clips, chunksize=4), 1):
            counts[status] = counts.get(status, 0) + 1
            if status in ("partial", "no_face"):
                print(f"{status}: {clip}", flush=True)
            if n % 250 == 0 or n == len(clips):
                rate = n / (time.time() - t0)
                print(f"{n}/{len(clips)} {counts} {rate:.1f} clips/s, ~{(len(clips) - n) / rate / 60:.0f} min left", flush=True)


if __name__ == "__main__":
    main()
