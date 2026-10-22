"""Load a CREMA-D clip's model inputs: the cached face crops and the WAV, per condition.

Two layouts hold the same bytes: the extraction output (derived/faces/<clip>/00..15.jpg + meta.json, and
AudioWAV/<clip>.wav: 18 files per clip) and a packed one (derived/packed/<clip>.npz: one file per clip, written by
scripts/pack_clips.py). On a network filesystem (Modal Volumes) the first read of each file is slow, so the packed
layout is used when present; both decode to identical arrays (tests/test_data.py).
"""
import io
import json
import wave
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("data/crema-d")


def _audio(wav_bytes, clip):
    w = wave.open(io.BytesIO(wav_bytes))
    if w.getframerate() != 16000 or w.getnchannels() != 1:
        raise ValueError(f"{clip}: expected 16 kHz mono")
    return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768


def pack(clip, root=ROOT):
    """-> dict of uint8/int64 arrays holding the clip's JPEG crops, WAV and meta.json bytes, as stored."""
    d = Path(root) / "derived" / "faces" / clip
    meta = (d / "meta.json").read_bytes()
    jpegs = [(d / f"{i:02d}.jpg").read_bytes() for i in range(len(json.loads(meta)["indices"]))]
    return {"jpegs": np.frombuffer(b"".join(jpegs), np.uint8),
            "jpeg_ends": np.cumsum([len(j) for j in jpegs]).astype(np.int64),
            "wav": np.frombuffer((Path(root) / "AudioWAV" / f"{clip}.wav").read_bytes(), np.uint8),
            "meta": np.frombuffer(meta, np.uint8)}


def load_media(clip, condition, root=ROOT):
    """-> (frames [16, 224, 224, 3] uint8 | None, fps | None, audio float32 16 kHz | None).
    av: both; face: frames only; voice: audio only. Frames were sampled from the span the WAV covers, so the
    two start together; fps is the rate of the sampled frames (what the processor needs to place them)."""
    packed = Path(root) / "derived" / "packed" / f"{clip}.npz"
    frames = fps = audio = None
    if packed.exists():
        z = np.load(packed)
        if condition in ("av", "face"):
            buf, ends = z["jpegs"].tobytes(), z["jpeg_ends"]
            starts = np.concatenate([[0], ends[:-1]])
            frames = np.stack([np.asarray(Image.open(io.BytesIO(buf[s:e]))) for s, e in zip(starts, ends)])
            fps = json.loads(z["meta"].tobytes())["sample_fps"]
        if condition in ("av", "voice"):
            audio = _audio(z["wav"].tobytes(), clip)
        return frames, fps, audio
    if condition in ("av", "face"):
        d = Path(root) / "derived" / "faces" / clip
        meta = json.loads((d / "meta.json").read_text())
        frames = np.stack([np.asarray(Image.open(d / f"{i:02d}.jpg")) for i in range(len(meta["indices"]))])
        fps = meta["sample_fps"]
    if condition in ("av", "voice"):
        audio = _audio((Path(root) / "AudioWAV" / f"{clip}.wav").read_bytes(), clip)
    return frames, fps, audio
