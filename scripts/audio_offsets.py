"""Measure where each CREMA-D WAV sits on its video's timeline.

The FLV carries its own (MP3) audio track, which is on the video timeline. Cross-correlating the WAV with it
gives the WAV's start offset in seconds (positive: the WAV starts that long after the video does) and the
peak normalized correlation (low: the WAV doesn't match this video at all). Writes
data/crema-d/derived/audio_offsets.jsonl.

Usage: nice -n 19 python scripts/audio_offsets.py [--workers 4]
"""
import argparse
import json
import subprocess
import wave
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path("data/crema-d")
SR = 4000   # envelope-level alignment; 0.25 ms resolution is plenty against a 40 ms audio token


def flv_audio(path):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-threads", "1", "-i", str(path), "-vn", "-ac", "1", "-ar", str(SR),
                          "-f", "s16le", "-"], check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.int16).astype(np.float64)


def wav_audio(path):
    w = wave.open(str(path))
    a = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float64)
    step = w.getframerate() // SR
    return a[: len(a) // step * step].reshape(-1, step).mean(1)


def offset(clip):
    v, a = flv_audio(ROOT / "VideoFlash" / f"{clip}.flv"), wav_audio(ROOT / "AudioWAV" / f"{clip}.wav")
    if len(v) == 0 or len(a) == 0 or a.std() == 0 or v.std() == 0:
        return {"clip": clip, "offset_s": None, "corr": 0.0, "flv_audio_s": len(v) / SR, "wav_s": len(a) / SR}
    n = 1 << int(np.ceil(np.log2(len(v) + len(a))))
    xc = np.fft.irfft(np.fft.rfft(v - v.mean(), n) * np.conj(np.fft.rfft(a - a.mean(), n)), n)
    lags = np.concatenate([np.arange(0, len(v)), np.arange(-len(a) + 1, 0)])
    xc = np.concatenate([xc[: len(v)], xc[n - len(a) + 1:]])
    k = int(np.argmax(xc))
    corr = xc[k] / (np.linalg.norm(a - a.mean()) * np.linalg.norm(v - v.mean()))
    return {"clip": clip, "offset_s": lags[k] / SR, "corr": float(corr), "flv_audio_s": len(v) / SR, "wav_s": len(a) / SR}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    clips = sorted(p.stem for p in (ROOT / "VideoFlash").glob("*.flv"))
    with Pool(a.workers) as pool, open(ROOT / "derived" / "audio_offsets.jsonl", "w") as f:
        for r in pool.imap(offset, clips, chunksize=16):
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
