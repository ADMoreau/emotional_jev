"""Frame sampling and face cropping for clips: decode, pick 16 frames, detect, smooth the box, crop.

Standard in audio-visual emotion work (MTCNN or RetinaFace per frame, crops resized to 224). The box is a
square around the detected face: its side is the median over the clip's frames (no scale jitter) and its
centre a 3-frame moving average (follows head motion without shaking).
"""
import json
import subprocess

import numpy as np
from PIL import Image


def probe(path):
    """-> (width, height, fps) of the first video stream."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,r_frame_rate", "-of", "json", str(path)],
                         check=True, capture_output=True, text=True).stdout
    s = json.loads(out)["streams"][0]
    num, den = map(int, s["r_frame_rate"].split("/"))
    return s["width"], s["height"], num / den


def decode(path):
    """-> (frames [N, H, W, 3] uint8 RGB, fps). Every frame, at native size."""
    w, h, fps = probe(path)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-threads", "1", "-i", str(path), "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3), fps


def sample_indices(n_frames, k=16, start=0, stop=None):
    """k frame indices at the centres of k equal segments of frames [start, stop) (default: the whole clip)."""
    stop = n_frames if stop is None else min(stop, n_frames)
    start = min(max(0, start), stop - 1)
    return (start + np.floor((np.arange(k) + 0.5) * (stop - start) / k)).astype(int).clip(start, stop - 1)


def fill_missing(boxes):
    """Boxes [k, 4] with NaN rows where detection failed -> linear interpolation over time (edges held)."""
    boxes = np.array(boxes, dtype=float)
    ok = ~np.isnan(boxes).any(1)
    if not ok.any():
        return None
    t = np.arange(len(boxes))
    return np.stack([np.interp(t, t[ok], boxes[ok, j]) for j in range(4)], 1)


def square_track(boxes, margin=1.3, smooth=3):
    """Per-frame square crops (cx, cy, side) from [k, 4] x1y1x2y2 boxes: median side x margin, smoothed centre."""
    cx, cy = (boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2
    side = float(np.median(np.maximum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]))) * margin
    kernel = np.ones(smooth) / smooth
    pad = smooth // 2
    sm = lambda x: np.convolve(np.pad(x, pad, mode="edge"), kernel, mode="valid")
    return np.stack([sm(cx), sm(cy), np.full(len(cx), side)], 1)


def crop(frame, cx, cy, side, out=224):
    """Square crop centred at (cx, cy), edge-padded where it leaves the frame, resized to out x out."""
    h, w = frame.shape[:2]
    half = side / 2
    x0, y0, x1, y1 = (int(round(v)) for v in (cx - half, cy - half, cx + half, cy + half))
    pad = max(0, -x0, -y0, x1 - w, y1 - h)
    if pad:
        frame = np.pad(frame, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
        x0, y0, x1, y1 = x0 + pad, y0 + pad, x1 + pad, y1 + pad
    return np.asarray(Image.fromarray(frame[y0:y1, x0:x1]).resize((out, out), Image.BICUBIC))
