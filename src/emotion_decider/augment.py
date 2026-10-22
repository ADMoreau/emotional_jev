"""Training-time augmentation of a clip's face crops and audio (never used for dev, calibration or test).

One draw per clip: every frame gets the same crop, flip and colour change, so the face moves as it did. A speed
change rescales the frame rate by the same factor, so audio and frames still cover the same span (TMRoPE).
`cfg` is the config's `augment` section; `rng` a random.Random seeded per (clip, epoch).
"""
import numpy as np
from PIL import Image, ImageEnhance


def augment_media(frames, fps, audio, rng, cfg):
    """(frames [T, H, W, 3] uint8 | None, fps | None, audio float32 16 kHz | None) -> the same, augmented."""
    speed = rng.uniform(*cfg["speed"]) if audio is not None else 1.0
    if frames is not None:
        frames = _video(frames, rng, cfg)
        fps = fps * speed
    if audio is not None:
        audio = _audio(audio, speed, rng, cfg)
    return frames, fps, audio


def _video(frames, rng, cfg):
    _, h, w, _ = frames.shape
    side = rng.uniform(*cfg["crop_scale"]) * min(h, w)          # square crop, resized back to h x w
    x0, y0 = rng.uniform(0, w - side), rng.uniform(0, h - side)
    flip = rng.random() < cfg["flip"]
    brightness, contrast, saturation = (rng.uniform(1 - cfg["color"], 1 + cfg["color"]) for _ in range(3))
    out = []
    for f in frames:
        im = Image.fromarray(f).resize((w, h), Image.BICUBIC, box=(x0, y0, x0 + side, y0 + side))
        if flip:
            im = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        im = ImageEnhance.Brightness(im).enhance(brightness)
        im = ImageEnhance.Contrast(im).enhance(contrast)
        im = ImageEnhance.Color(im).enhance(saturation)
        out.append(np.asarray(im))
    return np.stack(out)


def _audio(audio, speed, rng, cfg):
    n = int(round(len(audio) / speed))                          # speed > 1: shorter (and higher-pitched)
    audio = np.interp(np.arange(n) * speed, np.arange(len(audio)), audio).astype(np.float32)
    audio = audio * np.float32(10 ** (rng.uniform(-cfg["gain_db"], cfg["gain_db"]) / 20))
    if rng.random() < cfg["noise_p"]:
        power = float(np.mean(audio ** 2)) / 10 ** (rng.uniform(*cfg["noise_snr_db"]) / 10)
        audio = audio + np.random.default_rng(rng.getrandbits(32)).normal(0, np.sqrt(power), n).astype(np.float32)
    return np.clip(audio, -1, 1)
