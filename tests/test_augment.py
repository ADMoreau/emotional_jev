"""Training-time augmentation: deterministic per seed, shapes and alignment kept, questions and evaluation untouched."""
import random
from pathlib import Path

import numpy as np
import pytest

from facedecide.augment import augment_media
from facedecide.train import load_config

CFG = load_config(Path(__file__).resolve().parents[1] / "configs" / "omni_3b.yaml")["augment"]
HAVE_DATA = Path("data/crema-d/derived/targets.jsonl").exists()


def media(seconds=2.5, fps=6.4):
    rng = np.random.default_rng(0)
    return (rng.integers(0, 256, (16, 224, 224, 3), dtype=np.uint8), fps,
            (0.1 * rng.standard_normal(int(16000 * seconds))).astype(np.float32))


def test_deterministic_and_shapes_kept():
    frames, fps, audio = media()
    a = augment_media(frames, fps, audio, random.Random("x"), CFG)
    b = augment_media(frames, fps, audio, random.Random("x"), CFG)
    c = augment_media(frames, fps, audio, random.Random("y"), CFG)
    assert np.array_equal(a[0], b[0]) and a[1] == b[1] and np.array_equal(a[2], b[2])
    assert not np.array_equal(a[0], c[0])
    assert a[0].shape == frames.shape and a[0].dtype == np.uint8 and a[2].dtype == np.float32
    assert np.abs(a[2]).max() <= 1


@pytest.mark.parametrize("seed", range(5))
def test_speed_change_keeps_audio_and_frames_on_one_span(seed):
    frames, fps, audio = media()
    f2, fps2, a2 = augment_media(frames, fps, audio, random.Random(seed), CFG)
    before, after = (len(audio) / 16000) / (len(frames) / fps), (len(a2) / 16000) / (len(f2) / fps2)
    assert after == pytest.approx(before, rel=1e-3)


def test_single_modality_conditions():
    frames, fps, audio = media()
    f, r, a = augment_media(frames, fps, None, random.Random(0), CFG)
    assert a is None and r == fps and f.shape == frames.shape      # face only: no speed change to apply
    f, r, a = augment_media(None, None, audio, random.Random(0), CFG)
    assert f is None and r is None and a is not None


@pytest.mark.skipif(not HAVE_DATA, reason="CREMA-D not built")
def test_questions_identical_and_eval_never_augmented():
    from transformers import AutoProcessor
    from facedecide.train import Examples, Format, load_rows, predict  # noqa: F401
    root = "data/crema-d"
    rows = load_rows(root)
    clip = sorted(rows)[0]
    proc = AutoProcessor.from_pretrained("Qwen/Qwen2.5-Omni-3B")
    fmt = Format("<|im_start|>user\n", "<|im_end|>\n")
    items = [(clip, "train/0/0/x", None)]
    plain = Examples(items, rows, proc, fmt, root)[0]
    aug = Examples(items, rows, proc, fmt, root, augment=dict(CFG, enabled=True))[0]
    assert plain["condition"] == aug["condition"] and plain["questions"] == aug["questions"]
    assert plain["targets"] == aug["targets"]
    key = "pixel_values_videos" if "pixel_values_videos" in plain["clip"].mm else "input_features"
    assert not np.array_equal(plain["clip"].mm[key].numpy(), aug["clip"].mm[key].numpy())
    assert Examples(items, rows, proc, fmt, root).augment is None   # predict() builds Examples without augment
