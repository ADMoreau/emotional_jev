"""CREMA-D label handling and face-crop geometry (pure functions; no dataset needed)."""
import csv

import numpy as np
import pytest

from emotion_decider.crema import actor_folds, parse_clip, vote_distribution
from emotion_decider.faces import crop, fill_missing, sample_indices, square_track


def test_parse_clip():
    assert parse_clip("1001_DFA_ANG_XX") == {"actor": "1001", "sentence": "DFA", "intended": "A", "acted_level": "XX"}
    assert parse_clip("1050_IEO_HAP_HI")["acted_level"] == "HI"
    with pytest.raises(ValueError):
        parse_clip("1001_DFA_XYZ_XX")


def test_vote_distribution():
    d = vote_distribution({"A": 6, "D": 3, "F": 0, "H": 0, "N": 1, "S": 0})
    assert d["anger"] == pytest.approx(0.6) and sum(d.values()) == pytest.approx(1)


def test_folds_are_disjoint_and_balanced(tmp_path):
    path = tmp_path / "demo.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ActorID", "Age", "Sex", "Race", "Ethnicity"])
        for i in range(91):
            w.writerow([1001 + i, 20 + i % 50, "Female" if i % 2 else "Male", "x", "y"])
    folds = actor_folds(path, k=5, seed=0)
    flat = [a for f in folds for a in f]
    assert len(flat) == len(set(flat)) == 91
    assert max(map(len, folds)) - min(map(len, folds)) <= 1
    assert folds == actor_folds(path, k=5, seed=0)          # deterministic


def test_sample_indices_cover_the_clip():
    idx = sample_indices(68, 16)
    assert len(idx) == 16 and idx[0] >= 0 and idx[-1] <= 67 and (np.diff(idx) > 0).all()
    assert (sample_indices(10, 16) <= 9).all()               # shorter than k: repeats, never out of range


def test_fill_missing_interpolates_and_holds_edges():
    boxes = [[np.nan] * 4, [0, 0, 10, 10], [np.nan] * 4, [10, 10, 30, 30], [np.nan] * 4]
    out = fill_missing(boxes)
    assert out[0].tolist() == [0, 0, 10, 10] and out[2].tolist() == [5, 5, 20, 20] and out[4].tolist() == [10, 10, 30, 30]
    assert fill_missing([[np.nan] * 4] * 3) is None


def test_square_track_uses_median_side_and_smooths():
    boxes = np.array([[100, 100, 200, 220]] * 5, dtype=float)
    boxes[2] += 30                                            # one jumpy frame
    tr = square_track(boxes, margin=1.0, smooth=3)
    assert np.allclose(tr[:, 2], 120)                          # side = median of max(w, h)
    assert tr[2, 0] < 150 + 30 and tr[2, 0] > 150              # centre pulled back toward neighbours


def test_crop_pads_at_borders():
    frame = np.zeros((360, 480, 3), np.uint8)
    frame[:, :240] = 255
    out = crop(frame, cx=10, cy=10, side=100, out=224)         # mostly outside the frame: edge padding
    assert out.shape == (224, 224, 3)
    centred = crop(frame, cx=240, cy=180, side=100, out=224)
    assert centred[:, :100].mean() > 250 and centred[:, -100:].mean() < 5


def test_sample_indices_within_window():
    idx = sample_indices(101, 16, start=10, stop=78)    # e.g. a video that runs past its audio
    assert idx[0] >= 10 and idx[-1] <= 77 and (np.diff(idx) > 0).all()


@pytest.mark.skipif(not __import__("pathlib").Path("data/crema-d/derived/faces").exists(), reason="CREMA-D not built")
@pytest.mark.parametrize("condition", ["av", "face", "voice"])
def test_packed_clip_decodes_identically(tmp_path, condition):
    import shutil
    from pathlib import Path
    from emotion_decider.dataset import load_media, pack
    clip, src = "1001_DFA_ANG_XX", Path("data/crema-d")
    # a root holding only the packed file: load_media must not touch the separate files
    (tmp_path / "derived" / "packed").mkdir(parents=True)
    np.savez(tmp_path / "derived" / "packed" / f"{clip}.npz", **pack(clip, src))
    a, b = load_media(clip, condition, src), load_media(clip, condition, tmp_path)
    for x, y in zip(a, b):
        assert (x is None and y is None) or np.array_equal(x, y)
