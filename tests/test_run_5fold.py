"""5-fold aggregation (pure; no Modal)."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("run_5fold", Path(__file__).resolve().parents[1] / "scripts" / "run_5fold.py")
run_5fold = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_5fold)


def fold(acc_av, acc_face, t):
    m = lambda acc: {"acc": acc, "uar": acc, "f1_macro": acc, "ece": 0.02, "nll": 0.5}
    return {"temperatures": {"intended": t}, "selected_step": 100,
            "test": {"intended": {"av": m(acc_av), "face": m(acc_face)},
                     "perceived": {"av": {"kl": 0.3, "agree_majority": 0.8}}}}


def test_aggregate_mean_std_and_partial_conditions():
    s = run_5fold.aggregate({"a": fold(0.80, 0.60, 1.0), "b": fold(0.90, 0.70, 2.0)})
    mean, std, n = s["av"]["intended.acc"]
    assert (mean, n) == (pytest.approx(0.85), 2) and std == pytest.approx(0.0707, abs=1e-3)   # sample std
    assert "voice" in s and s["voice"] == {}                       # no fold had voice: empty, not an error
    assert s["temperatures"]["intended"][0] == pytest.approx(1.5)
    assert "| av | 2 |" in run_5fold.table(s, "x")


def test_single_fold_std_is_zero():
    s = run_5fold.aggregate({"a": fold(0.8, 0.6, 1.0)})
    assert s["face"]["intended.acc"] == [pytest.approx(0.6), 0.0, 1]
