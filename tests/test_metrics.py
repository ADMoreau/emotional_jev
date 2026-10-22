import numpy as np
import pytest

from emotion_decider.metrics import accuracy, aurc, ece, f1, fit_temperature, kl, nll, one_hot, softmax, uar


def test_classification_metrics():
    probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.6, 0.3, 0.1], [0.2, 0.2, 0.6]])
    labels = np.array([0, 1, 1, 2])
    assert accuracy(probs, labels) == 0.75
    assert uar(probs, labels) == pytest.approx((1 + 0.5 + 1) / 3)
    assert f1(probs, labels, "micro") == 0.75
    assert f1(probs, labels, "macro") == pytest.approx(np.mean([2 / 3, 2 / 3, 1.0]))


def test_nll_kl():
    t = np.array([[0.5, 0.5]])
    assert kl(t, t) == pytest.approx(0)
    assert nll(t, t) == pytest.approx(np.log(2))
    assert kl(t, np.array([[0.9, 0.1]])) > 0


def test_ece_perfectly_calibrated_is_zero():
    rng = np.random.default_rng(0)
    conf = rng.uniform(0.5, 1, 20000)
    labels = (rng.uniform(size=conf.size) > conf).astype(int)   # correct (label 0) with prob = conf
    probs = np.stack([conf, 1 - conf], 1)
    assert ece(probs, labels) < 0.02
    assert ece(np.stack([np.ones(100), np.zeros(100)], 1), np.r_[np.zeros(50, int), np.ones(50, int)]) == pytest.approx(0.5)


def test_aurc_prefers_confident_correct():
    labels = np.array([0, 1])
    error_least_confident = np.array([[0.9, 0.1], [0.6, 0.4]])   # item 1 is wrong, at confidence 0.6
    error_most_confident = np.array([[0.6, 0.4], [0.95, 0.05]])  # item 1 is wrong, at confidence 0.95
    assert aurc(error_least_confident, labels) < aurc(error_most_confident, labels)


def test_fit_temperature_recovers_true_temperature():
    rng = np.random.default_rng(0)
    logits = rng.normal(0, 3, (4000, 6))
    true_t = 2.5
    labels = np.array([rng.choice(6, p=p) for p in softmax(logits, true_t)])
    t = fit_temperature(list(logits), list(one_hot(labels, 6)))
    assert t == pytest.approx(true_t, rel=0.08)
