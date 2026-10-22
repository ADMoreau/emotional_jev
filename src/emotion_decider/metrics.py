"""Metrics over per-question probability vectors, and temperature scaling.

Every function takes `probs` [N, K] (rows sum to 1) and either `labels` [N] (hard) or `targets` [N, K] (soft,
e.g. crowd vote distributions). ECE is top-label ECE over 15 equal-width confidence bins, against hard labels.
"""
import math

import numpy as np

EPS = 1e-12


def accuracy(probs, labels):
    return float((np.argmax(probs, 1) == labels).mean())


def uar(probs, labels, k=None):
    """Unweighted average recall: mean over classes of per-class accuracy (classes present in labels)."""
    pred, k = np.argmax(probs, 1), k or probs.shape[1]
    return float(np.mean([(pred[labels == c] == c).mean() for c in range(k) if (labels == c).any()]))


def f1(probs, labels, average="macro"):
    pred, k = np.argmax(probs, 1), probs.shape[1]
    if average == "micro":   # single-label multiclass: micro-F1 == accuracy
        return accuracy(probs, labels)
    scores = []
    for c in range(k):
        tp = ((pred == c) & (labels == c)).sum()
        fp = ((pred == c) & (labels != c)).sum()
        fn = ((pred != c) & (labels == c)).sum()
        if tp + fp + fn:
            scores.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(scores))


def nll(probs, targets):
    """Cross-entropy against soft targets (hard labels as one-hot), nats per item."""
    return float(-(targets * np.log(probs + EPS)).sum(1).mean())


def kl(targets, probs):
    """KL(targets || probs), nats per item."""
    return float((targets * (np.log(targets + EPS) - np.log(probs + EPS))).sum(1).mean())


def ece(probs, labels, bins=15):
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs((pred[m] == labels[m]).mean() - conf[m].mean())
    return float(total)


def aurc(probs, labels):
    """Area under the risk-coverage curve (selective classification by confidence); lower is better."""
    order = np.argsort(-probs.max(1))
    errors = (probs.argmax(1) != labels)[order]
    return float(np.mean(np.cumsum(errors) / np.arange(1, len(errors) + 1)))


def one_hot(labels, k):
    return np.eye(k)[labels]


def softmax(logits, t=1.0):
    z = np.asarray(logits, dtype=np.float64) / t
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def fit_temperature(logits_list, targets_list, lo=0.05, hi=20.0, steps=200):
    """Temperature minimizing soft-target NLL. logits_list / targets_list: per-item arrays (option counts may
    differ across items). Golden-section search on log T: NLL(T) is unimodal in log T for a fixed set of logits."""
    def loss(log_t):
        t = math.exp(log_t)
        return sum(-(np.asarray(y) * np.log(softmax(z, t) + EPS)).sum() for z, y in zip(logits_list, targets_list))
    a, b = math.log(lo), math.log(hi)
    g = (math.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = loss(c), loss(d)
    for _ in range(steps):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a); fc = loss(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a); fd = loss(d)
        if b - a < 1e-5:
            break
    return math.exp((a + b) / 2)
