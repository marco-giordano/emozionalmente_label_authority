"""Hard and distributional scores shared by training and analysis."""

from __future__ import annotations

import numpy as np

from label_authority.config import N_CLASSES

EPS = 1e-12


def hard_metrics(true: np.ndarray, pred: np.ndarray) -> dict:
    """Accuracy, UAR (mean recall over the reference classes present) and macro F1."""
    true, pred = np.asarray(true, dtype=int), np.asarray(pred, dtype=int)
    recalls, f1s = [], []
    for c in range(N_CLASSES):
        tp = np.sum((true == c) & (pred == c))
        support, predicted = np.sum(true == c), np.sum(pred == c)
        if support:
            recalls.append(tp / support)
        if support or predicted:
            precision = tp / predicted if predicted else 0.0
            recall = tp / support if support else 0.0
            f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return {
        "accuracy": float(np.mean(true == pred)),
        "uar": float(np.mean(recalls)) if recalls else float("nan"),
        "macro_f1": float(np.mean(f1s)) if f1s else float("nan"),
    }


def _kl_rows(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    terms = np.zeros_like(p)
    mask = p > 0
    terms[mask] = p[mask] * (np.log(p[mask]) - np.log(np.maximum(q[mask], EPS)))
    return terms.sum(axis=1)


def distribution_rows(human: np.ndarray, model: np.ndarray) -> dict[str, np.ndarray]:
    """Per-clip divergences (nats) between the listener vote distribution and the model output.

    kl     KL(human || model), the primary proper-score evidence with brier
    brier  multiclass Brier score, sum_c (q_c - p_c)^2
    ce     cross-entropy H(human, model) = H(human) + KL
    js     Jensen-Shannon divergence
    """
    kl = _kl_rows(human, model)
    entropy = -_kl_rows(human, np.ones_like(human))  # -sum p log p
    mid = (human + model) / 2.0
    return {
        "kl": kl,
        "brier": np.sum((model - human) ** 2, axis=1),
        "ce": entropy + kl,
        "js": 0.5 * _kl_rows(human, mid) + 0.5 * _kl_rows(model, mid),
    }
