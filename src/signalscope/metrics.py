"""Binary metrics with explicit AI-positive scores and a fixed operating point."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def binary_metrics(labels, scores, threshold: float = .5) -> dict:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or scores.shape != labels.shape or len(labels) == 0:
        raise ValueError("Non-empty one-dimensional labels/scores of matching shape required.")
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError("Labels must be 0/1; scores must be finite.")
    if not ((scores >= 0) & (scores <= 1)).all() or not 0 <= threshold <= 1:
        raise ValueError("Probability scores and threshold must be in [0, 1].")
    predictions = (scores >= threshold).astype(np.int64)
    cm = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    return {
        "count": len(labels), "real_count": tn+fp, "ai_count": fn+tp,
        "roc_auc": float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None,
        "macro_f1": float(f1_score(labels, predictions, average="macro", labels=[0, 1], zero_division=0)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "false_positive_rate": fp/(tn+fp) if tn+fp else None,
        "true_positive_rate": tp/(tp+fn) if tp+fn else None,
        "brier_score": float(brier_score_loss(labels, scores)),
        "threshold": float(threshold), "positive_class": "ai_generated",
        "confusion_matrix": cm.tolist(), "confusion_matrix_order": ["real", "ai_generated"],
    }


def select_threshold(labels, scores, max_fpr: float = .05) -> float:
    """Validation-only threshold maximizing TPR subject to the requested FPR.

    Includes a threshold just above the highest real score. Uses vectorized
    sorted counts rather than repeatedly evaluating the full dataset.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    if not 0 <= max_fpr <= 1 or set(np.unique(labels)) != {0, 1}:
        raise ValueError("Both classes and a max_fpr in [0,1] are required.")
    candidates = np.unique(np.concatenate(([0., .5, 1.], scores,
                                          np.nextafter(scores, np.inf))))
    candidates = candidates[(candidates >= 0) & (candidates <= 1)]
    real, fake = np.sort(scores[labels == 0]), np.sort(scores[labels == 1])
    fpr = (len(real)-np.searchsorted(real, candidates, side="left"))/len(real)
    tpr = (len(fake)-np.searchsorted(fake, candidates, side="left"))/len(fake)
    eligible = np.flatnonzero(fpr <= max_fpr)
    if not len(eligible):
        raise ValueError("No threshold in [0,1] meets max_fpr; inspect saturated scores.")
    # Highest recall, then lower FPR, then higher threshold for equal predictions.
    best = max(eligible, key=lambda i: (tpr[i], -fpr[i], candidates[i]))
    return float(candidates[best])


def expected_calibration_error(labels, scores, bins: int = 15) -> float:
    """Weighted mean |accuracy - confidence| over equal-width bins of the 0.5-argmax class."""
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    predictions = scores >= .5
    confidence = np.where(predictions, scores, 1-scores)
    correct = predictions == labels.astype(bool)
    index = np.clip(np.digitize(confidence, np.linspace(.5, 1, bins+1)[1:-1]), 0, bins-1)
    return float(sum(abs(correct[index == b].mean()-confidence[index == b].mean())*(index == b).mean()
                     for b in range(bins) if (index == b).any()))

