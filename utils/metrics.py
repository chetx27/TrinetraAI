"""Ring-buffer latency tracker and accuracy helpers.

``LatencyTracker`` stores the last *N* frame-processing times in a
fixed-size circular buffer and computes descriptive statistics without
allocating on the hot path.
"""

from __future__ import annotations

from typing import Final

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix


class LatencyTracker:
    """Fixed-capacity ring buffer for frame latency measurements.

    Parameters
    ----------
    capacity:
        Maximum number of samples retained.  Older samples are silently
        evicted when the buffer is full.
    """

    def __init__(self, capacity: int = 300) -> None:
        self._buf: Final[np.ndarray] = np.zeros(capacity, dtype=np.float64)
        self._capacity: Final[int] = capacity
        self._idx: int = 0
        self._count: int = 0

    def record(self, ms: float) -> None:
        """Append a latency measurement (milliseconds)."""
        self._buf[self._idx] = ms
        self._idx = (self._idx + 1) % self._capacity
        if self._count < self._capacity:
            self._count += 1

    def summary(self) -> dict[str, float]:
        """Compute descriptive statistics over the current buffer contents.

        Returns
        -------
        dict[str, float]
            Keys: ``mean_ms``, ``p95_ms``, ``p99_ms``, ``max_ms``.
            All values are ``0.0`` if no samples have been recorded.
        """
        if self._count == 0:
            return {"mean_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "max_ms": 0.0}

        data = self._buf[: self._count]
        return {
            "mean_ms": float(np.mean(data)),
            "p95_ms": float(np.percentile(data, 95)),
            "p99_ms": float(np.percentile(data, 99)),
            "max_ms": float(np.max(data)),
        }

    @property
    def count(self) -> int:
        """Number of samples currently in the buffer."""
        return self._count

    def reset(self) -> None:
        """Clear all recorded samples."""
        self._buf[:] = 0.0
        self._idx = 0
        self._count = 0


def compute_accuracy(y_true: list[str], y_pred: list[str]) -> float:
    """Return top-1 accuracy as a float in ``[0, 1]``.

    Parameters
    ----------
    y_true:
        Ground-truth labels.
    y_pred:
        Predicted labels.
    """
    return float(accuracy_score(y_true, y_pred))


def compute_confusion_matrix(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str] | None = None,
) -> np.ndarray:
    """Return the confusion matrix as a numpy array.

    Parameters
    ----------
    y_true:
        Ground-truth labels.
    y_pred:
        Predicted labels.
    labels:
        Explicit label ordering.
    """
    return confusion_matrix(y_true, y_pred, labels=labels)


def false_action_rate(
    y_true: list[str],
    y_pred: list[str],
) -> float:
    """Fraction of *neutral* frames that triggered a non-neutral action.

    This is the key safety metric: it measures how often the system
    fires an unwanted action when the user is in the neutral pose.
    """
    neutral_mask = [t == "neutral" for t in y_true]
    if not any(neutral_mask):
        return 0.0
    false_actions = sum(
        1
        for is_neutral, pred in zip(neutral_mask, y_pred, strict=True)
        if is_neutral and pred != "neutral"
    )
    return false_actions / sum(neutral_mask)
