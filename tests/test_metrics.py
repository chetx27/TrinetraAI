"""Tests for utils.metrics — latency tracker and accuracy helpers."""

from __future__ import annotations

import numpy as np
import pytest

from utils.metrics import LatencyTracker, compute_accuracy, false_action_rate


class TestLatencyTracker:
    def test_empty_summary(self) -> None:
        lt = LatencyTracker(capacity=10)
        s = lt.summary()
        assert s["mean_ms"] == 0.0
        assert s["max_ms"] == 0.0

    def test_single_record(self) -> None:
        lt = LatencyTracker(capacity=10)
        lt.record(5.0)
        s = lt.summary()
        assert s["mean_ms"] == 5.0
        assert s["max_ms"] == 5.0

    def test_ring_buffer_wraps(self) -> None:
        lt = LatencyTracker(capacity=3)
        for v in [1.0, 2.0, 3.0, 100.0]:
            lt.record(v)
        s = lt.summary()
        assert lt.count == 3
        assert s["max_ms"] == 100.0

    def test_percentiles(self) -> None:
        lt = LatencyTracker(capacity=100)
        for i in range(100):
            lt.record(float(i))
        s = lt.summary()
        assert s["p95_ms"] >= 90.0
        assert s["p99_ms"] >= 95.0

    def test_reset(self) -> None:
        lt = LatencyTracker(capacity=10)
        lt.record(42.0)
        lt.reset()
        assert lt.count == 0
        assert lt.summary()["mean_ms"] == 0.0


def test_compute_accuracy() -> None:
    y_true = ["a", "b", "a", "b"]
    y_pred = ["a", "b", "b", "b"]
    assert compute_accuracy(y_true, y_pred) == 0.75


def test_false_action_rate_zero() -> None:
    y_true = ["neutral", "neutral"]
    y_pred = ["neutral", "neutral"]
    assert false_action_rate(y_true, y_pred) == 0.0


def test_false_action_rate_nonzero() -> None:
    y_true = ["neutral", "neutral", "open_palm"]
    y_pred = ["open_palm", "neutral", "open_palm"]
    assert false_action_rate(y_true, y_pred) == 0.5


def test_false_action_rate_no_neutral() -> None:
    y_true = ["open_palm", "fist"]
    y_pred = ["open_palm", "fist"]
    assert false_action_rate(y_true, y_pred) == 0.0
