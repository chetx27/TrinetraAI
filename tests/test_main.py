"""Tests for main — event loop wiring and head-label resolution."""

from __future__ import annotations

from main import _resolve_head_label


def test_head_turn_left() -> None:
    assert _resolve_head_label(20.0, 0.0, 15.0, 10.0) == "head_turn_left"


def test_head_turn_right() -> None:
    assert _resolve_head_label(-20.0, 0.0, 15.0, 10.0) == "head_turn_right"


def test_head_nod_down() -> None:
    assert _resolve_head_label(0.0, 15.0, 15.0, 10.0) == "head_nod_down"


def test_head_tilt_up() -> None:
    assert _resolve_head_label(0.0, -15.0, 15.0, 10.0) == "head_tilt_up"


def test_neutral_head() -> None:
    assert _resolve_head_label(5.0, 3.0, 15.0, 10.0) is None


def test_yaw_priority_over_pitch() -> None:
    """When both yaw and pitch exceed thresholds, yaw wins."""
    label = _resolve_head_label(20.0, 15.0, 15.0, 10.0)
    assert label == "head_turn_left"
