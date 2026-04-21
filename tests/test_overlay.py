"""Tests for vision.overlay — HUD rendering."""

from __future__ import annotations

import numpy as np

from config.loader import OverlayConfig
from vision.overlay import Overlay


def _blank_frame(h: int = 480, w: int = 640) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_draw_returns_same_array() -> None:
    cfg = OverlayConfig()
    ov = Overlay(cfg)
    frame = _blank_frame()
    result = ov.draw(frame, label="test", latency_ms=5.0)
    assert result is frame  # in-place mutation


def test_draw_modifies_frame_with_label() -> None:
    cfg = OverlayConfig(show_label=True, show_latency=False, show_landmarks=False)
    ov = Overlay(cfg)
    frame = _blank_frame()
    original_sum = int(frame.sum())
    ov.draw(frame, label="open_palm")
    assert int(frame.sum()) > original_sum  # pixels changed


def test_draw_with_landmarks() -> None:
    cfg = OverlayConfig(show_landmarks=True, show_label=False, show_latency=False)
    ov = Overlay(cfg)
    frame = _blank_frame()
    landmarks = [(0.5, 0.5), (0.6, 0.6)]
    ov.draw(frame, face_landmarks=landmarks)
    assert int(frame.sum()) > 0


def test_draw_no_flags() -> None:
    """Nothing should be drawn when all flags are off."""
    cfg = OverlayConfig(show_landmarks=False, show_label=False, show_latency=False)
    ov = Overlay(cfg)
    frame = _blank_frame()
    ov.draw(frame, label="test", latency_ms=10.0)
    assert int(frame.sum()) == 0


def test_draw_with_none_landmarks() -> None:
    cfg = OverlayConfig(show_landmarks=True)
    ov = Overlay(cfg)
    frame = _blank_frame()
    ov.draw(frame, face_landmarks=None, hand_landmarks=None)
    # Should not raise even with None landmarks
