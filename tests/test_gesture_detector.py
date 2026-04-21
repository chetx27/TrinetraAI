"""Tests for vision.gesture_detector — hand landmark extraction."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from vision.gesture_detector import GestureDetector


def _make_synthetic_frame(w: int = 640, h: int = 480) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


@patch("vision.gesture_detector.mp.solutions.hands.Hands")
def test_process_no_hand_returns_none(mock_hands_cls: MagicMock) -> None:
    mock_instance = MagicMock()
    mock_instance.process.return_value = MagicMock(multi_hand_landmarks=None)
    mock_hands_cls.return_value = mock_instance

    detector = GestureDetector()
    result = detector.process(_make_synthetic_frame())
    assert result is None
    assert detector.landmarks is None


@patch("vision.gesture_detector.mp.solutions.hands.Hands")
def test_process_returns_63d_vector(mock_hands_cls: MagicMock) -> None:
    """Simulate a detected hand with 21 landmarks."""
    landmarks = []
    for i in range(21):
        lm = MagicMock()
        lm.x = 0.3 + i * 0.02
        lm.y = 0.4 + i * 0.01
        lm.z = 0.0
        landmarks.append(lm)

    mock_hand = MagicMock()
    mock_hand.landmark = landmarks

    mock_result = MagicMock()
    mock_result.multi_hand_landmarks = [mock_hand]

    mock_instance = MagicMock()
    mock_instance.process.return_value = mock_result
    mock_hands_cls.return_value = mock_instance

    detector = GestureDetector()
    result = detector.process(_make_synthetic_frame())

    assert result is not None
    assert result.shape == (63,)
    assert result.dtype == np.float32


@patch("vision.gesture_detector.mp.solutions.hands.Hands")
def test_wrist_is_origin(mock_hands_cls: MagicMock) -> None:
    """After normalisation the wrist (landmark 0) should be at origin."""
    landmarks = []
    for i in range(21):
        lm = MagicMock()
        lm.x = 0.5 + i * 0.01
        lm.y = 0.5 + i * 0.01
        lm.z = 0.0
        landmarks.append(lm)

    mock_hand = MagicMock()
    mock_hand.landmark = landmarks
    mock_result = MagicMock()
    mock_result.multi_hand_landmarks = [mock_hand]
    mock_instance = MagicMock()
    mock_instance.process.return_value = mock_result
    mock_hands_cls.return_value = mock_instance

    detector = GestureDetector()
    vec = detector.process(_make_synthetic_frame())
    assert vec is not None

    # First 3 values (wrist) should be ~0
    np.testing.assert_allclose(vec[:3], [0.0, 0.0, 0.0], atol=1e-5)


@patch("vision.gesture_detector.mp.solutions.hands.Hands")
def test_close_does_not_raise(mock_hands_cls: MagicMock) -> None:
    detector = GestureDetector()
    detector.close()
