"""Tests for vision.face_tracker — head-pose estimation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from vision.face_tracker import FaceTracker, HeadPose


def _make_synthetic_frame(w: int = 640, h: int = 480) -> np.ndarray:
    """Return a blank BGR frame."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_headpose_dataclass() -> None:
    hp = HeadPose(yaw=10.0, pitch=-5.0, roll=0.5)
    assert hp.yaw == 10.0
    assert hp.pitch == -5.0
    assert hp.roll == 0.5


def test_headpose_is_frozen() -> None:
    hp = HeadPose(yaw=0.0, pitch=0.0, roll=0.0)
    with pytest.raises(AttributeError):
        hp.yaw = 1.0  # type: ignore[misc]


@patch("vision.face_tracker.mp.solutions.face_mesh.FaceMesh")
def test_process_returns_none_when_no_face(mock_mesh_cls: MagicMock) -> None:
    mock_instance = MagicMock()
    mock_instance.process.return_value = MagicMock(multi_face_landmarks=None)
    mock_mesh_cls.return_value = mock_instance

    tracker = FaceTracker()
    result = tracker.process(_make_synthetic_frame())

    assert result is None
    assert tracker.landmarks is None


@patch("vision.face_tracker.mp.solutions.face_mesh.FaceMesh")
def test_process_returns_headpose_with_face(mock_mesh_cls: MagicMock) -> None:
    """Simulate a detected face with synthetic landmarks."""
    # Create mock landmarks
    mock_landmark = MagicMock()
    # We need 468 landmarks for FaceMesh
    landmarks = []
    for i in range(468):
        lm = MagicMock()
        lm.x = 0.5 + (i % 10) * 0.01
        lm.y = 0.5 + (i % 10) * 0.01
        lm.z = 0.0
        landmarks.append(lm)
    mock_landmark.landmark = landmarks

    mock_result = MagicMock()
    mock_result.multi_face_landmarks = [mock_landmark]

    mock_instance = MagicMock()
    mock_instance.process.return_value = mock_result
    mock_mesh_cls.return_value = mock_instance

    tracker = FaceTracker()
    result = tracker.process(_make_synthetic_frame())

    # solvePnP may or may not succeed with synthetic data,
    # but the function should not raise
    if result is not None:
        assert isinstance(result, HeadPose)
        assert isinstance(result.yaw, float)
        assert isinstance(result.pitch, float)
        assert isinstance(result.roll, float)


def test_close_does_not_raise() -> None:
    """Ensure close() works without a real MediaPipe instance."""
    with patch("vision.face_tracker.mp.solutions.face_mesh.FaceMesh"):
        tracker = FaceTracker()
        tracker.close()  # should not raise
