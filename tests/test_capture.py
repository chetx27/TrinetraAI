"""Tests for vision.capture — threaded webcam capture.

Uses a monkeypatched cv2.VideoCapture so that no real camera is needed.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from vision.capture import CaptureError, WebcamCapture


def _make_fake_capture(
    frames: list[np.ndarray] | None = None,
    open_ok: bool = True,
) -> MagicMock:
    """Build a mock cv2.VideoCapture that yields synthetic frames."""
    mock = MagicMock()
    mock.isOpened.return_value = open_ok

    if frames is None:
        frames = [np.zeros((480, 640, 3), dtype=np.uint8)]

    frame_iter = iter(frames * 100)  # repeat so the thread doesn't starve

    def _read() -> tuple[bool, np.ndarray | None]:
        try:
            return True, next(frame_iter)
        except StopIteration:
            return False, None

    mock.read.side_effect = _read
    mock.set.return_value = True
    return mock


@patch("vision.capture.cv2.VideoCapture")
def test_read_returns_bgr_frame(mock_vc_cls: MagicMock) -> None:
    frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    mock_vc_cls.return_value = _make_fake_capture([frame])

    cam = WebcamCapture(index=0)
    result = cam.read()

    assert result.shape == (480, 640, 3)
    assert result.dtype == np.uint8
    cam.release()


@patch("vision.capture.cv2.VideoCapture")
def test_read_returns_copy(mock_vc_cls: MagicMock) -> None:
    frame = np.full((480, 640, 3), 42, dtype=np.uint8)
    mock_vc_cls.return_value = _make_fake_capture([frame])

    cam = WebcamCapture(index=0)
    a = cam.read()
    b = cam.read()
    assert a is not b  # different objects
    cam.release()


@patch("vision.capture.cv2.VideoCapture")
def test_camera_not_opened_raises(mock_vc_cls: MagicMock) -> None:
    mock_vc_cls.return_value = _make_fake_capture(open_ok=False)
    with pytest.raises(CaptureError, match="Cannot open"):
        WebcamCapture(index=99)


@patch("vision.capture.cv2.VideoCapture")
def test_release_stops_thread(mock_vc_cls: MagicMock) -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_vc_cls.return_value = _make_fake_capture([frame])

    cam = WebcamCapture(index=0)
    cam.release()
    assert not cam._running
