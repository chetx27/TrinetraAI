"""Threaded webcam capture with double-buffered frame delivery.

Runs OpenCV's ``VideoCapture`` in a background daemon thread so that
``read()`` never blocks on I/O — it returns the most recent decoded
frame immediately.  This keeps the main processing loop latency-free
even if the USB driver occasionally stalls.
"""

from __future__ import annotations

import threading
from typing import Final

import cv2
import numpy as np


class CaptureError(Exception):
    """Raised when the webcam cannot be opened or a frame cannot be read."""


class WebcamCapture:
    """Non-blocking, threaded webcam reader.

    Parameters
    ----------
    index:
        ``/dev/videoN`` index (Linux) or DirectShow index (Windows).
    width:
        Requested horizontal resolution.
    height:
        Requested vertical resolution.
    """

    def __init__(self, index: int = 0, width: int = 640, height: int = 480) -> None:
        self._cap: Final[cv2.VideoCapture] = cv2.VideoCapture(index)
        if not self._cap.isOpened():
            raise CaptureError(f"Cannot open camera at index {index}")

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1.0)

        # Read one frame to initialise the buffer
        ok, frame = self._cap.read()
        if not ok or frame is None:
            raise CaptureError("Failed to read initial frame from camera")

        self._frame: np.ndarray = frame
        self._lock: Final[threading.Lock] = threading.Lock()
        self._running: bool = True

        self._thread: Final[threading.Thread] = threading.Thread(
            target=self._capture_loop, daemon=True, name="webcam-capture"
        )
        self._thread.start()

    # ── background loop ───────────────────────────────────────────

    def _capture_loop(self) -> None:
        """Continuously grab frames and store the latest one."""
        while self._running:
            ok, frame = self._cap.read()
            if not ok or frame is None:
                continue
            with self._lock:
                self._frame = frame

    # ── public API ────────────────────────────────────────────────

    def read(self) -> np.ndarray:
        """Return the most recent BGR frame.

        This call never blocks on camera I/O; it returns a *copy* of the
        latest frame held by the background thread.

        Returns
        -------
        np.ndarray
            BGR image of shape ``(H, W, 3)``, dtype ``uint8``.
        """
        with self._lock:
            return self._frame.copy()

    def release(self) -> None:
        """Stop the capture thread and release the camera device."""
        self._running = False
        self._thread.join(timeout=2.0)
        self._cap.release()
