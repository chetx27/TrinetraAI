"""HUD overlay: draw landmarks, current gesture label, and latency.

All drawing is done *in-place* on the passed frame to avoid an extra
memory copy.  Colours and font parameters are controlled through
``OverlayConfig``.
"""

from __future__ import annotations

import cv2
import numpy as np

from config.loader import OverlayConfig


class Overlay:
    """Heads-up display rendered on top of the camera frame.

    Parameters
    ----------
    config:
        Visual settings (colours, font scale, toggle flags).
    """

    def __init__(self, config: OverlayConfig) -> None:
        self._cfg: OverlayConfig = config

    # ── public API ────────────────────────────────────────────────

    def draw(
        self,
        frame: np.ndarray,
        *,
        label: str = "",
        latency_ms: float = 0.0,
        face_landmarks: list[tuple[float, float]] | None = None,
        hand_landmarks: list[tuple[float, float]] | None = None,
    ) -> np.ndarray:
        """Render the overlay onto *frame* (mutated in-place and returned).

        Parameters
        ----------
        frame:
            BGR image of shape ``(H, W, 3)``.
        label:
            Current gesture / head-pose label string.
        latency_ms:
            End-to-end pipeline latency for this frame.
        face_landmarks:
            Normalised ``(x, y)`` list from ``FaceTracker.landmarks``.
        hand_landmarks:
            Normalised ``(x, y)`` list from ``GestureDetector.landmarks``.

        Returns
        -------
        np.ndarray
            The same *frame* array with drawings applied.
        """
        h, w = frame.shape[:2]

        if self._cfg.show_landmarks:
            self._draw_landmarks(frame, face_landmarks, w, h, radius=1)
            self._draw_landmarks(frame, hand_landmarks, w, h, radius=3)

        if self._cfg.show_label and label:
            self._draw_label(frame, label, w, h)

        if self._cfg.show_latency:
            self._draw_latency(frame, latency_ms, h)

        return frame

    # ── private helpers ───────────────────────────────────────────

    def _draw_landmarks(
        self,
        frame: np.ndarray,
        landmarks: list[tuple[float, float]] | None,
        w: int,
        h: int,
        radius: int,
    ) -> None:
        if landmarks is None:
            return
        colour = (0, 255, 128)  # light green
        for lx, ly in landmarks:
            cx = int(lx * w)
            cy = int(ly * h)
            cv2.circle(frame, (cx, cy), radius, colour, -1)

    def _draw_label(self, frame: np.ndarray, label: str, w: int, h: int) -> None:
        text = f"Gesture: {label}"
        colour = self._cfg.label_color
        cv2.putText(
            frame,
            text,
            (10, h - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            self._cfg.font_scale,
            colour,
            self._cfg.font_thickness,
            cv2.LINE_AA,
        )

    def _draw_latency(self, frame: np.ndarray, latency_ms: float, h: int) -> None:
        text = f"Latency: {latency_ms:.1f} ms"
        colour = self._cfg.latency_color
        cv2.putText(
            frame,
            text,
            (10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            self._cfg.font_scale,
            colour,
            self._cfg.font_thickness,
            cv2.LINE_AA,
        )
