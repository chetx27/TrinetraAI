"""Hand-gesture landmark extraction via MediaPipe Hands.

Returns a flat 63-dimensional feature vector (21 landmarks × 3 coords)
that is normalised relative to the wrist so the classifier is
translation-invariant.  Coordinates are further scaled by the
palm-to-middle-finger distance for size invariance.
"""

from __future__ import annotations

from typing import Final

import cv2
import mediapipe as mp
import numpy as np

# Number of hand landmarks produced by MediaPipe Hands
_NUM_LANDMARKS: Final[int] = 21
_FEATURE_DIM: Final[int] = _NUM_LANDMARKS * 3  # 63


class GestureDetector:
    """MediaPipe Hands → normalised 63-d landmark vector.

    Parameters
    ----------
    max_num_hands:
        Maximum number of hands to detect (only the first is used).
    min_detection_confidence:
        MediaPipe detection threshold.
    min_tracking_confidence:
        MediaPipe tracking threshold.
    """

    def __init__(
        self,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self._hands = mp.solutions.hands.Hands(  # type: ignore[attr-defined]
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._raw_landmarks: list[tuple[float, float]] | None = None

    # ── public API ────────────────────────────────────────────────

    def process(self, frame: np.ndarray) -> np.ndarray | None:
        """Extract a normalised landmark vector from a BGR frame.

        Parameters
        ----------
        frame:
            BGR image of shape ``(H, W, 3)``, dtype ``uint8``.

        Returns
        -------
        np.ndarray | None
            Flat ``(63,)`` float32 vector, or ``None`` if no hand is
            detected.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb)

        if results.multi_hand_landmarks is None:  # type: ignore[union-attr]
            self._raw_landmarks = None
            return None

        hand = results.multi_hand_landmarks[0]  # type: ignore[union-attr]

        # Store raw normalised (x, y) for overlay drawing
        self._raw_landmarks = [
            (lm.x, lm.y) for lm in hand.landmark
        ]

        # Build raw (21, 3) array
        raw = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand.landmark],
            dtype=np.float32,
        )

        # Translate so wrist (landmark 0) is the origin
        wrist = raw[0].copy()
        centred = raw - wrist

        # Scale by palm→middle-finger-MCP distance for size invariance
        # Landmark 0 = wrist, landmark 9 = middle-finger MCP
        scale = float(np.linalg.norm(centred[9]))
        if scale < 1e-6:
            scale = 1.0
        normalised = centred / scale

        return normalised.flatten()

    @property
    def landmarks(self) -> list[tuple[float, float]] | None:
        """Normalised (x, y) landmarks from the last ``process()`` call."""
        return self._raw_landmarks

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._hands.close()
