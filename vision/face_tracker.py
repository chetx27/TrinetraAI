"""Head-pose estimation via MediaPipe FaceMesh.

Extracts yaw / pitch / roll from the Perspective-n-Point (PnP) solution
using six canonical 3-D face landmarks and their 2-D projections.  All
angles are returned in **degrees** (matching the config thresholds).

The 3-D reference model is taken from a generic human-skull metric scan
and stays fixed across sessions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import cv2
import mediapipe as mp
import numpy as np

# ── Canonical 3-D model points (metric skull, mm) ────────────────────
# nose tip, chin, left eye corner, right eye corner, left mouth, right mouth
_MODEL_POINTS: Final[np.ndarray] = np.array(
    [
        [0.0, 0.0, 0.0],  # nose tip
        [0.0, -330.0, -65.0],  # chin
        [-225.0, 170.0, -135.0],  # left eye left corner
        [225.0, 170.0, -135.0],  # right eye right corner
        [-150.0, -150.0, -125.0],  # left mouth corner
        [150.0, -150.0, -125.0],  # right mouth corner
    ],
    dtype=np.float64,
)

# MediaPipe FaceMesh landmark indices for the six points above
_LANDMARK_IDS: Final[list[int]] = [1, 152, 33, 263, 61, 291]


@dataclass(frozen=True, slots=True)
class HeadPose:
    """Euler angles of the head in degrees (right-hand rule, camera frame)."""

    yaw: float
    pitch: float
    roll: float


class FaceTracker:
    """MediaPipe FaceMesh → head-pose angles.

    Parameters
    ----------
    max_num_faces:
        Maximum number of faces to detect (only the first is used).
    min_detection_confidence:
        MediaPipe detection threshold.
    min_tracking_confidence:
        MediaPipe tracking threshold.
    """

    def __init__(
        self,
        max_num_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(  # type: ignore[attr-defined]
            static_image_mode=False,
            max_num_faces=max_num_faces,
            refine_landmarks=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarks: list[tuple[float, float]] | None = None

    # ── public API ────────────────────────────────────────────────

    def process(self, frame: np.ndarray) -> HeadPose | None:
        """Estimate head pose from a BGR frame.

        Parameters
        ----------
        frame:
            BGR image of shape ``(H, W, 3)``, dtype ``uint8``.

        Returns
        -------
        HeadPose | None
            Euler angles in degrees, or ``None`` if no face is detected.
        """
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb)

        if results.multi_face_landmarks is None:  # type: ignore[union-attr]
            self._landmarks = None
            return None

        face = results.multi_face_landmarks[0]  # type: ignore[union-attr]

        # Extract 2-D projections of the six canonical points
        image_points = np.array(
            [
                [face.landmark[idx].x * w, face.landmark[idx].y * h]
                for idx in _LANDMARK_IDS
            ],
            dtype=np.float64,
        )

        # Store all landmarks for overlay drawing
        self._landmarks = [
            (face.landmark[i].x, face.landmark[i].y)
            for i in range(len(face.landmark))
        ]

        # Camera intrinsics (approximate)
        focal_length = float(w)
        cx, cy = w / 2.0, h / 2.0
        camera_matrix = np.array(
            [
                [focal_length, 0.0, cx],
                [0.0, focal_length, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rotation_vec, _ = cv2.solvePnP(
            _MODEL_POINTS,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return None

        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        # Decompose rotation matrix → Euler angles
        sy = float(np.sqrt(rotation_mat[0, 0] ** 2 + rotation_mat[1, 0] ** 2))
        singular = sy < 1e-6

        if not singular:
            pitch = float(np.degrees(np.arctan2(rotation_mat[2, 1], rotation_mat[2, 2])))
            yaw = float(np.degrees(np.arctan2(-rotation_mat[2, 0], sy)))
            roll = float(np.degrees(np.arctan2(rotation_mat[1, 0], rotation_mat[0, 0])))
        else:
            pitch = float(np.degrees(np.arctan2(-rotation_mat[1, 2], rotation_mat[1, 1])))
            yaw = float(np.degrees(np.arctan2(-rotation_mat[2, 0], sy)))
            roll = 0.0

        return HeadPose(yaw=yaw, pitch=pitch, roll=roll)

    @property
    def landmarks(self) -> list[tuple[float, float]] | None:
        """Normalised (x, y) landmarks from the last ``process()`` call."""
        return self._landmarks

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._face_mesh.close()
