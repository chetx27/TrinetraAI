"""Ambient-Vision Assistant — main entry point.

Wires together the webcam capture, face tracker, gesture detector,
classifier, action mapper, and overlay into a single real-time event
loop.  Head-pose gestures are resolved by thresholding; hand gestures
go through the trained classifier.

Usage
-----
::

    python main.py --config config/settings.yaml
    python main.py --config config/settings.yaml --no-actions   # preview only
    python main.py --config config/settings.yaml --lstm         # LSTM backend
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

from classifier.gesture_model import GestureModel
from config.loader import AppConfig, load_config
from control.action_mapper import ActionMapper
from utils.metrics import LatencyTracker
from vision.capture import WebcamCapture
from vision.face_tracker import FaceTracker
from vision.gesture_detector import GestureDetector
from vision.overlay import Overlay


def _resolve_head_label(
    yaw: float,
    pitch: float,
    yaw_thresh: float,
    pitch_thresh: float,
) -> str | None:
    """Map head-pose angles to a gesture label via simple thresholding.

    Returns ``None`` if the pose is within the neutral zone.
    """
    if yaw > yaw_thresh:
        return "head_turn_left"
    if yaw < -yaw_thresh:
        return "head_turn_right"
    if pitch > pitch_thresh:
        return "head_nod_down"
    if pitch < -pitch_thresh:
        return "head_tilt_up"
    return None


def run(config: AppConfig, *, execute_actions: bool = True, use_lstm: bool = False) -> None:
    """Main event loop.

    Parameters
    ----------
    config:
        Application configuration object.
    execute_actions:
        If ``False``, actions are suppressed (preview-only mode).
    use_lstm:
        Use LSTM classifier backend.
    """
    # ── initialise components ─────────────────────────────────────
    cam = WebcamCapture(
        index=config.camera.index,
        width=config.camera.width,
        height=config.camera.height,
    )
    face_tracker = FaceTracker(
        max_num_faces=config.face_tracking.max_num_faces,
        min_detection_confidence=config.face_tracking.min_detection_confidence,
        min_tracking_confidence=config.face_tracking.min_tracking_confidence,
    )
    gesture_detector = GestureDetector(
        max_num_hands=config.hand_tracking.max_num_hands,
        min_detection_confidence=config.hand_tracking.min_detection_confidence,
        min_tracking_confidence=config.hand_tracking.min_tracking_confidence,
    )
    overlay = Overlay(config.overlay)
    latency = LatencyTracker(capacity=config.metrics.buffer_size)

    # ── classifier ────────────────────────────────────────────────
    model = GestureModel(
        use_lstm=use_lstm,
        confidence_threshold=config.classifier.confidence_threshold,
    )
    model_path = Path(config.classifier.model_path)
    if model_path.exists():
        model.load(model_path)
        print(f"  ✓ Loaded gesture model from {model_path}")
    else:
        print(
            f"  ⚠ No model found at {model_path} — "
            f"hand gestures will default to 'neutral'.  "
            f"Run 'python -m classifier.trainer --fit' first."
        )

    # ── action mapper ─────────────────────────────────────────────
    mapper = ActionMapper(
        gesture_action_map=config.gesture_action_map,
        actions_config=config.actions,
    )

    # ── cooldown to avoid action spam ─────────────────────────────
    last_action_time: float = 0.0
    action_cooldown_s: float = 0.5  # 500 ms between actions
    last_label: str = "neutral"

    print("\n  Ambient-Vision Assistant running.  Press ESC to quit.\n")

    try:
        while True:
            t0 = time.perf_counter()

            frame = cam.read()

            # ── face tracking ─────────────────────────────────────
            head_pose = face_tracker.process(frame)
            head_label: str | None = None
            if head_pose is not None:
                head_label = _resolve_head_label(
                    head_pose.yaw,
                    head_pose.pitch,
                    config.face_tracking.yaw_threshold,
                    config.face_tracking.pitch_threshold,
                )

            # ── hand gesture detection ────────────────────────────
            hand_features = gesture_detector.process(frame)
            hand_label: str = "neutral"
            if hand_features is not None and model_path.exists():
                hand_label = model.predict(hand_features)

            # ── fuse: head gesture takes priority ─────────────────
            label = head_label if head_label is not None else hand_label

            # ── execute action (with cooldown) ────────────────────
            now = time.perf_counter()
            if (
                execute_actions
                and label != "neutral"
                and (now - last_action_time) > action_cooldown_s
            ):
                mapper.execute(label)
                last_action_time = now

            last_label = label

            # ── measure latency ───────────────────────────────────
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latency.record(elapsed_ms)

            # ── overlay + display ─────────────────────────────────
            overlay.draw(
                frame,
                label=last_label,
                latency_ms=elapsed_ms,
                face_landmarks=face_tracker.landmarks,
                hand_landmarks=gesture_detector.landmarks,
            )
            cv2.imshow("Ambient-Vision Assistant", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

    except KeyboardInterrupt:
        pass
    finally:
        # ── print session stats ───────────────────────────────────
        stats = latency.summary()
        print("\n── Session Latency Stats ─────────────────────────────")
        print(f"   Mean : {stats['mean_ms']:6.1f} ms")
        print(f"   P95  : {stats['p95_ms']:6.1f} ms")
        print(f"   P99  : {stats['p99_ms']:6.1f} ms")
        print(f"   Max  : {stats['max_ms']:6.1f} ms")
        print(f"   Frames tracked: {latency.count}\n")

        cam.release()
        face_tracker.close()
        gesture_detector.close()
        cv2.destroyAllWindows()


def main(argv: list[str] | None = None) -> None:
    """Parse CLI arguments and launch the event loop."""
    parser = argparse.ArgumentParser(
        description="Ambient-Vision Assistant — webcam-based desktop control."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/settings.yaml",
        help="Path to settings YAML (default: config/settings.yaml).",
    )
    parser.add_argument(
        "--no-actions",
        action="store_true",
        help="Preview-only mode: suppress all OS actions.",
    )
    parser.add_argument(
        "--lstm",
        action="store_true",
        help="Use Keras LSTM backend instead of SVM.",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    run(config, execute_actions=not args.no_actions, use_lstm=args.lstm)


if __name__ == "__main__":
    main()
