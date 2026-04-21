"""CLI tool for recording gesture samples and training the classifier.

Workflow
--------
1. Opens the webcam, shows a live overlay.
2. Counts down 3 seconds, then records *N* frames of landmarks.
3. Saves the feature array to ``data/{gesture}.npy``.
4. When ``--fit`` is passed (or all gesture files exist), fits the SVM
   and prints ``sklearn.metrics.classification_report``.

Usage
-----
::

    # Record one gesture class
    python -m classifier.trainer --gesture open_palm --samples 60

    # Fit after all classes have been recorded
    python -m classifier.trainer --fit
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import classification_report

from classifier.gesture_model import GestureModel
from config.loader import AppConfig, load_config
from vision.capture import WebcamCapture
from vision.gesture_detector import GestureDetector
from vision.overlay import Overlay

# The full set of gestures that involve hand landmarks
_HAND_GESTURES: set[str] = {"open_palm", "fist", "peace_sign", "neutral"}


def _record_gesture(
    gesture: str,
    num_samples: int,
    config: AppConfig,
) -> np.ndarray:
    """Open the webcam, count down, and record *num_samples* frames.

    Returns
    -------
    np.ndarray
        Feature matrix of shape ``(num_samples, 63)``.
    """
    cam = WebcamCapture(
        index=config.camera.index,
        width=config.camera.width,
        height=config.camera.height,
    )
    detector = GestureDetector(
        max_num_hands=config.hand_tracking.max_num_hands,
        min_detection_confidence=config.hand_tracking.min_detection_confidence,
        min_tracking_confidence=config.hand_tracking.min_tracking_confidence,
    )
    overlay = Overlay(config.overlay)

    features: list[np.ndarray] = []
    countdown = config.training.countdown_seconds

    print(f"\n  Recording gesture: '{gesture}'")
    print(f"  Samples needed:    {num_samples}")
    print(f"  Countdown:         {countdown} s\n")

    # ── countdown phase ───────────────────────────────────────────
    start = time.monotonic()
    while time.monotonic() - start < countdown:
        frame = cam.read()
        remaining = countdown - int(time.monotonic() - start)
        cv2.putText(
            frame,
            f"Get ready: {remaining}",
            (180, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 0, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.imshow("Trainer", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            cam.release()
            detector.close()
            cv2.destroyAllWindows()
            sys.exit(0)

    # ── recording phase ───────────────────────────────────────────
    print("  ▶ Recording …")
    while len(features) < num_samples:
        frame = cam.read()
        vec = detector.process(frame)
        overlay.draw(
            frame,
            label=f"Recording {gesture} [{len(features)}/{num_samples}]",
            hand_landmarks=detector.landmarks,
        )
        cv2.imshow("Trainer", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        if vec is not None:
            features.append(vec)

    cam.release()
    detector.close()
    cv2.destroyAllWindows()

    if len(features) < num_samples:
        print(
            f"  ⚠  Only captured {len(features)}/{num_samples} samples "
            f"(hand may not have been detected in some frames)."
        )

    return np.array(features, dtype=np.float32)


def _fit_model(data_dir: Path, model_path: Path, use_lstm: bool) -> None:
    """Load all ``.npy`` files from *data_dir*, fit, evaluate, and save."""
    npy_files = sorted(data_dir.glob("*.npy"))
    if not npy_files:
        print("  ✗ No .npy data files found.  Record gestures first.")
        sys.exit(1)

    all_features: list[np.ndarray] = []
    all_labels: list[str] = []

    for fp in npy_files:
        gesture_name = fp.stem
        arr = np.load(fp)
        all_features.append(arr)
        all_labels.extend([gesture_name] * arr.shape[0])
        print(f"  Loaded {arr.shape[0]:>4d} samples for '{gesture_name}'")

    x = np.vstack(all_features)
    y = np.array(all_labels)

    model = GestureModel(use_lstm=use_lstm)
    model.fit(x, y)

    # Evaluate on training data (held-out eval requires more data)
    preds = [model.predict(row) for row in x]
    report: str = classification_report(y, preds)
    print("\n── Classification Report ──────────────────────────────")
    print(report)

    model.save(model_path)
    print(f"  ✓ Model saved to {model_path}\n")


def main(argv: list[str] | None = None) -> None:
    """Entry point for ``python -m classifier.trainer``."""
    parser = argparse.ArgumentParser(
        description="Record gesture samples and/or train the classifier."
    )
    parser.add_argument(
        "--gesture",
        type=str,
        default=None,
        help="Name of the gesture class to record (e.g. 'open_palm').",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=60,
        help="Number of frames to record (default: 60).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="classifier/data",
        help="Directory to save .npy feature files.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/settings.yaml",
        help="Path to settings YAML.",
    )
    parser.add_argument(
        "--fit",
        action="store_true",
        help="Fit the model on all recorded data (skip recording).",
    )
    parser.add_argument(
        "--lstm",
        action="store_true",
        help="Use Keras LSTM backend instead of SVM.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to save the trained model (default: from config).",
    )

    args = parser.parse_args(argv)
    config = load_config(args.config)

    data_dir = Path(args.output)
    data_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(args.model_path) if args.model_path else Path(config.classifier.model_path)

    # ── record mode ───────────────────────────────────────────────
    if args.gesture and not args.fit:
        if args.gesture not in _HAND_GESTURES:
            print(
                f"  ⚠  '{args.gesture}' is a head-pose gesture — "
                f"head gestures are detected by thresholds, not the classifier. "
                f"Only hand gestures need training data: {sorted(_HAND_GESTURES)}"
            )
            sys.exit(1)

        features = _record_gesture(args.gesture, args.samples, config)
        out_path = data_dir / f"{args.gesture}.npy"
        np.save(out_path, features)
        print(f"  ✓ Saved {features.shape[0]} samples → {out_path}\n")

        # Auto-fit if all hand gesture files are present
        existing = {fp.stem for fp in data_dir.glob("*.npy")}
        if _HAND_GESTURES.issubset(existing):
            print("  All hand gesture classes recorded — fitting model …\n")
            _fit_model(data_dir, model_path, args.lstm)
        else:
            missing = _HAND_GESTURES - existing
            print(f"  Still need recordings for: {sorted(missing)}\n")
        return

    # ── fit-only mode ─────────────────────────────────────────────
    if args.fit:
        _fit_model(data_dir, model_path, args.lstm)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
