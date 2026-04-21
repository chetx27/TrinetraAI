"""Typed configuration loader with fail-fast validation.

Reads ``settings.yaml`` and maps every section to a frozen dataclass.
Unknown or missing keys raise ``ConfigError`` immediately so that the user
gets an actionable error message instead of a silent default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(Exception):
    """Raised when the configuration file is invalid or missing keys."""


# ── Section dataclasses ───────────────────────────────────────────────


@dataclass(frozen=True)
class CameraConfig:
    index: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30


@dataclass(frozen=True)
class FaceTrackingConfig:
    max_num_faces: int = 1
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    yaw_threshold: float = 15.0
    pitch_threshold: float = 10.0


@dataclass(frozen=True)
class HandTrackingConfig:
    max_num_hands: int = 1
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


@dataclass(frozen=True)
class ClassifierConfig:
    model_path: str = "classifier/data/model.pkl"
    use_lstm: bool = False
    confidence_threshold: float = 0.6


@dataclass(frozen=True)
class ActionsConfig:
    scroll_amount: int = 5
    app_launcher_hotkey: str = "win"


@dataclass(frozen=True)
class OverlayConfig:
    show_landmarks: bool = True
    show_label: bool = True
    show_latency: bool = True
    font_scale: float = 0.7
    font_thickness: int = 2
    label_color: tuple[int, int, int] = (0, 255, 0)
    latency_color: tuple[int, int, int] = (255, 255, 0)


@dataclass(frozen=True)
class TrainingConfig:
    default_samples: int = 60
    countdown_seconds: int = 3
    output_dir: str = "classifier/data"


@dataclass(frozen=True)
class MetricsConfig:
    buffer_size: int = 300


# ── Root config ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    face_tracking: FaceTrackingConfig = field(default_factory=FaceTrackingConfig)
    hand_tracking: HandTrackingConfig = field(default_factory=HandTrackingConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    gesture_action_map: dict[str, str] = field(default_factory=dict)
    actions: ActionsConfig = field(default_factory=ActionsConfig)
    overlay: OverlayConfig = field(default_factory=OverlayConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)


# ── Helpers ───────────────────────────────────────────────────────────


def _build_dataclass(cls: type[Any], raw: dict[str, Any] | None, section: str) -> Any:
    """Instantiate *cls* from *raw* dict, raising on unknown keys."""
    if raw is None:
        return cls()
    allowed = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    unknown = set(raw.keys()) - allowed
    if unknown:
        raise ConfigError(
            f"Unknown key(s) in '{section}': {sorted(unknown)}. "
            f"Allowed: {sorted(allowed)}"
        )
    # Convert list→tuple for colour fields
    converted: dict[str, Any] = {}
    for k, v in raw.items():
        if isinstance(v, list):
            converted[k] = tuple(v)
        else:
            converted[k] = v
    return cls(**converted)


def load_config(path: str | Path) -> AppConfig:
    """Load and validate configuration from a YAML file.

    Parameters
    ----------
    path:
        Filesystem path to the YAML settings file.

    Returns
    -------
    AppConfig
        Fully validated, frozen configuration object.

    Raises
    ------
    ConfigError
        If the file is missing, unparseable, or contains invalid keys.
    """
    filepath = Path(path)
    if not filepath.exists():
        raise ConfigError(f"Config file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as fh:
        raw: dict[str, Any] | None = yaml.safe_load(fh)

    if raw is None:
        raise ConfigError(f"Config file is empty: {filepath}")

    if not isinstance(raw, dict):
        raise ConfigError(f"Config root must be a mapping, got {type(raw).__name__}")

    # Validate top-level keys
    allowed_sections = {
        "camera",
        "face_tracking",
        "hand_tracking",
        "classifier",
        "gesture_action_map",
        "actions",
        "overlay",
        "training",
        "metrics",
    }
    unknown_sections = set(raw.keys()) - allowed_sections
    if unknown_sections:
        raise ConfigError(
            f"Unknown top-level section(s): {sorted(unknown_sections)}. "
            f"Allowed: {sorted(allowed_sections)}"
        )

    # Build section configs
    camera = _build_dataclass(CameraConfig, raw.get("camera"), "camera")
    face_tracking = _build_dataclass(
        FaceTrackingConfig, raw.get("face_tracking"), "face_tracking"
    )
    hand_tracking = _build_dataclass(
        HandTrackingConfig, raw.get("hand_tracking"), "hand_tracking"
    )
    classifier = _build_dataclass(
        ClassifierConfig, raw.get("classifier"), "classifier"
    )
    actions = _build_dataclass(ActionsConfig, raw.get("actions"), "actions")
    overlay = _build_dataclass(OverlayConfig, raw.get("overlay"), "overlay")
    training = _build_dataclass(TrainingConfig, raw.get("training"), "training")
    metrics = _build_dataclass(MetricsConfig, raw.get("metrics"), "metrics")

    # gesture_action_map is a plain dict
    gesture_map: dict[str, str] = raw.get("gesture_action_map", {})
    if not isinstance(gesture_map, dict):
        raise ConfigError(
            f"'gesture_action_map' must be a mapping, got {type(gesture_map).__name__}"
        )

    return AppConfig(
        camera=camera,
        face_tracking=face_tracking,
        hand_tracking=hand_tracking,
        classifier=classifier,
        gesture_action_map=gesture_map,
        actions=actions,
        overlay=overlay,
        training=training,
        metrics=metrics,
    )
