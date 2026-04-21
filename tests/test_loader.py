"""Tests for config.loader — typed config loading with fail-fast validation."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from config.loader import (
    AppConfig,
    CameraConfig,
    ConfigError,
    FaceTrackingConfig,
    load_config,
)


@pytest.fixture()
def valid_yaml(tmp_path: Path) -> Path:
    """Write a minimal valid settings file and return its path."""
    content = textwrap.dedent("""\
        camera:
          index: 0
          width: 640
          height: 480
          fps: 30
        face_tracking:
          max_num_faces: 1
          yaw_threshold: 15.0
          pitch_threshold: 10.0
        hand_tracking:
          max_num_hands: 1
        classifier:
          model_path: "classifier/data/model.pkl"
          use_lstm: false
          confidence_threshold: 0.6
        gesture_action_map:
          neutral: "no_action"
        actions:
          scroll_amount: 5
        overlay:
          show_landmarks: true
        training:
          default_samples: 60
        metrics:
          buffer_size: 300
    """)
    p = tmp_path / "settings.yaml"
    p.write_text(content, encoding="utf-8")
    return p


@pytest.fixture()
def bad_key_yaml(tmp_path: Path) -> Path:
    """YAML with an unknown top-level key."""
    content = textwrap.dedent("""\
        camera:
          index: 0
        bogus_section:
          foo: bar
    """)
    p = tmp_path / "bad.yaml"
    p.write_text(content, encoding="utf-8")
    return p


@pytest.fixture()
def bad_nested_yaml(tmp_path: Path) -> Path:
    """YAML with an unknown key inside a section."""
    content = textwrap.dedent("""\
        camera:
          index: 0
          unknown_field: 42
    """)
    p = tmp_path / "bad_nested.yaml"
    p.write_text(content, encoding="utf-8")
    return p


def test_load_valid_config(valid_yaml: Path) -> None:
    cfg = load_config(valid_yaml)
    assert isinstance(cfg, AppConfig)
    assert cfg.camera.width == 640
    assert cfg.face_tracking.yaw_threshold == 15.0
    assert cfg.gesture_action_map["neutral"] == "no_action"


def test_camera_defaults() -> None:
    cam = CameraConfig()
    assert cam.index == 0
    assert cam.fps == 30


def test_face_tracking_defaults() -> None:
    ft = FaceTrackingConfig()
    assert ft.yaw_threshold == 15.0
    assert ft.pitch_threshold == 10.0


def test_missing_file_raises() -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config("/nonexistent/path.yaml")


def test_unknown_top_level_key_raises(bad_key_yaml: Path) -> None:
    with pytest.raises(ConfigError, match="Unknown top-level"):
        load_config(bad_key_yaml)


def test_unknown_nested_key_raises(bad_nested_yaml: Path) -> None:
    with pytest.raises(ConfigError, match="Unknown key"):
        load_config(bad_nested_yaml)


def test_empty_yaml_raises(tmp_path: Path) -> None:
    p = tmp_path / "empty.yaml"
    p.write_text("", encoding="utf-8")
    with pytest.raises(ConfigError, match="empty"):
        load_config(p)


def test_frozen_config(valid_yaml: Path) -> None:
    cfg = load_config(valid_yaml)
    with pytest.raises(AttributeError):
        cfg.camera = CameraConfig(index=1)  # type: ignore[misc]
