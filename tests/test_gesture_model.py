"""Tests for classifier.gesture_model — SVM gesture classifier."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from classifier.gesture_model import GestureModel


@pytest.fixture()
def trained_model() -> GestureModel:
    """Return a model fitted on simple synthetic data."""
    rng = np.random.RandomState(42)
    # 3 classes, 10-d features, well-separated clusters
    n_per_class = 50
    x_a = rng.randn(n_per_class, 10).astype(np.float32) + 5.0
    x_b = rng.randn(n_per_class, 10).astype(np.float32) - 5.0
    x_c = rng.randn(n_per_class, 10).astype(np.float32)

    features = np.vstack([x_a, x_b, x_c])
    labels = np.array(
        ["open_palm"] * n_per_class
        + ["fist"] * n_per_class
        + ["neutral"] * n_per_class
    )
    model = GestureModel(confidence_threshold=0.3)
    model.fit(features, labels)
    return model


def test_predict_returns_string(trained_model: GestureModel) -> None:
    vec = np.full(10, 5.0, dtype=np.float32)
    label = trained_model.predict(vec)
    assert isinstance(label, str)


def test_predict_correct_class(trained_model: GestureModel) -> None:
    # Should predict "open_palm" for a vector near the cluster centre
    vec = np.full(10, 5.0, dtype=np.float32)
    assert trained_model.predict(vec) == "open_palm"


def test_predict_fist_class(trained_model: GestureModel) -> None:
    vec = np.full(10, -5.0, dtype=np.float32)
    assert trained_model.predict(vec) == "fist"


def test_predict_below_threshold_returns_neutral() -> None:
    """With a very high threshold, everything should be neutral."""
    rng = np.random.RandomState(0)
    features = rng.randn(60, 10).astype(np.float32)
    labels = np.array(["a"] * 20 + ["b"] * 20 + ["c"] * 20)

    model = GestureModel(confidence_threshold=0.99)
    model.fit(features, labels)

    # Predictions on noisy data with 0.99 threshold → mostly neutral
    neutral_count = sum(
        1 for _ in range(20) if model.predict(rng.randn(10).astype(np.float32)) == "neutral"
    )
    assert neutral_count > 10  # most should be neutral


def test_save_and_load(trained_model: GestureModel, tmp_path: Path) -> None:
    model_path = tmp_path / "model.pkl"
    trained_model.save(model_path)
    assert model_path.exists()

    loaded = GestureModel()
    loaded.load(model_path)

    vec = np.full(10, 5.0, dtype=np.float32)
    assert loaded.predict(vec) == trained_model.predict(vec)


def test_unfitted_model_returns_neutral() -> None:
    model = GestureModel()
    vec = np.zeros(63, dtype=np.float32)
    assert model.predict(vec) == "neutral"
