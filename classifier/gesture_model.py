"""Gesture classifier: SVM (default) or Keras LSTM.

The ``GestureModel`` façade exposes a uniform ``predict / save / load``
interface regardless of the backend.  The SVM path uses scikit-learn's
``SVC`` with an RBF kernel and probability calibration so that we can
apply a confidence threshold to suppress false positives.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def _build_svm_pipeline() -> Pipeline:
    """Construct a fresh StandardScaler → RBF-SVM pipeline."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(
                    kernel="rbf",
                    C=10.0,
                    gamma="scale",
                    probability=True,
                    class_weight="balanced",
                ),
            ),
        ]
    )


class GestureModel:
    """Unified gesture classifier.

    Parameters
    ----------
    use_lstm:
        If ``True``, use a Keras LSTM backend (requires TensorFlow).
        Defaults to SVM.
    confidence_threshold:
        Minimum predicted probability to accept a label.  Below this
        the model returns ``"neutral"``.
    """

    def __init__(
        self,
        use_lstm: bool = False,
        confidence_threshold: float = 0.6,
    ) -> None:
        self._use_lstm: bool = use_lstm
        self._confidence_threshold: float = confidence_threshold
        self._pipeline: Pipeline | None = None
        self._lstm_model: Any = None
        self._labels: list[str] = []

        if not use_lstm:
            self._pipeline = _build_svm_pipeline()

    # ── training ──────────────────────────────────────────────────

    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Fit the model on labelled feature vectors.

        Parameters
        ----------
        features:
            Array of shape ``(N, D)`` — e.g. ``(N, 63)`` for hand gestures.
        labels:
            Array of shape ``(N,)`` with string class labels.
        """
        self._labels = sorted(set(labels.tolist()))

        if self._use_lstm:
            self._fit_lstm(features, labels)
        else:
            if self._pipeline is None:
                self._pipeline = _build_svm_pipeline()
            self._pipeline.fit(features, labels)

    # ── inference ─────────────────────────────────────────────────

    def predict(self, features: np.ndarray) -> str:
        """Classify a single feature vector.

        Parameters
        ----------
        features:
            1-D array of shape ``(D,)`` or 2-D of shape ``(1, D)``.

        Returns
        -------
        str
            Predicted gesture label, or ``"neutral"`` if confidence is
            below threshold.
        """
        x = features.reshape(1, -1)

        if self._use_lstm:
            return self._predict_lstm(x)

        if self._pipeline is None:
            return "neutral"

        proba: np.ndarray = self._pipeline.predict_proba(x)[0]
        max_idx = int(np.argmax(proba))
        confidence = float(proba[max_idx])

        if confidence < self._confidence_threshold:
            return "neutral"

        classes: np.ndarray = self._pipeline.classes_
        return str(classes[max_idx])

    # ── persistence ───────────────────────────────────────────────

    def save(self, path: Path) -> None:
        """Serialise the model to *path* (pickle for SVM, .keras for LSTM)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        if self._use_lstm:
            self._save_lstm(path)
        else:
            with open(path, "wb") as fh:
                pickle.dump(
                    {
                        "pipeline": self._pipeline,
                        "labels": self._labels,
                        "confidence_threshold": self._confidence_threshold,
                    },
                    fh,
                )

    def load(self, path: Path) -> None:
        """Deserialise a model from *path*."""
        if self._use_lstm:
            self._load_lstm(path)
        else:
            with open(path, "rb") as fh:
                data: dict[str, Any] = pickle.load(fh)  # noqa: S301
            self._pipeline = data["pipeline"]
            self._labels = data["labels"]
            self._confidence_threshold = data.get(
                "confidence_threshold", self._confidence_threshold
            )

    # ── LSTM backend (optional) ───────────────────────────────────

    def _fit_lstm(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Build and train a simple Keras LSTM model."""
        try:
            from tensorflow import keras  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "LSTM backend requires TensorFlow.  Install with: "
                "pip install tensorflow"
            ) from exc

        from sklearn.preprocessing import LabelEncoder  # local to avoid top-level cost

        le = LabelEncoder()
        y_enc = le.fit_transform(labels)
        self._labels = le.classes_.tolist()
        num_classes = len(self._labels)

        # Reshape for LSTM: (N, 1 timestep, D features)
        x = features.reshape(features.shape[0], 1, features.shape[1])

        model = keras.Sequential(
            [
                keras.layers.LSTM(64, input_shape=(1, features.shape[1])),
                keras.layers.Dropout(0.3),
                keras.layers.Dense(32, activation="relu"),
                keras.layers.Dense(num_classes, activation="softmax"),
            ]
        )
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.fit(x, y_enc, epochs=30, batch_size=16, validation_split=0.2, verbose=1)
        self._lstm_model = model

    def _predict_lstm(self, features: np.ndarray) -> str:
        if self._lstm_model is None:
            return "neutral"
        x = features.reshape(1, 1, features.shape[1])
        proba: np.ndarray = self._lstm_model.predict(x, verbose=0)[0]
        max_idx = int(np.argmax(proba))
        confidence = float(proba[max_idx])
        if confidence < self._confidence_threshold:
            return "neutral"
        return self._labels[max_idx]

    def _save_lstm(self, path: Path) -> None:
        if self._lstm_model is None:
            return
        keras_path = path.with_suffix(".keras")
        self._lstm_model.save(keras_path)
        meta_path = path.with_suffix(".meta.pkl")
        with open(meta_path, "wb") as fh:
            pickle.dump(
                {"labels": self._labels, "confidence_threshold": self._confidence_threshold},
                fh,
            )

    def _load_lstm(self, path: Path) -> None:
        try:
            from tensorflow import keras  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "LSTM backend requires TensorFlow.  Install with: "
                "pip install tensorflow"
            ) from exc

        keras_path = path.with_suffix(".keras")
        self._lstm_model = keras.models.load_model(keras_path)
        meta_path = path.with_suffix(".meta.pkl")
        if meta_path.exists():
            with open(meta_path, "rb") as fh:
                meta: dict[str, Any] = pickle.load(fh)  # noqa: S301
            self._labels = meta["labels"]
            self._confidence_threshold = meta.get(
                "confidence_threshold", self._confidence_threshold
            )
