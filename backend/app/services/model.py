from __future__ import annotations

import json
from functools import lru_cache
from importlib import util as _importlib_util
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from ..schemas import FeatureResponse, InferenceRequest

# ── Paths ─────────────────────────────────────────────────────────────────────

ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "train" / "artifacts"
SEQUENCE_LENGTH = 128


# ── Availability checks (cached, so the import attempt runs only once) ────────


@lru_cache(maxsize=1)
def _torch_available() -> bool:
    """Return True if torch is importable in the current environment."""
    return _importlib_util.find_spec("torch") is not None


@lru_cache(maxsize=1)
def _shap_available() -> bool:
    """Return True if shap is importable in the current environment."""
    return _importlib_util.find_spec("shap") is not None


# Keep module-level aliases so other modules can read the flags cheaply.
# These are assigned once and never reassigned, so Pyright accepts them.
TORCH_AVAILABLE: bool = _torch_available()
SHAP_AVAILABLE: bool = _shap_available()


# ── Artifact loaders (all LRU-cached — loaded once per process) ───────────────


@lru_cache(maxsize=1)
def load_rf_artifacts() -> tuple[Any, list[str], dict[str, float]] | None:
    """
    Load the RandomForest baseline model, its feature-column list, and the
    human/review decision thresholds saved during training.
    Returns None when any required file is missing.
    """
    model_path = ARTIFACTS_DIR / "baseline_model.joblib"
    columns_path = ARTIFACTS_DIR / "feature_columns.json"
    thresholds_path = ARTIFACTS_DIR / "thresholds.json"

    if not all(p.exists() for p in (model_path, columns_path, thresholds_path)):
        return None

    model: Any = joblib.load(model_path)
    columns: list[str] = json.loads(columns_path.read_text(encoding="utf-8"))
    thresholds: dict[str, float] = json.loads(
        thresholds_path.read_text(encoding="utf-8")
    )
    return model, columns, thresholds


@lru_cache(maxsize=1)
def load_xgb_artifacts() -> tuple[Any, list[str], dict[str, float]] | None:
    """
    Load the calibrated XGBoost model, its feature-column list, and the
    precision-recall optimised thresholds saved during training.
    Returns None when any required file is missing.
    """
    model_path = ARTIFACTS_DIR / "xgboost_calibrated_model.joblib"
    columns_path = ARTIFACTS_DIR / "xgboost_feature_columns.json"
    thresholds_path = ARTIFACTS_DIR / "xgboost_thresholds.json"

    if not all(p.exists() for p in (model_path, columns_path, thresholds_path)):
        return None

    model: Any = joblib.load(model_path)
    columns: list[str] = json.loads(columns_path.read_text(encoding="utf-8"))
    thresholds: dict[str, float] = json.loads(
        thresholds_path.read_text(encoding="utf-8")
    )
    return model, columns, thresholds


@lru_cache(maxsize=1)
def load_lstm_artifacts() -> tuple[Any, dict[str, Any]] | None:
    """
    Load the PyTorch LSTM mouse-sequence model and its sequence config.

    The LSTMClassifier class is defined locally here so that importing this
    module never fails when torch is not installed.  The class definition is
    only reached after the _torch_available() guard passes.

    Returns None when torch is unavailable or the .pt file is missing.
    """
    if not _torch_available():
        return None

    model_path = ARTIFACTS_DIR / "lstm_mouse_model.pt"
    config_path = ARTIFACTS_DIR / "lstm_sequence_config.json"

    if not model_path.exists():
        return None

    # Local import — safe because _torch_available() is True at this point.
    import torch  # noqa: PLC0415
    import torch.nn as nn  # noqa: PLC0415

    config: dict[str, Any] = {}
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))

    input_size: int = int(config.get("input_size", 2))
    hidden_size: int = int(config.get("hidden_size", 48))

    # ── Inline LSTMClassifier definition ─────────────────────────────────────
    # Architecture must stay byte-for-byte identical to train/train_lstm.py
    # so that saved state-dicts load correctly.

    class LSTMClassifier(nn.Module):
        def __init__(self, in_size: int, h_size: int) -> None:
            super().__init__()
            self.lstm: nn.LSTM = nn.LSTM(
                input_size=in_size, hidden_size=h_size, batch_first=True
            )
            self.head: nn.Sequential = nn.Sequential(
                nn.Linear(h_size, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            _, (h, _) = self.lstm(x)
            return self.head(h[-1]).squeeze(1)  # type: ignore[no-any-return]

    model = LSTMClassifier(input_size, hidden_size)
    # weights_only=True is required since PyTorch 2.x for security.
    _ = model.load_state_dict(
        torch.load(model_path, map_location="cpu", weights_only=True)
    )
    model.eval()
    return model, config


# ── Backward-compatibility shim ───────────────────────────────────────────────


def load_artifacts() -> tuple[Any, list[str], dict[str, float]] | None:
    """Legacy entry-point used by the original scoring.py; delegates to RF."""
    return load_rf_artifacts()


# ── SHAP explainer loaders (cached) ──────────────────────────────────────────


@lru_cache(maxsize=1)
def load_shap_rf() -> tuple[Any, list[str]] | None:
    """
    Build and cache a SHAP TreeExplainer for the RandomForest model.
    Returns None when SHAP is unavailable or the RF artifact is missing.
    """
    if not _shap_available():
        return None

    arts = load_rf_artifacts()
    if arts is None:
        return None

    import importlib as _il

    shap_lib = _il.import_module("shap")

    model, columns, _ = arts
    try:
        explainer: Any = shap_lib.TreeExplainer(model)
        return explainer, columns
    except Exception:
        return None


@lru_cache(maxsize=1)
def load_shap_xgb() -> tuple[Any, list[str]] | None:
    """
    Build and cache a SHAP TreeExplainer for the XGBoost base estimator.

    sklearn's CalibratedClassifierCV wraps the base model; we extract the
    first fold's XGBClassifier for SHAP.  The direction of each feature's
    contribution is accurate even though absolute magnitudes reflect the
    uncalibrated probability scale.
    Returns None when SHAP is unavailable or the XGB artifact is missing.
    """
    if not _shap_available():
        return None

    arts = load_xgb_artifacts()
    if arts is None:
        return None

    import importlib as _il

    shap_lib = _il.import_module("shap")

    calibrated_model, columns, _ = arts
    try:
        base_model: Any = calibrated_model.calibrated_classifiers_[0].estimator
        explainer: Any = shap_lib.TreeExplainer(base_model)
        return explainer, columns
    except Exception:
        return None


# ── Feature helpers ───────────────────────────────────────────────────────────


def feature_row(features: FeatureResponse, columns: list[str]) -> list[float]:
    """
    Convert a FeatureResponse into a flat float list aligned with `columns`.

    Uses .get() with a 0.0 default so that columns absent from an older
    artifact after a feature-schema upgrade don't raise KeyError.
    """
    values: dict[str, Any] = features.model_dump()
    return [float(values.get(col, 0.0)) for col in columns]


# ── LSTM sequence preparation ─────────────────────────────────────────────────


def _normalize_sequence(
    points: list[tuple[float, float]],
    limit: int = SEQUENCE_LENGTH,
) -> list[list[float]]:
    """
    Min-max normalise a list of (x, y) mouse coordinates and return a fixed-
    length list of [x_norm, y_norm] pairs.

    • If len(points) > limit  → subsample evenly
    • If len(points) < limit  → pad by repeating the last point

    Matches the preprocessing in train/train_lstm.py exactly.
    """
    if not points:
        return [[0.0, 0.0]] * limit

    xs: list[float] = [p[0] for p in points]
    ys: list[float] = [p[1] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    rx = max(max_x - min_x, 1.0)
    ry = max(max_y - min_y, 1.0)

    normed: list[list[float]] = [
        [(x - min_x) / rx, (y - min_y) / ry] for x, y in points
    ]

    if len(normed) >= limit:
        step = len(normed) / limit
        return [normed[int(i * step)] for i in range(limit)]

    while len(normed) < limit:
        normed.append(normed[-1])
    return normed[:limit]


def prepare_lstm_input(payload: InferenceRequest) -> Any | None:
    """
    Extract mouse-move coordinates from the raw request, normalise them, and
    return a (1, SEQUENCE_LENGTH, 2) float32 tensor ready for the LSTM.

    Returns None when:
      • torch is not installed
      • fewer than 5 move events were recorded (not enough for a meaningful
        sequence inference)
    """
    if not _torch_available():
        return None

    points: list[tuple[float, float]] = [
        (s.x, s.y) for s in payload.pointer if s.type == "move"
    ]
    if len(points) < 5:
        return None

    import torch  # noqa: PLC0415

    sequence = _normalize_sequence(points, SEQUENCE_LENGTH)
    return torch.tensor([sequence], dtype=torch.float32)  # shape: (1, 128, 2)


# ── SHAP value computation ────────────────────────────────────────────────────


def compute_shap_values(
    features: FeatureResponse,
    top_n: int = 12,
) -> dict[str, float]:
    """
    Return the top `top_n` SHAP feature contributions for the best available
    tabular model as a {feature_name: signed_contribution} dict.

    • Positive value  → feature pushed the score toward "human"
    • Negative value  → feature pushed the score toward "bot"

    Preference order: XGBoost > RandomForest > feature_importances_ fallback.
    Returns an empty dict when nothing is available.
    """
    # XGBoost SHAP first (sharper values from a boosting model)
    result = _shap_from_explainer(load_shap_xgb(), features, top_n)
    if result:
        return result

    # Fall back to RandomForest SHAP
    result = _shap_from_explainer(load_shap_rf(), features, top_n)
    if result:
        return result

    # Last resort: unsigned feature_importances_ from any available model
    return _fallback_importances(top_n)


def _shap_from_explainer(
    bundle: tuple[Any, list[str]] | None,
    features: FeatureResponse,
    top_n: int,
) -> dict[str, float]:
    """
    Run SHAP on a single sample and return the top_n signed contributions.

    Handles the two different return shapes TreeExplainer produces:
      • RandomForest binary → list of two (1, n_features) arrays;
                              index 1 = human class contributions
      • XGBoost binary      → single (1, n_features) array
    """
    if bundle is None:
        return {}

    explainer, columns = bundle
    row: Any = np.array([feature_row(features, columns)])  # shape (1, n_features)

    try:
        raw: Any = explainer.shap_values(row)

        # Normalise to a flat Python list[float] for the human class
        if isinstance(raw, list):
            # RandomForest: raw[0]=bot contributions, raw[1]=human contributions
            raw_vals: Any = raw[1][0] if len(raw) > 1 else raw[0][0]
        else:
            # XGBoost: single 2-D array — take the first (only) row
            raw_vals = raw[0]

        float_vals: list[float] = [float(v) for v in raw_vals]
        pairs: list[tuple[str, float]] = sorted(
            zip(columns, float_vals),
            key=lambda item: abs(item[1]),
            reverse=True,
        )
        return {name: round(val, 6) for name, val in pairs[:top_n]}

    except Exception:
        return {}


def _fallback_importances(top_n: int) -> dict[str, float]:
    """
    When SHAP is unavailable, fall back to unsigned model.feature_importances_.
    Tries XGBoost first, then RandomForest.

    Values are positive importance fractions (not signed SHAP contributions),
    but still useful for showing which features the model weighted most heavily.
    """
    for loader in (load_xgb_artifacts, load_rf_artifacts):
        arts = loader()
        if arts is None:
            continue
        model, columns, _ = arts
        try:
            base: Any = getattr(model, "calibrated_classifiers_", None)
            raw_imp: Any = (
                base[0].estimator.feature_importances_
                if base is not None
                else model.feature_importances_
            )
            float_imp: list[float] = [float(v) for v in raw_imp]
            pairs: list[tuple[str, float]] = sorted(
                zip(columns, float_imp),
                key=lambda item: abs(item[1]),
                reverse=True,
            )
            return {name: round(val, 6) for name, val in pairs[:top_n]}
        except Exception:
            continue

    return {}
