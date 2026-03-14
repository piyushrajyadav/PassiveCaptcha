from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    pass

from ..schemas import FeatureResponse, InferenceRequest, InferenceResponse, ModelScores
from .model import (
    TORCH_AVAILABLE,
    compute_shap_values,
    feature_row,
    load_lstm_artifacts,
    load_rf_artifacts,
    load_xgb_artifacts,
    prepare_lstm_input,
)

# ── Ensemble weights ──────────────────────────────────────────────────────────
# Must sum to 1.0.  XGBoost gets the highest weight because it uses calibrated
# probabilities and auto-tuned precision/recall thresholds.  The LSTM adds
# sequence-level signal that tabular features can't capture.  RF acts as a
# diversity anchor to reduce variance.
_WEIGHT_RF: float = 0.25
_WEIGHT_XGB: float = 0.45
_WEIGHT_LSTM: float = 0.30


# ── Utilities ─────────────────────────────────────────────────────────────────


def _clamp(value: float, lo: float = 0.01, hi: float = 0.99) -> float:
    return min(max(value, lo), hi)


def _ensemble_score(
    rf: float | None,
    xgb: float | None,
    lstm: float | None,
) -> float:
    """
    Weighted average of available model scores.

    Weights are re-normalised on the fly so the result is always in [0, 1]
    regardless of which subset of models produced a score.  If no model
    produced a score at all, returns 0.5 (maximum uncertainty).
    """
    pool: list[tuple[float, float]] = []

    if rf is not None:
        pool.append((_WEIGHT_RF, rf))
    if xgb is not None:
        pool.append((_WEIGHT_XGB, xgb))
    if lstm is not None:
        pool.append((_WEIGHT_LSTM, lstm))

    if not pool:
        return 0.5

    total_weight = sum(w for w, _ in pool)
    return sum((w / total_weight) * s for w, s in pool)


def _verdict(
    score: float,
    human_threshold: float,
    review_threshold: float,
) -> Literal["human", "bot", "review"]:
    if score >= human_threshold:
        return "human"
    if score <= review_threshold:
        return "bot"
    return "review"


# ── Main entry point ──────────────────────────────────────────────────────────


def score_features(
    features: FeatureResponse,
    payload: InferenceRequest | None = None,
) -> InferenceResponse:
    """
    Score a feature vector against every available model and return a full
    InferenceResponse that includes:

    • ensemble verdict and score
    • per-model probability breakdown (ModelScores)
    • SHAP feature contributions (shapValues)
    • list of individually flagged signals

    Falls back to the hand-crafted heuristic when no trained artifacts exist,
    so the API is always usable — even before any training run has been executed.
    """

    rf_score: float | None = None
    xgb_score: float | None = None
    lstm_score: float | None = None

    # The best thresholds we can find — will be upgraded as better models load
    human_threshold: float = 0.75
    review_threshold: float = 0.45

    # ── 1. RandomForest ───────────────────────────────────────────────────────
    rf_arts = load_rf_artifacts()
    if rf_arts is not None:
        rf_model, rf_cols, rf_thresh = rf_arts
        human_threshold = float(rf_thresh.get("human", 0.75))
        review_threshold = float(rf_thresh.get("review", 0.45))
        try:
            rf_score = float(
                rf_model.predict_proba([feature_row(features, rf_cols)])[0][1]
            )
        except Exception:
            rf_score = None

    # ── 2. XGBoost ────────────────────────────────────────────────────────────
    # XGBoost thresholds are precision/recall-optimised, so prefer them when
    # both models are present.
    xgb_arts = load_xgb_artifacts()
    if xgb_arts is not None:
        xgb_model, xgb_cols, xgb_thresh = xgb_arts
        human_threshold = float(xgb_thresh.get("human", human_threshold))
        review_threshold = float(xgb_thresh.get("review", review_threshold))
        try:
            xgb_score = float(
                xgb_model.predict_proba([feature_row(features, xgb_cols)])[0][1]
            )
        except Exception:
            xgb_score = None

    # ── 3. LSTM (mouse-sequence model) ────────────────────────────────────────
    # Only runs when torch is installed, the .pt artifact exists, and the raw
    # request payload was forwarded from the API endpoint.
    if TORCH_AVAILABLE and payload is not None:
        lstm_arts = load_lstm_artifacts()
        if lstm_arts is not None:
            lstm_model, _ = lstm_arts
            x = prepare_lstm_input(payload)
            if x is not None:
                try:
                    import torch

                    with torch.no_grad():
                        lstm_score = float(torch.sigmoid(lstm_model(x)).item())
                except Exception:
                    lstm_score = None

    # ── 4. Ensemble ───────────────────────────────────────────────────────────
    has_any_model = any(s is not None for s in (rf_score, xgb_score, lstm_score))

    if has_any_model:
        raw_ensemble = _ensemble_score(rf_score, xgb_score, lstm_score)
        ensemble = _clamp(raw_ensemble)
        flagged = _collect_flags(features)
        shap_vals = compute_shap_values(features)

        return InferenceResponse(
            score=ensemble,
            verdict=_verdict(ensemble, human_threshold, review_threshold),
            flaggedSignals=flagged,
            features=features,
            modelScores=ModelScores(
                randomForest=rf_score,
                xgboost=xgb_score,
                lstm=lstm_score,
                ensemble=ensemble,
            ),
            shapValues=shap_vals,
        )

    # ── 5. Heuristic fallback (no artifacts present) ──────────────────────────
    heuristic = _clamp(_heuristic_score(features))
    flagged = _collect_flags(features)

    return InferenceResponse(
        score=heuristic,
        verdict=_verdict(heuristic, 0.75, 0.45),
        flaggedSignals=flagged,
        features=features,
        modelScores=ModelScores(
            randomForest=None,
            xgboost=None,
            lstm=None,
            ensemble=heuristic,
        ),
        shapValues={},
    )


# ── Heuristic scorer ──────────────────────────────────────────────────────────


def _heuristic_score(features: FeatureResponse) -> float:  # noqa: C901
    """
    Rule-based score used as a fallback when no ML artifacts are present.

    Score starts at 0.50 (maximum uncertainty).
    Each rule nudges it up (more human) or down (more bot).
    """
    score = 0.5

    # ── Sample volume ─────────────────────────────────────────────────────────
    if features.sampleCount > 30:
        score += 0.06
    if features.sampleCount > 80:
        score += 0.04

    # ── Mouse linearity ───────────────────────────────────────────────────────
    # Bots move in perfectly straight lines; humans curve and self-correct.
    if features.mouseLinearity < 0.90:
        score += 0.08
    if features.mouseLinearity > 0.98:
        score -= 0.16

    # ── Mouse speed standard deviation ───────────────────────────────────────
    # Constant speed (low std) is robotic; humans accelerate and decelerate.
    if features.mouseSpeedStd > 0.5:
        score += 0.07
    if features.mouseSpeedStd < 0.05 and features.sampleCount > 20:
        score -= 0.12

    # ── Mouse acceleration ────────────────────────────────────────────────────
    if features.mouseAccelerationMean > 0.1:
        score += 0.06
    if features.mouseAccelerationMean < 0.01 and features.sampleCount > 20:
        score -= 0.10

    # ── Mouse jerk ────────────────────────────────────────────────────────────
    if features.mouseJerkMean > 0.05:
        score += 0.04
    if features.mouseJerkMean < 0.005 and features.sampleCount > 20:
        score -= 0.08

    # ── Scroll behaviour ──────────────────────────────────────────────────────
    if features.scrollEvents > 3:
        score += 0.05
    if features.scrollDirectionChanges > 1:
        score += 0.06
    if features.scrollEvents > 5 and features.scrollVelocityStd < 0.01:
        score -= 0.10

    # ── Typing cadence ────────────────────────────────────────────────────────
    if features.averageKeyInterval > 40 and features.keyIntervalStd > 20:
        score += 0.08
    if 0 < features.averageKeyInterval < 20:
        score -= 0.10
    if features.keyEvents > 4 and features.keyIntervalStd < 5:
        score -= 0.10

    # ── Key hold duration ─────────────────────────────────────────────────────
    # Humans hold keys 60–200 ms; programmatic presses are near-instantaneous.
    if 50 < features.keyHoldDurationMean < 300:
        score += 0.06
    if 0 < features.keyHoldDurationMean < 15:
        score -= 0.10

    # ── Time to first interaction ─────────────────────────────────────────────
    if (
        features.timeToFirstInteractionMs is not None
        and features.timeToFirstInteractionMs > 400
    ):
        score += 0.05
    if (
        features.timeToFirstInteractionMs is not None
        and features.timeToFirstInteractionMs < 300
    ):
        score -= 0.08

    # ── Tab / window visibility ───────────────────────────────────────────────
    if features.visibilityChanges > 3:
        score -= 0.04
    if (
        features.visibilityChanges > 1
        and features.meanTimeBetweenVisibilityChanges > 0
        and features.meanTimeBetweenVisibilityChanges < 200
    ):
        score -= 0.08

    # ── Browser entropy ───────────────────────────────────────────────────────
    if features.hasWebGL == 0:
        score -= 0.10
    if 0.99 < features.screenViewportRatio <= 1.0:
        score -= 0.07
    if features.devicePixelRatio > 1:
        score += 0.04
    if features.devicePixelRatio == 1 and features.hasWebGL == 0:
        score -= 0.05
    if features.touchPoints > 0:
        score += 0.05
    if features.isSuspiciousUA == 1:
        score -= 0.25

    return score


# ── Shared flag collector ─────────────────────────────────────────────────────


def _collect_flags(features: FeatureResponse) -> list[str]:  # noqa: C901
    """
    Rule-based signal flags applied on top of the ML probability score.

    These flags are included in every InferenceResponse regardless of which
    scorer produced the final probability, giving operators a human-readable
    explanation of which signals were individually suspicious.
    """
    flagged: list[str] = []

    # Mouse
    if features.mouseLinearity > 0.98:
        flagged.append("mouse_linearity")
    if features.mouseSpeedStd < 0.05 and features.sampleCount > 20:
        flagged.append("mouse_speed_constant")
    if features.mouseAccelerationMean < 0.01 and features.sampleCount > 20:
        flagged.append("mouse_acceleration_flat")
    if features.mouseJerkMean < 0.005 and features.sampleCount > 20:
        flagged.append("mouse_jerk_flat")

    # Scroll
    if features.scrollEvents > 5 and features.scrollVelocityStd < 0.01:
        flagged.append("scroll_velocity_constant")

    # Keyboard
    if 0 < features.averageKeyInterval < 20:
        flagged.append("typing_too_fast")
    if features.keyEvents > 4 and features.keyIntervalStd < 5:
        flagged.append("typing_rhythm_constant")
    if 0 < features.keyHoldDurationMean < 15:
        flagged.append("key_hold_too_short")

    # Timing
    if (
        features.timeToFirstInteractionMs is not None
        and features.timeToFirstInteractionMs < 300
    ):
        flagged.append("time_to_first_interaction")

    # Visibility
    if features.visibilityChanges > 3:
        flagged.append("tab_visibility_frequent")
    if (
        features.visibilityChanges > 1
        and features.meanTimeBetweenVisibilityChanges > 0
        and features.meanTimeBetweenVisibilityChanges < 200
    ):
        flagged.append("tab_switching_rapid")

    # Browser entropy
    if features.hasWebGL == 0:
        flagged.append("no_webgl")
    if 0.99 < features.screenViewportRatio <= 1.0:
        flagged.append("viewport_matches_screen")
    if features.devicePixelRatio == 1 and features.hasWebGL == 0:
        flagged.append("dpr_one_no_webgl")
    if features.isSuspiciousUA == 1:
        flagged.append("suspicious_user_agent")

    return flagged
