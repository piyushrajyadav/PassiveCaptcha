import { FeatureVector, ModelScores, SessionAssessment } from "../types";

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

/**
 * Heuristic fallback scorer.
 *
 * Used when the FastAPI backend is unreachable.
 * Each check nudges the score up (more human) or down (more bot) and
 * optionally pushes a signal name into flaggedSignals.
 *
 * Score starts at 0.50 (uncertain).
 * ≥ 0.75  → "human"
 * ≤ 0.45  → "bot"
 * between → "review"
 */
export function scoreSession(features: FeatureVector): SessionAssessment {
  let score = 0.5;
  const flaggedSignals: string[] = [];

  // ── Sample volume ──────────────────────────────────────────────────────────
  // A real session accumulates many mixed events quickly.
  if (features.sampleCount > 30) score += 0.06;
  if (features.sampleCount > 80) score += 0.04;

  // ── Mouse linearity ────────────────────────────────────────────────────────
  // Bots move in perfectly straight lines (linearity ≈ 1.0).
  // Humans curve, overshoot, and correct.
  if (features.mouseLinearity < 0.90) score += 0.08;
  if (features.mouseLinearity > 0.98) {
    score -= 0.16;
    flaggedSignals.push("mouse_linearity");
  }

  // ── Mouse speed standard deviation ────────────────────────────────────────
  // Bots driven by easing functions or linear interpolation maintain an
  // almost constant speed → very low std dev.
  // Real mice accelerate and decelerate constantly.
  if (features.mouseSpeedStd > 0.5) score += 0.07;
  if (features.mouseSpeedStd < 0.05 && features.sampleCount > 20) {
    score -= 0.12;
    flaggedSignals.push("mouse_speed_constant");
  }

  // ── Mouse acceleration ─────────────────────────────────────────────────────
  // Near-zero mean acceleration means constant velocity → robotic.
  if (features.mouseAccelerationMean > 0.1) score += 0.06;
  if (features.mouseAccelerationMean < 0.01 && features.sampleCount > 20) {
    score -= 0.10;
    flaggedSignals.push("mouse_acceleration_flat");
  }

  // ── Mouse jerk ────────────────────────────────────────────────────────────
  // Humans have irregular jerk; scripted paths are smooth (near-zero jerk).
  if (features.mouseJerkMean > 0.05) score += 0.04;
  if (features.mouseJerkMean < 0.005 && features.sampleCount > 20) {
    score -= 0.08;
    flaggedSignals.push("mouse_jerk_flat");
  }

  // ── Scroll behaviour ──────────────────────────────────────────────────────
  // Humans scroll variably and reverse direction.
  if (features.scrollEvents > 3) score += 0.05;
  if (features.scrollDirectionChanges > 1) score += 0.06;

  // Bots that scroll do so at perfectly constant velocity.
  if (features.scrollEvents > 5 && features.scrollVelocityStd < 0.01) {
    score -= 0.10;
    flaggedSignals.push("scroll_velocity_constant");
  }

  // ── Typing cadence ─────────────────────────────────────────────────────────
  // Humans have variable inter-key timing.
  if (features.averageKeyInterval > 40 && features.keyIntervalStd > 20) score += 0.08;

  // Very fast, perfectly uniform keystrokes are robotic.
  if (features.averageKeyInterval > 0 && features.averageKeyInterval < 20) {
    score -= 0.10;
    flaggedSignals.push("typing_too_fast");
  }
  if (features.keyEvents > 4 && features.keyIntervalStd < 5) {
    score -= 0.10;
    flaggedSignals.push("typing_rhythm_constant");
  }

  // ── Key hold duration ──────────────────────────────────────────────────────
  // Humans hold keys for 60–200 ms on average.
  // Bots typically produce near-zero hold durations (programmatic keypress).
  if (features.keyHoldDurationMean > 50 && features.keyHoldDurationMean < 300) {
    score += 0.06;
  }
  if (features.keyHoldDurationMean > 0 && features.keyHoldDurationMean < 15) {
    score -= 0.10;
    flaggedSignals.push("key_hold_too_short");
  }

  // ── Time to first interaction ──────────────────────────────────────────────
  // Humans take a moment to orient before interacting (usually > 400 ms).
  // Bots start interacting almost instantly.
  if (
    features.timeToFirstInteractionMs !== null &&
    features.timeToFirstInteractionMs > 400
  ) {
    score += 0.05;
  }
  if (
    features.timeToFirstInteractionMs !== null &&
    features.timeToFirstInteractionMs < 300
  ) {
    score -= 0.08;
    flaggedSignals.push("time_to_first_interaction");
  }

  // ── Tab / window visibility ────────────────────────────────────────────────
  // Occasional tab switching is normal. Rapid automated switching is not.
  if (features.visibilityChanges > 3) {
    score -= 0.04;
    flaggedSignals.push("tab_visibility_frequent");
  }
  // Very short average gap between visibility changes suggests scripted toggling.
  if (
    features.visibilityChanges > 1 &&
    features.meanTimeBetweenVisibilityChanges > 0 &&
    features.meanTimeBetweenVisibilityChanges < 200
  ) {
    score -= 0.08;
    flaggedSignals.push("tab_switching_rapid");
  }

  // ── Browser entropy ────────────────────────────────────────────────────────
  // Headless browsers commonly lack WebGL, have viewport == screen,
  // have DPR exactly 1, and carry automation strings in the UA.

  if (features.hasWebGL === 0) {
    score -= 0.10;
    flaggedSignals.push("no_webgl");
  }

  // Real browsers almost never fill the full screen width exactly.
  if (features.screenViewportRatio > 0.99 && features.screenViewportRatio <= 1.0) {
    score -= 0.07;
    flaggedSignals.push("viewport_matches_screen");
  }

  // Non-integer or unusual DPR is typical of real devices (HiDPI, mobile).
  if (features.devicePixelRatio > 1) score += 0.04;
  if (features.devicePixelRatio === 1 && features.hasWebGL === 0) {
    // Reinforce the headless signal — both together are suspicious.
    score -= 0.05;
    flaggedSignals.push("dpr_one_no_webgl");
  }

  // Touch-capable devices are virtually never headless automation targets.
  if (features.touchPoints > 0) score += 0.05;

  // Hard stop: if the UA contains known automation strings, strong bot signal.
  if (features.isSuspiciousUA === 1) {
    score -= 0.25;
    flaggedSignals.push("suspicious_user_agent");
  }

  // ── Final verdict ──────────────────────────────────────────────────────────
  const boundedScore = clamp(score, 0.01, 0.99);
  const verdict =
    boundedScore >= 0.75 ? "human" : boundedScore <= 0.45 ? "bot" : "review";

  // Heuristic fallback has no ML models — report null for each model slot and
  // use the heuristic score itself as the ensemble value.
  const modelScores: ModelScores = {
    randomForest: null,
    xgboost: null,
    lstm: null,
    ensemble: boundedScore,
  };

  return {
    score: boundedScore,
    verdict,
    flaggedSignals,
    features,
    modelScores,
    // No SHAP values available on the client side — the backend provides these
    // when it is reachable.
    shapValues: {},
  };
}
