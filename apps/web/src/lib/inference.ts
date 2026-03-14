import { FeatureVector, SessionAssessment } from "../types";

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

export function scoreSession(features: FeatureVector): SessionAssessment {
  let score = 0.5;
  const flaggedSignals: string[] = [];

  if (features.sampleCount > 20) score += 0.1;
  if (features.mouseLinearity < 0.92) score += 0.12;
  if (features.mouseLinearity > 0.98) {
    score -= 0.18;
    flaggedSignals.push("mouse_linearity");
  }

  if (features.averageKeyInterval > 40) score += 0.08;
  if (features.averageKeyInterval > 0 && features.averageKeyInterval < 20) {
    score -= 0.1;
    flaggedSignals.push("typing_rhythm");
  }

  if (features.timeToFirstInteractionMs !== null && features.timeToFirstInteractionMs < 300) {
    score -= 0.08;
    flaggedSignals.push("time_to_first_interaction");
  }

  if (features.visibilityChanges > 3) {
    score -= 0.04;
    flaggedSignals.push("tab_visibility");
  }

  const boundedScore = clamp(score, 0.01, 0.99);
  const verdict =
    boundedScore >= 0.75 ? "human" : boundedScore <= 0.45 ? "bot" : "review";

  return {
    score: boundedScore,
    verdict,
    flaggedSignals,
    features
  };
}
