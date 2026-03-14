import { FeatureVector, PassiveSignals } from "../types";

function distance(x1: number, y1: number, x2: number, y2: number): number {
  return Math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2);
}

export function extractFeatures(signals: PassiveSignals): FeatureVector {
  const moves = signals.pointer.filter((sample) => sample.type === "move");
  let totalSpeed = 0;
  let speedSamples = 0;
  let directDistance = 0;
  let pathDistance = 0;

  for (let index = 1; index < moves.length; index += 1) {
    const previous = moves[index - 1];
    const current = moves[index];
    const dt = Math.max(current.t - previous.t, 1);
    const dist = distance(previous.x, previous.y, current.x, current.y);
    totalSpeed += dist / dt;
    pathDistance += dist;
    speedSamples += 1;
  }

  if (moves.length > 1) {
    directDistance = distance(
      moves[0].x,
      moves[0].y,
      moves[moves.length - 1].x,
      moves[moves.length - 1].y
    );
  }

  const keyDowns = signals.keyboard.filter((sample) => sample.type === "down");
  let keyIntervals = 0;
  for (let index = 1; index < keyDowns.length; index += 1) {
    keyIntervals += keyDowns[index].t - keyDowns[index - 1].t;
  }

  return {
    sessionId: signals.sessionId,
    sampleCount: signals.pointer.length + signals.keyboard.length + signals.scroll.length,
    mouseSpeedMean: speedSamples === 0 ? 0 : totalSpeed / speedSamples,
    mouseLinearity: pathDistance === 0 ? 0 : directDistance / pathDistance,
    clickCount: signals.pointer.filter((sample) => sample.type === "click").length,
    scrollEvents: signals.scroll.length,
    keyEvents: keyDowns.length,
    averageKeyInterval: keyDowns.length < 2 ? 0 : keyIntervals / (keyDowns.length - 1),
    visibilityChanges: signals.visibilityChanges,
    timeToFirstInteractionMs:
      signals.firstInteractionAt === null ? null : signals.firstInteractionAt - signals.startedAt
  };
}
