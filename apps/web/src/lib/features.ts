import { FeatureVector, PassiveSignals } from "../types";

// ── Math helpers ────────────────────────────────────────────────────────────

function distance(x1: number, y1: number, x2: number, y2: number): number {
  return Math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2);
}

function meanOf(values: number[]): number {
  if (values.length === 0) return 0;
  return values.reduce((a, b) => a + b, 0) / values.length;
}

function stdDevOf(values: number[]): number {
  if (values.length < 2) return 0;
  const avg = meanOf(values);
  const variance = values.reduce((a, b) => a + (b - avg) ** 2, 0) / values.length;
  return Math.sqrt(variance);
}

// ── Browser entropy helpers ─────────────────────────────────────────────────

/**
 * Returns 1 if the user-agent string contains known headless / automation markers.
 * Headless Chrome, Selenium, Puppeteer, Playwright, PhantomJS, jsdom all leave
 * distinctive strings in the UA or in navigator properties.
 */
function detectSuspiciousUA(ua: string): number {
  const patterns = [
    /headless/i,
    /phantomjs/i,
    /selenium/i,
    /webdriver/i,
    /puppeteer/i,
    /playwright/i,
    /jsdom/i,
    /nightmare/i,
    /slimerjs/i,
    /htmlunit/i,
  ];
  return patterns.some((p) => p.test(ua)) ? 1 : 0;
}

// ── Main extractor ──────────────────────────────────────────────────────────

export function extractFeatures(signals: PassiveSignals): FeatureVector {

  // ── 1. Mouse movement ───────────────────────────────────────────────────
  const moves = signals.pointer.filter((s) => s.type === "move");

  const speedSeries: number[] = [];
  let pathDistance = 0;
  let directDistance = 0;

  for (let i = 1; i < moves.length; i++) {
    const prev = moves[i - 1];
    const curr = moves[i];
    const dt = Math.max(curr.t - prev.t, 1);       // avoid division by zero
    const dist = distance(prev.x, prev.y, curr.x, curr.y);
    speedSeries.push(dist / dt);
    pathDistance += dist;
  }

  if (moves.length > 1) {
    directDistance = distance(
      moves[0].x, moves[0].y,
      moves[moves.length - 1].x, moves[moves.length - 1].y
    );
  }

  // Acceleration series: |Δspeed| between consecutive movement samples
  // Bots driven by linear interpolation produce near-zero acceleration.
  const accelerations: number[] = [];
  for (let i = 1; i < speedSeries.length; i++) {
    accelerations.push(Math.abs(speedSeries[i] - speedSeries[i - 1]));
  }

  // Jerk series: |Δacceleration| — the "smoothness" of motion changes
  // Real humans have irregular jerk; scripted paths have near-zero jerk.
  const jerks: number[] = [];
  for (let i = 1; i < accelerations.length; i++) {
    jerks.push(Math.abs(accelerations[i] - accelerations[i - 1]));
  }

  // ── 2. Scroll ──────────────────────────────────────────────────────────
  const scrollVelocities: number[] = [];
  let scrollDirectionChanges = 0;
  let lastScrollDir = 0; // -1 up, 0 no-move, 1 down

  for (let i = 1; i < signals.scroll.length; i++) {
    const prev = signals.scroll[i - 1];
    const curr = signals.scroll[i];
    const dt = Math.max(curr.t - prev.t, 1);
    const dy = curr.y - prev.y;

    scrollVelocities.push(Math.abs(dy) / dt);

    const dir = dy > 0 ? 1 : dy < 0 ? -1 : 0;
    if (dir !== 0 && lastScrollDir !== 0 && dir !== lastScrollDir) {
      scrollDirectionChanges += 1;
    }
    if (dir !== 0) lastScrollDir = dir;
  }

  // ── 3. Keyboard ────────────────────────────────────────────────────────
  const keyDowns = signals.keyboard.filter((s) => s.type === "down");

  // Inter-keystroke intervals (keydown → next keydown)
  const keyIntervals: number[] = [];
  for (let i = 1; i < keyDowns.length; i++) {
    keyIntervals.push(keyDowns[i].t - keyDowns[i - 1].t);
  }

  // Key hold durations (keydown → corresponding keyup), stored on "up" events
  const keyUps = signals.keyboard.filter((s) => s.type === "up");
  const holdDurations: number[] = keyUps
    .filter((s) => s.hold !== undefined)
    .map((s) => s.hold as number);

  // ── 4. Tab / window visibility ─────────────────────────────────────────
  const vTimes = signals.visibilityChangeTimes;
  const sessionDurationMs = Math.max(Date.now() - signals.startedAt, 1000);

  // Rate: changes per minute — very high rate is bot-like
  const visibilityChangeRate = (vTimes.length / sessionDurationMs) * 60_000;

  // Mean gap between successive visibility changes
  // Very short gaps (< 200 ms) suggest scripted tab toggling
  let meanTimeBetweenVisibilityChanges = 0;
  if (vTimes.length > 1) {
    let totalGap = 0;
    for (let i = 1; i < vTimes.length; i++) {
      totalGap += vTimes[i] - vTimes[i - 1];
    }
    meanTimeBetweenVisibilityChanges = totalGap / (vTimes.length - 1);
  }

  // ── 5. Browser entropy ─────────────────────────────────────────────────
  // Headless browsers (Puppeteer, Playwright, etc.) typically:
  //   • have no WebGL renderer
  //   • report viewport === screen size (ratio = 1.0 exactly)
  //   • have devicePixelRatio = 1 exactly
  //   • have 0 touch points
  //   • leave automation strings in the user-agent
  const hasWebGL = signals.webglRenderer !== null ? 1 : 0;

  const screenViewportRatio =
    signals.screen.width > 0
      ? signals.viewport.width / signals.screen.width
      : 0;

  // ── Assemble the feature vector ────────────────────────────────────────
  return {
    sessionId: signals.sessionId,
    sampleCount: signals.pointer.length + signals.keyboard.length + signals.scroll.length,

    // Mouse
    mouseSpeedMean: meanOf(speedSeries),
    mouseSpeedStd: stdDevOf(speedSeries),
    mouseLinearity: pathDistance === 0 ? 0 : directDistance / pathDistance,
    mouseAccelerationMean: meanOf(accelerations),
    mouseJerkMean: meanOf(jerks),

    // Clicks
    clickCount: signals.pointer.filter((s) => s.type === "click").length,

    // Scroll
    scrollEvents: signals.scroll.length,
    scrollVelocityMean: meanOf(scrollVelocities),
    scrollVelocityStd: stdDevOf(scrollVelocities),
    scrollDirectionChanges,

    // Keyboard
    keyEvents: keyDowns.length,
    averageKeyInterval: meanOf(keyIntervals),
    keyIntervalStd: stdDevOf(keyIntervals),
    keyHoldDurationMean: meanOf(holdDurations),

    // Visibility
    visibilityChanges: signals.visibilityChanges,
    visibilityChangeRate,
    meanTimeBetweenVisibilityChanges,

    // Browser entropy
    hasWebGL,
    screenViewportRatio,
    devicePixelRatio: signals.screen.devicePixelRatio,
    touchPoints: signals.touchPoints,
    isSuspiciousUA: detectSuspiciousUA(signals.userAgent),

    // Timing
    timeToFirstInteractionMs:
      signals.firstInteractionAt === null
        ? null
        : signals.firstInteractionAt - signals.startedAt,
  };
}
