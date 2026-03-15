export type PointerSample = {
  x: number;
  y: number;
  t: number;
  type: "move" | "click" | "down" | "up" | "enter";
};

export type KeyboardSample = {
  key: string;
  t: number;
  type: "down" | "up";
  hold?: number; // ms between keydown and keyup for this key (present on "up" events)
};

export type ScrollSample = {
  x: number;
  y: number;
  t: number;
};

export type PassiveSignals = {
  sessionId: string;
  startedAt: number;
  pointer: PointerSample[];
  keyboard: KeyboardSample[];
  scroll: ScrollSample[];
  visibilityChanges: number;
  visibilityChangeTimes: number[]; // timestamp of every visibility change event
  firstInteractionAt: number | null;
  screen: {
    width: number;
    height: number;
    devicePixelRatio: number;
  };
  viewport: {
    width: number;
    height: number;
  };
  platform: string;
  userAgent: string;
  language: string;
  touchPoints: number;
  webglRenderer: string | null;
  canvasHash: string;
};

export type FeatureVector = {
  sessionId: string;
  sampleCount: number;

  // ── Mouse movement ────────────────────────────────────────────────────────
  mouseSpeedMean: number;
  mouseSpeedStd: number;        // low std = suspiciously constant speed (bot)
  mouseLinearity: number;       // near 1.0 = perfectly straight path (bot)
  mouseAccelerationMean: number; // mean |Δspeed| – bots show near-zero
  mouseJerkMean: number;        // mean |Δacceleration| – bots show near-zero

  // ── Clicks ────────────────────────────────────────────────────────────────
  clickCount: number;

  // ── Scroll ────────────────────────────────────────────────────────────────
  scrollEvents: number;
  scrollVelocityMean: number;   // mean |Δy / Δt|
  scrollVelocityStd: number;    // near-zero std = robotic constant scroll
  scrollDirectionChanges: number; // humans reverse scroll; bots rarely do

  // ── Keyboard ─────────────────────────────────────────────────────────────
  keyEvents: number;
  averageKeyInterval: number;   // ms between successive keydown events
  keyIntervalStd: number;       // near-zero = robotic typing cadence
  keyHoldDurationMean: number;  // mean ms a key is physically held down

  // ── Tab / window visibility ───────────────────────────────────────────────
  visibilityChanges: number;
  visibilityChangeRate: number;           // changes per minute
  meanTimeBetweenVisibilityChanges: number; // ms – very low = rapid bot switching

  // ── Browser entropy ───────────────────────────────────────────────────────
  hasWebGL: number;              // 0 = no WebGL (common in headless browsers)
  screenViewportRatio: number;   // viewport.width / screen.width (headless = 1.0 exactly)
  devicePixelRatio: number;      // 1.0 exact on headless; varies on real devices
  touchPoints: number;           // 0 = no touch support (expected for desktop)
  isSuspiciousUA: number;        // 1 if UA contains headless/automation markers

  // ── Timing ───────────────────────────────────────────────────────────────
  timeToFirstInteractionMs: number | null; // very low (<300 ms) is suspicious
};

export type ModelScores = {
  randomForest: number | null; // P(human) from RandomForest
  xgboost: number | null;      // P(human) from calibrated XGBoost
  lstm: number | null;         // P(human) from mouse-sequence LSTM
  ensemble: number;            // weighted average of available models
};

export type SessionAssessment = {
  score: number;
  verdict: "human" | "bot" | "review";
  flaggedSignals: string[];
  features: FeatureVector;
  modelScores: ModelScores;          // per-model breakdown
  shapValues: Record<string, number>; // feature_name → signed SHAP contribution
};
