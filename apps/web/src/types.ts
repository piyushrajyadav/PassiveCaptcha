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
  mouseSpeedMean: number;
  mouseLinearity: number;
  clickCount: number;
  scrollEvents: number;
  keyEvents: number;
  averageKeyInterval: number;
  visibilityChanges: number;
  timeToFirstInteractionMs: number | null;
};

export type SessionAssessment = {
  score: number;
  verdict: "human" | "bot" | "review";
  flaggedSignals: string[];
  features: FeatureVector;
};
