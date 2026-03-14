from typing import Literal

from pydantic import BaseModel, Field


class PointerSample(BaseModel):
    x: float
    y: float
    t: int
    type: Literal["move", "click", "down", "up", "enter"]


class KeyboardSample(BaseModel):
    key: str
    t: int
    type: Literal["down", "up"]
    hold: int | None = None  # ms between keydown and keyup (present on "up" events)


class ScrollSample(BaseModel):
    x: float
    y: float
    t: int


class ScreenInfo(BaseModel):
    width: int
    height: int
    devicePixelRatio: float


class ViewportInfo(BaseModel):
    width: int
    height: int


class InferenceRequest(BaseModel):
    sessionId: str
    startedAt: int
    pointer: list[PointerSample]
    keyboard: list[KeyboardSample]
    scroll: list[ScrollSample]
    visibilityChanges: int
    visibilityChangeTimes: list[int] = Field(
        default_factory=list
    )  # timestamp of every visibility change event
    firstInteractionAt: int | None
    screen: ScreenInfo
    viewport: ViewportInfo
    platform: str
    userAgent: str
    language: str
    touchPoints: int
    webglRenderer: str | None
    canvasHash: str


class FeatureResponse(BaseModel):
    sessionId: str
    sampleCount: int

    # ── Mouse movement ────────────────────────────────────────────────────────
    mouseSpeedMean: float
    mouseSpeedStd: float  # low std = suspiciously constant speed (bot)
    mouseLinearity: float  # near 1.0 = perfectly straight path (bot)
    mouseAccelerationMean: float  # mean |Δspeed| – bots show near-zero
    mouseJerkMean: float  # mean |Δacceleration| – bots show near-zero

    # ── Clicks ────────────────────────────────────────────────────────────────
    clickCount: int

    # ── Scroll ────────────────────────────────────────────────────────────────
    scrollEvents: int
    scrollVelocityMean: float  # mean |Δy / Δt|
    scrollVelocityStd: float  # near-zero std = robotic constant scroll
    scrollDirectionChanges: int  # humans reverse scroll; bots rarely do

    # ── Keyboard ─────────────────────────────────────────────────────────────
    keyEvents: int
    averageKeyInterval: float  # ms between successive keydown events
    keyIntervalStd: float  # near-zero = robotic typing cadence
    keyHoldDurationMean: float  # mean ms a key is physically held down

    # ── Tab / window visibility ───────────────────────────────────────────────
    visibilityChanges: int
    visibilityChangeRate: float  # changes per minute
    meanTimeBetweenVisibilityChanges: float  # ms – very low = rapid bot switching

    # ── Browser entropy ───────────────────────────────────────────────────────
    hasWebGL: int  # 0 = no WebGL (common in headless browsers)
    screenViewportRatio: float  # viewport.width / screen.width (headless = 1.0 exactly)
    devicePixelRatio: float  # 1.0 exact on headless; varies on real devices
    touchPoints: int  # 0 = no touch (expected for desktop)
    isSuspiciousUA: int  # 1 if UA contains headless/automation markers

    # ── Timing ───────────────────────────────────────────────────────────────
    timeToFirstInteractionMs: int | None  # very low (<300 ms) is suspicious


class ModelScores(BaseModel):
    """Per-model probability scores and the final weighted ensemble score."""

    randomForest: float | None = None  # P(human) from RandomForest classifier
    xgboost: float | None = None  # P(human) from calibrated XGBoost
    lstm: float | None = None  # P(human) from mouse-sequence LSTM
    ensemble: float = 0.5  # weighted average of all available models


class InferenceResponse(BaseModel):
    score: float
    verdict: Literal["human", "bot", "review"]
    flaggedSignals: list[str]
    features: FeatureResponse

    # ── Phase 2 additions ─────────────────────────────────────────────────────
    modelScores: ModelScores = Field(
        default_factory=lambda: ModelScores(ensemble=0.5),
        description="Individual model probabilities and the ensemble score.",
    )
    shapValues: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "SHAP feature contributions for the best available tabular model. "
            "Positive values push toward 'human', negative toward 'bot'."
        ),
    )
