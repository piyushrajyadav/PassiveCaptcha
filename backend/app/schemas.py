from typing import Literal

from pydantic import BaseModel


class PointerSample(BaseModel):
    x: float
    y: float
    t: int
    type: Literal["move", "click", "down", "up", "enter"]


class KeyboardSample(BaseModel):
    key: str
    t: int
    type: Literal["down", "up"]


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
    mouseSpeedMean: float
    mouseLinearity: float
    clickCount: int
    scrollEvents: int
    keyEvents: int
    averageKeyInterval: float
    visibilityChanges: int
    timeToFirstInteractionMs: int | None


class InferenceResponse(BaseModel):
    score: float
    verdict: Literal["human", "bot", "review"]
    flaggedSignals: list[str]
    features: FeatureResponse
