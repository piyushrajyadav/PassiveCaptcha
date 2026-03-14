from math import sqrt

from ..schemas import FeatureResponse, InferenceRequest


def _distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def extract_features(payload: InferenceRequest) -> FeatureResponse:
    moves = [sample for sample in payload.pointer if sample.type == "move"]
    total_speed = 0.0
    path_distance = 0.0
    direct_distance = 0.0

    for index in range(1, len(moves)):
        previous = moves[index - 1]
        current = moves[index]
        delta_t = max(current.t - previous.t, 1)
        delta_distance = _distance(previous.x, previous.y, current.x, current.y)
        total_speed += delta_distance / delta_t
        path_distance += delta_distance

    if len(moves) > 1:
        direct_distance = _distance(
            moves[0].x, moves[0].y, moves[-1].x, moves[-1].y
        )

    key_downs = [sample for sample in payload.keyboard if sample.type == "down"]
    total_key_interval = 0

    for index in range(1, len(key_downs)):
        total_key_interval += key_downs[index].t - key_downs[index - 1].t

    return FeatureResponse(
        sessionId=payload.sessionId,
        sampleCount=len(payload.pointer) + len(payload.keyboard) + len(payload.scroll),
        mouseSpeedMean=0 if len(moves) < 2 else total_speed / (len(moves) - 1),
        mouseLinearity=0 if path_distance == 0 else direct_distance / path_distance,
        clickCount=len([sample for sample in payload.pointer if sample.type == "click"]),
        scrollEvents=len(payload.scroll),
        keyEvents=len(key_downs),
        averageKeyInterval=0
        if len(key_downs) < 2
        else total_key_interval / (len(key_downs) - 1),
        visibilityChanges=payload.visibilityChanges,
        timeToFirstInteractionMs=None
        if payload.firstInteractionAt is None
        else payload.firstInteractionAt - payload.startedAt,
    )
