from __future__ import annotations

import re
from math import sqrt
from statistics import mean, stdev

from ..schemas import FeatureResponse, InferenceRequest

# ── Math helpers ─────────────────────────────────────────────────────────────


def _distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def _mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def _std(values: list[float]) -> float:
    return stdev(values) if len(values) >= 2 else 0.0


# ── Browser entropy helpers ──────────────────────────────────────────────────

# Known automation / headless substrings found in user-agent strings or
# navigator.platform values reported by Puppeteer, Playwright, Selenium,
# PhantomJS, jsdom, NightmareJS, SlimerJS, and HtmlUnit.
_SUSPICIOUS_UA_PATTERNS = re.compile(
    r"headless|phantomjs|selenium|webdriver|puppeteer|playwright|jsdom|nightmare|slimerjs|htmlunit",
    re.IGNORECASE,
)


def _is_suspicious_ua(user_agent: str) -> int:
    """Return 1 if the UA string contains known automation markers, else 0."""
    return 1 if _SUSPICIOUS_UA_PATTERNS.search(user_agent) else 0


# ── Main extractor ───────────────────────────────────────────────────────────


def extract_features(payload: InferenceRequest) -> FeatureResponse:  # noqa: C901

    # ── 1. Mouse movement ────────────────────────────────────────────────────
    moves = [s for s in payload.pointer if s.type == "move"]

    speed_series: list[float] = []
    path_distance = 0.0
    direct_distance = 0.0

    for i in range(1, len(moves)):
        prev = moves[i - 1]
        curr = moves[i]
        dt = max(curr.t - prev.t, 1)  # avoid division by zero
        dist = _distance(prev.x, prev.y, curr.x, curr.y)
        speed_series.append(dist / dt)
        path_distance += dist

    if len(moves) > 1:
        direct_distance = _distance(moves[0].x, moves[0].y, moves[-1].x, moves[-1].y)

    # Acceleration series: |Δspeed| between consecutive movement samples.
    # Bots driven by linear interpolation produce near-zero acceleration.
    accelerations: list[float] = [
        abs(speed_series[i] - speed_series[i - 1]) for i in range(1, len(speed_series))
    ]

    # Jerk series: |Δacceleration|.
    # Real human paths have irregular jerk; scripted paths approach zero.
    jerks: list[float] = [
        abs(accelerations[i] - accelerations[i - 1])
        for i in range(1, len(accelerations))
    ]

    # ── 2. Scroll ────────────────────────────────────────────────────────────
    scroll_velocities: list[float] = []
    scroll_direction_changes = 0
    last_scroll_dir = 0  # -1 up, 0 none, 1 down

    for i in range(1, len(payload.scroll)):
        prev_s = payload.scroll[i - 1]
        curr_s = payload.scroll[i]
        dt = max(curr_s.t - prev_s.t, 1)
        dy = curr_s.y - prev_s.y
        scroll_velocities.append(abs(dy) / dt)

        direction = 1 if dy > 0 else (-1 if dy < 0 else 0)
        if direction != 0 and last_scroll_dir != 0 and direction != last_scroll_dir:
            scroll_direction_changes += 1
        if direction != 0:
            last_scroll_dir = direction

    # ── 3. Keyboard ──────────────────────────────────────────────────────────
    key_downs = [s for s in payload.keyboard if s.type == "down"]

    # Inter-keystroke intervals (keydown → next keydown in ms)
    key_intervals: list[float] = [
        float(key_downs[i].t - key_downs[i - 1].t) for i in range(1, len(key_downs))
    ]

    # Key hold durations (ms key was physically depressed) from "up" events
    hold_durations: list[float] = [
        float(s.hold) for s in payload.keyboard if s.type == "up" and s.hold is not None
    ]

    # ── 4. Tab / window visibility ───────────────────────────────────────────
    v_times = payload.visibilityChangeTimes
    # Use the time between the payload's session start and "now" as session length.
    # We approximate "now" from the latest event timestamp to avoid server clock skew.
    all_timestamps = (
        [s.t for s in payload.pointer]
        + [s.t for s in payload.keyboard]
        + [s.t for s in payload.scroll]
        + v_times
    )
    latest_t = max(all_timestamps) if all_timestamps else payload.startedAt
    session_duration_ms = max(latest_t - payload.startedAt, 1000)

    # Changes per minute — very high rate is bot-like
    visibility_change_rate = (len(v_times) / session_duration_ms) * 60_000

    # Mean gap between successive visibility-change events
    # Very short gaps (< 200 ms) suggest scripted tab toggling
    mean_time_between_visibility_changes = 0.0
    if len(v_times) > 1:
        gaps = [float(v_times[i] - v_times[i - 1]) for i in range(1, len(v_times))]
        mean_time_between_visibility_changes = _mean(gaps)

    # ── 5. Browser entropy ───────────────────────────────────────────────────
    # Headless browsers (Puppeteer, Playwright, etc.) typically:
    #   • have no WebGL renderer string
    #   • report viewport width == screen width (ratio ≈ 1.0 exactly)
    #   • have devicePixelRatio of exactly 1
    #   • have 0 touch points
    #   • leave automation strings in the user-agent
    has_webgl = 0 if payload.webglRenderer is None else 1

    screen_viewport_ratio = (
        payload.viewport.width / payload.screen.width
        if payload.screen.width > 0
        else 0.0
    )

    # ── Assemble the feature vector ──────────────────────────────────────────
    time_to_first_ms: int | None = (
        None
        if payload.firstInteractionAt is None
        else payload.firstInteractionAt - payload.startedAt
    )

    return FeatureResponse(
        sessionId=payload.sessionId,
        sampleCount=(
            len(payload.pointer) + len(payload.keyboard) + len(payload.scroll)
        ),
        # Mouse
        mouseSpeedMean=_mean(speed_series),
        mouseSpeedStd=_std(speed_series),
        mouseLinearity=(0.0 if path_distance == 0 else direct_distance / path_distance),
        mouseAccelerationMean=_mean(accelerations),
        mouseJerkMean=_mean(jerks),
        # Clicks
        clickCount=len([s for s in payload.pointer if s.type == "click"]),
        # Scroll
        scrollEvents=len(payload.scroll),
        scrollVelocityMean=_mean(scroll_velocities),
        scrollVelocityStd=_std(scroll_velocities),
        scrollDirectionChanges=scroll_direction_changes,
        # Keyboard
        keyEvents=len(key_downs),
        averageKeyInterval=_mean(key_intervals),
        keyIntervalStd=_std(key_intervals),
        keyHoldDurationMean=_mean(hold_durations),
        # Visibility
        visibilityChanges=payload.visibilityChanges,
        visibilityChangeRate=visibility_change_rate,
        meanTimeBetweenVisibilityChanges=mean_time_between_visibility_changes,
        # Browser entropy
        hasWebGL=has_webgl,
        screenViewportRatio=screen_viewport_ratio,
        devicePixelRatio=payload.screen.devicePixelRatio,
        touchPoints=payload.touchPoints,
        isSuspiciousUA=_is_suspicious_ua(payload.userAgent),
        # Timing
        timeToFirstInteractionMs=time_to_first_ms,
    )
