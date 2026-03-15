from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parent
RAW_ROOT = ROOT / "raw" / "web_bot_detection_dataset"
OUTPUT_PATH = ROOT / "data" / "sessions.csv"

POINT_PATTERN = re.compile(r"\[(?:m)?\((\d+),(\d+)\)\]")
CLICK_PATTERN = re.compile(r"\[c\([^)]+\)\]")


# ── CSV column order (must stay in sync with FeatureResponse / scoring.py) ───

COLUMNS = [
    "sessionId",
    # mouse
    "sampleCount",
    "mouseSpeedMean",
    "mouseSpeedStd",
    "mouseLinearity",
    "mouseAccelerationMean",
    "mouseJerkMean",
    # clicks
    "clickCount",
    # scroll  (BORDaR has no scroll data → always 0)
    "scrollEvents",
    "scrollVelocityMean",
    "scrollVelocityStd",
    "scrollDirectionChanges",
    # keyboard  (BORDaR has no keyboard data → always 0)
    "keyEvents",
    "averageKeyInterval",
    "keyIntervalStd",
    "keyHoldDurationMean",
    # visibility  (BORDaR has no tab-switch data → always 0)
    "visibilityChanges",
    "visibilityChangeRate",
    "meanTimeBetweenVisibilityChanges",
    # browser entropy  (BORDaR has no browser data → conservative defaults)
    "hasWebGL",
    "screenViewportRatio",
    "devicePixelRatio",
    "touchPoints",
    "isSuspiciousUA",
    # timing  (BORDaR has no session timing → 0)
    "timeToFirstInteractionMs",
    # label
    "label",
]


# ── Math helpers ─────────────────────────────────────────────────────────────


def _distance(x1: int, y1: int, x2: int, y2: int) -> float:
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def _mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def _std(values: list[float]) -> float:
    return stdev(values) if len(values) >= 2 else 0.0


# ── Data classes ─────────────────────────────────────────────────────────────


@dataclass
class SessionFeatures:
    session_id: str
    # mouse
    sample_count: int
    mouse_speed_mean: float
    mouse_speed_std: float
    mouse_linearity: float
    mouse_acceleration_mean: float
    mouse_jerk_mean: float
    # clicks
    click_count: int
    # scroll (zero-filled for BORDaR)
    scroll_events: int = 0
    scroll_velocity_mean: float = 0.0
    scroll_velocity_std: float = 0.0
    scroll_direction_changes: int = 0
    # keyboard (zero-filled for BORDaR)
    key_events: int = 0
    average_key_interval: float = 0.0
    key_interval_std: float = 0.0
    key_hold_duration_mean: float = 0.0
    # visibility (zero-filled for BORDaR)
    visibility_changes: int = 0
    visibility_change_rate: float = 0.0
    mean_time_between_visibility_changes: float = 0.0
    # browser entropy (conservative neutral defaults for BORDaR)
    has_webgl: int = 1  # assume normal browser; model won't penalise
    screen_viewport_ratio: float = 0.85  # typical desktop value
    device_pixel_ratio: float = 1.0
    touch_points: int = 0
    is_suspicious_ua: int = 0
    # timing (zero-filled for BORDaR)
    time_to_first_interaction_ms: int = 0
    # target
    label: str = "human"

    def as_row(self) -> dict[str, str | int | float]:
        return {
            "sessionId": self.session_id,
            "sampleCount": self.sample_count,
            "mouseSpeedMean": round(self.mouse_speed_mean, 6),
            "mouseSpeedStd": round(self.mouse_speed_std, 6),
            "mouseLinearity": round(self.mouse_linearity, 6),
            "mouseAccelerationMean": round(self.mouse_acceleration_mean, 6),
            "mouseJerkMean": round(self.mouse_jerk_mean, 6),
            "clickCount": self.click_count,
            "scrollEvents": self.scroll_events,
            "scrollVelocityMean": round(self.scroll_velocity_mean, 6),
            "scrollVelocityStd": round(self.scroll_velocity_std, 6),
            "scrollDirectionChanges": self.scroll_direction_changes,
            "keyEvents": self.key_events,
            "averageKeyInterval": round(self.average_key_interval, 6),
            "keyIntervalStd": round(self.key_interval_std, 6),
            "keyHoldDurationMean": round(self.key_hold_duration_mean, 6),
            "visibilityChanges": self.visibility_changes,
            "visibilityChangeRate": round(self.visibility_change_rate, 6),
            "meanTimeBetweenVisibilityChanges": round(
                self.mean_time_between_visibility_changes, 6
            ),
            "hasWebGL": self.has_webgl,
            "screenViewportRatio": round(self.screen_viewport_ratio, 6),
            "devicePixelRatio": round(self.device_pixel_ratio, 6),
            "touchPoints": self.touch_points,
            "isSuspiciousUA": self.is_suspicious_ua,
            "timeToFirstInteractionMs": self.time_to_first_interaction_ms,
            "label": self.label,
        }


# ── Feature computation ───────────────────────────────────────────────────────


def parse_points(behaviour: str) -> list[tuple[int, int]]:
    return [(int(x), int(y)) for x, y in POINT_PATTERN.findall(behaviour)]


def parse_times(times_blob: str | None, expected_points: int) -> list[int]:
    """
    Parse the comma-separated timestamp string from BORDaR payloads.
    Falls back to a synthetic 1-ms-per-point sequence when data is absent
    or shorter than the point list, so every session has usable timing.
    """
    if not times_blob:
        return list(range(expected_points))

    values = [int(v) for v in times_blob.split(",") if v.strip()]
    if len(values) >= expected_points:
        return values[:expected_points]

    if not values:
        return list(range(expected_points))

    # Pad by incrementing the last timestamp by 1 ms per missing entry
    last = values[-1]
    while len(values) < expected_points:
        last += 1
        values.append(last)
    return values


def compute_features(
    session_id: str,
    behaviour: str,
    times_blob: str | None,
    label: str,
) -> SessionFeatures:
    points = parse_points(behaviour)
    times = parse_times(times_blob, len(points))
    click_count = len(CLICK_PATTERN.findall(behaviour))

    # ── Speed series ─────────────────────────────────────────────────────────
    speed_series: list[float] = []
    path_distance = 0.0
    direct_distance = 0.0

    for i in range(1, len(points)):
        prev_x, prev_y = points[i - 1]
        curr_x, curr_y = points[i]
        dt = max(times[i] - times[i - 1], 1)  # avoid division by zero
        dist = _distance(prev_x, prev_y, curr_x, curr_y)
        speed_series.append(dist / dt)
        path_distance += dist

    if len(points) > 1:
        direct_distance = _distance(
            points[0][0], points[0][1], points[-1][0], points[-1][1]
        )

    # ── Acceleration series: |Δspeed| ────────────────────────────────────────
    # Bots driven by linear interpolation produce near-zero acceleration.
    accelerations: list[float] = [
        abs(speed_series[i] - speed_series[i - 1]) for i in range(1, len(speed_series))
    ]

    # ── Jerk series: |Δacceleration| ─────────────────────────────────────────
    # Real human paths have irregular jerk; scripted paths approach zero.
    jerks: list[float] = [
        abs(accelerations[i] - accelerations[i - 1])
        for i in range(1, len(accelerations))
    ]

    return SessionFeatures(
        session_id=session_id,
        sample_count=len(points),
        mouse_speed_mean=_mean(speed_series),
        mouse_speed_std=_std(speed_series),
        mouse_linearity=(
            0.0 if path_distance == 0 else direct_distance / path_distance
        ),
        mouse_acceleration_mean=_mean(accelerations),
        mouse_jerk_mean=_mean(jerks),
        click_count=click_count,
        label="human" if label == "human" else "bot",
    )


# ── Annotation loaders ────────────────────────────────────────────────────────


def load_phase1_annotations(annotation_path: Path) -> dict[str, str]:
    labels: dict[str, str] = {}
    for split_name in ("train", "test"):
        split_path = annotation_path / split_name
        for line in split_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            session_id, label = line.split()
            labels[session_id] = label
    return labels


def load_phase2_annotations(annotation_path: Path) -> dict[str, str]:
    labels: dict[str, str] = {}
    for line in annotation_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        session_id, label = line.split()
        # Phase-2 annotation IDs have a trailing _N suffix; strip it so they
        # match the session_id field in the mouse-movement JSONL files.
        base_session_id = session_id.rsplit("_", 1)[0]
        labels[base_session_id] = label
    return labels


# ── Dataset importers ─────────────────────────────────────────────────────────


def import_phase1_variant(variant_name: str) -> list[SessionFeatures]:
    labels = load_phase1_annotations(RAW_ROOT / "phase1" / "annotations" / variant_name)
    base_dir = RAW_ROOT / "phase1" / "data" / "mouse_movements" / variant_name
    rows: list[SessionFeatures] = []

    for session_id, label in labels.items():
        payload_path = base_dir / session_id / "mouse_movements.json"
        if not payload_path.exists():
            continue
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        behaviour = payload.get("total_behaviour") or payload.get(
            "mousemove_total_behaviour", ""
        )
        rows.append(
            compute_features(
                session_id=session_id,
                behaviour=behaviour,
                times_blob=payload.get("mousemove_times"),
                label=label,
            )
        )

    return rows


def import_phase2_file(
    data_path: Path, labels: dict[str, str]
) -> list[SessionFeatures]:
    rows: list[SessionFeatures] = []

    with data_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            session_id = payload["session_id"]
            label = labels.get(session_id)
            if not label:
                continue
            rows.append(
                compute_features(
                    session_id=session_id,
                    behaviour=payload.get("mousemove_total_behaviour", ""),
                    times_blob=payload.get("mousemove_times"),
                    label=label,
                )
            )

    return rows


# ── CSV writer ────────────────────────────────────────────────────────────────


def write_output(rows: list[SessionFeatures]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_row())


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    rows: list[SessionFeatures] = []

    # Phase 1 — per-session JSON files
    rows.extend(import_phase1_variant("humans_and_moderate_bots"))
    rows.extend(import_phase1_variant("humans_and_advanced_bots"))

    # Phase 2 — JSONL files
    phase2_labels = load_phase2_annotations(
        RAW_ROOT
        / "phase2"
        / "annotations"
        / "humans_and_moderate_and_advanced_bots"
        / "humans_and_moderate_and_advanced_bots"
    )
    rows.extend(
        import_phase2_file(
            RAW_ROOT
            / "phase2"
            / "data"
            / "mouse_movements"
            / "humans"
            / "mouse_movements_humans.json",
            phase2_labels,
        )
    )
    rows.extend(
        import_phase2_file(
            RAW_ROOT
            / "phase2"
            / "data"
            / "mouse_movements"
            / "bots"
            / "mouse_movements_moderate_bots.json",
            phase2_labels,
        )
    )
    rows.extend(
        import_phase2_file(
            RAW_ROOT
            / "phase2"
            / "data"
            / "mouse_movements"
            / "bots"
            / "mouse_movements_advanced_bots.json",
            phase2_labels,
        )
    )

    # Deduplicate by session_id (keep last occurrence)
    deduped = list({row.session_id: row for row in rows}.values())
    write_output(deduped)

    human_count = sum(1 for r in deduped if r.label == "human")
    bot_count = sum(1 for r in deduped if r.label == "bot")
    print(
        f"Wrote {len(deduped)} rows to {OUTPUT_PATH} "
        f"({human_count} human / {bot_count} bot).\n"
        f"Columns: {', '.join(COLUMNS)}"
    )


if __name__ == "__main__":
    main()
