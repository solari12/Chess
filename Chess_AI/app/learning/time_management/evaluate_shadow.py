"""Offline metrics for Phase 2F runtime shadow JSONL telemetry."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
from typing import Any, Sequence


CLOCK_BUCKETS = (
    ("30s_plus", 30_000, None),
    ("10s_to_30s", 10_000, 30_000),
    ("3s_to_10s", 3_000, 10_000),
    ("under_3s", 0, 3_000),
)


def evaluate_shadow_records(
    records: Sequence[dict[str, Any]],
    teacher_records: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    """Report prediction quality, clock caps, and inference latency."""
    valid_predictions = [record for record in records if _finite(record.get("raw_prediction_ms"))]
    paired_actual = [record for record in valid_predictions if _finite(record.get("actual_time_ms"))]
    teacher_by_position = {
        (record.get("fen"), int(record.get("clock", {}).get("remaining_time_ms", -1))): record
        for record in teacher_records
        if record.get("teacher", {}).get("label_status") == "labeled"
        and _finite(record.get("teacher", {}).get("teacher_time_ms"))
    }
    teacher_pairs: list[tuple[dict[str, Any], float]] = []
    for record in valid_predictions:
        key = (record.get("fen"), int(record.get("remaining_time_ms", -1)))
        teacher_record = teacher_by_position.get(key)
        if teacher_record is not None:
            teacher_pairs.append((record, float(teacher_record["teacher"]["teacher_time_ms"])))

    capped_count = sum(bool(record.get("clock_cap_activated")) for record in records)
    report: dict[str, Any] = {
        "phase": "2F",
        "shadow_record_count": len(records),
        "model_versions": sorted({str(record.get("model_version", "unknown")) for record in records}),
        "raw_prediction_ms": _distribution([float(record["raw_prediction_ms"]) for record in valid_predictions]),
        "shadow_budget_ms": _distribution([float(record["shadow_budget_ms"]) for record in valid_predictions if _finite(record.get("shadow_budget_ms"))]),
        "actual_time_ms": _distribution([float(record["actual_time_ms"]) for record in records if _finite(record.get("actual_time_ms"))]),
        "prediction_vs_actual": _comparison(
            [float(record["shadow_budget_ms"]) for record in paired_actual],
            [float(record["actual_time_ms"]) for record in paired_actual],
        ),
        "raw_prediction_vs_actual": _comparison(
            [float(record["raw_prediction_ms"]) for record in paired_actual],
            [float(record["actual_time_ms"]) for record in paired_actual],
        ),
        "prediction_vs_teacher": {
            "matched_record_count": len(teacher_pairs),
            "raw_prediction_mae_ms": _mae(
                [float(record["raw_prediction_ms"]) for record, _ in teacher_pairs],
                [target for _, target in teacher_pairs],
            ),
            "shadow_budget_mae_ms": _mae(
                [float(record["shadow_budget_ms"]) for record, _ in teacher_pairs],
                [target for _, target in teacher_pairs],
            ),
        },
        "clock_cap": {
            "activated_count": capped_count,
            "activation_rate": capped_count / len(records) if records else 0.0,
        },
        "recommendation_by_clock_bucket": _clock_buckets(records),
        "inference_latency_ms": _distribution(
            [float(record["inference_time_ms"]) for record in records if _finite(record.get("inference_time_ms"))]
        ),
    }
    return report


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path} on line {line_number}: {error}") from error
            if not isinstance(value, dict):
                raise ValueError(f"Expected a JSON object in {path} on line {line_number}")
            records.append(value)
    return records


def _clock_buckets(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for name, lower, upper in CLOCK_BUCKETS:
        group = [
            record
            for record in records
            if _finite(record.get("remaining_time_ms"))
            and float(record["remaining_time_ms"]) >= lower
            and (upper is None or float(record["remaining_time_ms"]) < upper)
        ]
        report[name] = {
            "record_count": len(group),
            "shadow_budget_ms": _distribution(
                [float(record["shadow_budget_ms"]) for record in group if _finite(record.get("shadow_budget_ms"))]
            ),
            "actual_time_ms": _distribution(
                [float(record["actual_time_ms"]) for record in group if _finite(record.get("actual_time_ms"))]
            ),
        }
    return report


def _comparison(predictions: Sequence[float], actuals: Sequence[float]) -> dict[str, Any]:
    above = sum(prediction > actual for prediction, actual in zip(predictions, actuals))
    below = sum(prediction < actual for prediction, actual in zip(predictions, actuals))
    count = len(predictions)
    return {
        "record_count": count,
        "mae_ms": _mae(predictions, actuals),
        "prediction_above_actual_percent": above / count * 100 if count else None,
        "prediction_below_actual_percent": below / count * 100 if count else None,
        "prediction_equal_actual_percent": (count - above - below) / count * 100 if count else None,
    }


def _mae(predictions: Sequence[float], actuals: Sequence[float]) -> float | None:
    if not predictions:
        return None
    return statistics.fmean(abs(prediction - actual) for prediction, actual in zip(predictions, actuals))


def _distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "median": None, "mean": None, "p90": None, "max": None}
    ordered = sorted(float(value) for value in values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "median": statistics.median(ordered),
        "mean": statistics.fmean(ordered),
        "p90": _percentile(ordered, 0.9),
        "max": ordered[-1],
    }


def _percentile(sorted_values: Sequence[float], percentile: float) -> float:
    position = (len(sorted_values) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (position - lower)


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate Phase 2F runtime shadow JSONL telemetry.")
    parser.add_argument("--input", default="data/time_management/runtime_shadow.jsonl")
    parser.add_argument("--teacher-dataset", help="Optional labeled JSONL to join by FEN and remaining clock.")
    parser.add_argument("--output", default="data/time_management/runtime_shadow_report.json")
    args = parser.parse_args(argv)
    records = load_jsonl(args.input)
    teacher_records = load_jsonl(args.teacher_dataset) if args.teacher_dataset else []
    report = evaluate_shadow_records(records, teacher_records)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    print(f"Shadow report: {output}")


if __name__ == "__main__":
    main()
