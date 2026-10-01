"""Objective distribution analysis for a raw and teacher-labeled JSONL pair."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
from typing import Any, Sequence

EXPECTED_CLOCKS_MS = (300_000, 120_000, 30_000, 10_000, 3_000)


def build_validation_report(
    raw_records: Sequence[dict[str, Any]],
    labeled_records: Sequence[dict[str, Any]],
    *,
    min_gain_for_continue: float = 0.12,
) -> dict[str, Any]:
    """Summarize label, depth, time, confidence, stability, and clock groups."""
    if not 0 <= min_gain_for_continue <= 1:
        raise ValueError("min_gain_for_continue must be in the range 0..1")
    raw_by_id = _index_by_sample_id(raw_records)
    labeled_by_id = _index_by_sample_id(labeled_records)
    if raw_by_id.keys() != labeled_by_id.keys():
        raise ValueError("Raw and labeled datasets must contain matching sample_id values")
    for sample_id, raw_record in raw_by_id.items():
        labeled_record = labeled_by_id[sample_id]
        original_fields = {key: value for key, value in labeled_record.items() if key != "teacher"}
        if original_fields != raw_record:
            raise ValueError(f"Labeled record {sample_id} does not preserve its raw fields")

    labeled = [record for record in labeled_records if record.get("teacher", {}).get("label_status") == "labeled"]
    insufficient = [record for record in labeled_records if record.get("teacher", {}).get("label_status") != "labeled"]
    teacher_times = [_finite_number(record["teacher"].get("teacher_time_ms")) for record in labeled]
    teacher_depths = [_finite_number(record["teacher"].get("teacher_depth")) for record in labeled]
    confidences = [_finite_number(record["teacher"].get("teacher_confidence")) for record in labeled]
    information_gains = [
        _finite_number(record["teacher"].get("information_gain_score"))
        for record in labeled_records
    ]
    depth_counts = _count_values(teacher_depths)

    clock_stats: dict[str, dict[str, Any]] = {}
    clock_values = list(EXPECTED_CLOCKS_MS)
    clock_values.extend(
        sorted(
            {
                int(record["clock"]["remaining_time_ms"])
                for record in raw_records
                if int(record["clock"]["remaining_time_ms"]) not in EXPECTED_CLOCKS_MS
            },
            reverse=True,
        )
    )
    for clock_ms in clock_values:
        group = [
            record
            for record in labeled_records
            if int(record["clock"]["remaining_time_ms"]) == clock_ms
        ]
        group_labeled = [record for record in group if record.get("teacher", {}).get("label_status") == "labeled"]
        group_times = [_finite_number(record["teacher"].get("teacher_time_ms")) for record in group_labeled]
        group_depths = [_finite_number(record["teacher"].get("teacher_depth")) for record in group_labeled]
        clock_stats[str(clock_ms)] = {
            "samples": len(group),
            "labeled_count": len(group_labeled),
            "median_teacher_time_ms": _percentile(group_times, 0.5),
            "p90_teacher_time_ms": _percentile(group_times, 0.9),
            "median_teacher_depth": _percentile(group_depths, 0.5),
        }

    time_distribution = _distribution(teacher_times, (0.25, 0.5, 0.75, 0.9, 0.95))
    information_gain_distribution = _distribution(information_gains, (0.5, 0.75, 0.9))
    confidence_distribution = _distribution(confidences, (0.5,))
    transitions = [
        transition
        for record in labeled_records
        for transition in record.get("teacher", {}).get("depth_transitions", [])
    ]
    move_change_counts = [record["teacher"].get("best_move_change_count", 0) for record in labeled_records]
    significant_score_positions = sum(
        any(
            _finite_number(transition.get("evaluation_change_signal")) >= min_gain_for_continue
            for transition in record.get("teacher", {}).get("depth_transitions", [])
        )
        for record in labeled_records
    )
    stable_count = sum(
        bool(record.get("teacher", {}).get("depth_transitions"))
        and _finite_number(record["teacher"].get("search_stability")) >= 1.0 - min_gain_for_continue
        for record in labeled_records
    )
    repeated_move_change_count = sum(count >= 2 for count in move_change_counts)
    clock_medians = [value["median_teacher_time_ms"] for value in clock_stats.values() if value["median_teacher_time_ms"] is not None]
    position_times: dict[str, list[float]] = {}
    for record in labeled:
        position_times.setdefault(record["fen"], []).append(
            _finite_number(record["teacher"].get("teacher_time_ms"))
        )
    position_median_times = [statistics.median(values) for values in position_times.values()]

    return {
        "raw_records": len(raw_records),
        "labeled_records": len(labeled),
        "insufficient_records": len(insufficient),
        "label_rate": len(labeled) / len(raw_records) if raw_records else 0.0,
        "insufficient_rate": len(insufficient) / len(raw_records) if raw_records else 0.0,
        "teacher_depth_distribution": {
            "counts": dict(sorted(depth_counts.items(), key=lambda item: int(item[0]))),
            "depth_2_fraction": depth_counts.get("2", 0) / len(labeled) if labeled else 0.0,
        },
        "teacher_time_ms_distribution": time_distribution,
        "teacher_confidence_distribution": confidence_distribution,
        "information_gain_distribution": information_gain_distribution,
        "search_stability": {
            "stable_positions": stable_count,
            "positions_with_best_move_changes": sum(count > 0 for count in move_change_counts),
            "positions_with_repeated_best_move_changes": repeated_move_change_count,
            "positions_with_significant_score_changes": significant_score_positions,
            "transition_count": len(transitions),
        },
        "clock_distribution": clock_stats,
        "target_variation": {
            "unique_teacher_time_count": len(set(teacher_times)),
            "teacher_time_range_ms": _range(teacher_times),
            "teacher_time_iqr_ms": _iqr(teacher_times),
            "teacher_time_standard_deviation_ms": statistics.pstdev(teacher_times) if len(teacher_times) > 1 else 0.0,
            "positions_with_at_least_one_labeled_clock": len(position_times),
            "unique_per_position_median_target_count": len(set(position_median_times)),
            "per_position_median_teacher_time_ms": _distribution(
                position_median_times, (0.25, 0.5, 0.75, 0.9, 0.95)
            ),
            "per_position_median_target_iqr_ms": _iqr(position_median_times),
            "teacher_depth_unique_count": len(set(teacher_depths)),
            "clock_group_median_target_range_ms": _range(clock_medians),
            "teacher_depth_concentrated_at_2_or_more": (
                depth_counts.get("2", 0) / len(labeled) >= 0.9 if labeled else False
            ),
        },
        "teacher_config_threshold": {"min_gain_for_continue": min_gain_for_continue},
    }


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number} in {path}: {error}") from error
    return records


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validate target distributions from Phase 2A and Phase 2B JSONL files.")
    parser.add_argument("--raw", default="data/time_management/validation_dataset.jsonl")
    parser.add_argument("--labeled", default="data/time_management/validation_training_dataset.jsonl")
    parser.add_argument("--output", default="data/time_management/validation_report.json")
    parser.add_argument("--min-gain", type=float, default=0.12)
    args = parser.parse_args(argv)

    report = build_validation_report(
        load_jsonl(args.raw),
        load_jsonl(args.labeled),
        min_gain_for_continue=args.min_gain,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    print(f"Validation report: {output}")


def _index_by_sample_id(records: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        sample_id = record.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("Every record must have a non-empty sample_id")
        if sample_id in indexed:
            raise ValueError(f"Duplicate sample_id: {sample_id}")
        indexed[sample_id] = record
    return indexed


def _finite_number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Expected a finite numeric value, got {value!r}")
    return float(value)


def _count_values(values: Sequence[float]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(int(value)) if value.is_integer() else str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    if not 0 <= percentile <= 1:
        raise ValueError("percentile must be between 0 and 1")
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    fraction = index - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _distribution(values: Sequence[float], extra_percentiles: Sequence[float]) -> dict[str, float | None]:
    distribution: dict[str, float | None] = {
        "min": min(values) if values else None,
        "median": _percentile(values, 0.5),
        "mean": statistics.fmean(values) if values else None,
        "max": max(values) if values else None,
    }
    for percentile in extra_percentiles:
        distribution[f"p{int(percentile * 100)}"] = _percentile(values, percentile)
    return distribution


def _iqr(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return _percentile(values, 0.75) - _percentile(values, 0.25)


def _range(values: Sequence[float]) -> float | None:
    return max(values) - min(values) if values else None


if __name__ == "__main__":
    main()
