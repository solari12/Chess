"""Generate and validate a dedicated low-clock teacher dataset for Phase 2E."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Sequence

import chess

from app.learning.time_management.dataset import build_dataset, validate_record, write_dataset
from app.learning.time_management.features import extract_position_features
from app.learning.time_management.models import DatasetSummary
from app.learning.time_management.teacher import (
    TeacherConfig,
    label_dataset,
    summarize_labels,
)


TIME_PRESSURE_CLOCKS_MS = (30_000, 10_000, 5_000, 3_000, 2_000, 1_500, 1_000, 750, 500, 300)
DEFAULT_OUTPUT_DIR = Path("data/time_management")


def enrich_with_clock_diagnostics(
    raw_records: Sequence[dict[str, Any]],
    config: TeacherConfig = TeacherConfig(),
) -> list[dict[str, Any]]:
    """Label copies of raw records and attach per-record cap diagnostics."""
    labeled = label_dataset(raw_records, config)
    for record in labeled:
        teacher = record["teacher"]
        natural = teacher["natural_teacher_time_ms"]
        usable = teacher["usable_clock_ms"]
        cap_amount = max(float(natural) - float(usable), 0.0) if natural is not None else 0.0
        record["teacher_diagnostics"] = {
            "remaining_time_ms": record["clock"]["remaining_time_ms"],
            "usable_time_ms": usable,
            "natural_teacher_time_ms": natural,
            "teacher_time_ms": teacher["teacher_time_ms"],
            "clock_cap_activated": natural is not None and cap_amount > 0,
            "cap_amount_ms": cap_amount,
            "teacher_depth": teacher["teacher_depth"],
            "natural_teacher_depth": teacher["natural_teacher_depth"],
            "label_status": teacher["label_status"],
            "confidence": teacher["teacher_confidence"],
            "information_gain": teacher["information_gain_score"],
        }
    return labeled


def build_clock_report(
    raw_records: Sequence[dict[str, Any]],
    labeled_records: Sequence[dict[str, Any]],
    *,
    config: TeacherConfig = TeacherConfig(),
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Summarize clock caps, same-FEN monotonicity, examples, and search health."""
    if len(raw_records) != len(labeled_records):
        raise ValueError("Raw and labeled Phase 2E datasets must have matching lengths")
    raw_by_id = {record["sample_id"]: record for record in raw_records}
    labeled_by_id = {record["sample_id"]: record for record in labeled_records}
    if raw_by_id.keys() != labeled_by_id.keys():
        raise ValueError("Raw and labeled Phase 2E datasets must have matching sample IDs")
    for sample_id, raw in raw_by_id.items():
        labeled = labeled_by_id[sample_id]
        preserved = {key: value for key, value in labeled.items() if key not in {"teacher", "teacher_diagnostics"}}
        if preserved != raw:
            raise ValueError(f"Labeled Phase 2E record {sample_id} changed its raw fields")

    clock_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    fen_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in labeled_records:
        clock = str(record["clock"]["remaining_time_ms"])
        clock_groups[clock].append(record)
        fen_groups[record["fen"]].append(record)

    per_clock: dict[str, Any] = {}
    for clock in TIME_PRESSURE_CLOCKS_MS:
        group = clock_groups[str(clock)]
        diagnostics = [record["teacher_diagnostics"] for record in group]
        targets = [float(item["teacher_time_ms"]) for item in diagnostics if item["teacher_time_ms"] is not None]
        usable = [float(item["usable_time_ms"]) for item in diagnostics]
        activated = [item for item in diagnostics if item["clock_cap_activated"]]
        per_clock[str(clock)] = {
            "sample_count": len(group),
            "labeled_count": sum(item["label_status"] == "labeled" for item in diagnostics),
            "insufficient_count": sum(item["label_status"] != "labeled" for item in diagnostics),
            "cap_activation_count": len(activated),
            "cap_activation_rate": len(activated) / len(group) if group else 0.0,
            "final_target_ms": _distribution(targets, (0.9,)),
            "usable_clock_ms": _distribution(usable, ()),
        }

    comparable_groups = 0
    monotonic_groups = 0
    monotonic_examples: list[dict[str, Any]] = []
    changed_examples: list[dict[str, Any]] = []
    unchanged_examples: list[dict[str, Any]] = []
    low_clock_examples: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []
    for fen, group in fen_groups.items():
        ordered = sorted(group, key=lambda item: int(item["clock"]["remaining_time_ms"]))
        labeled_points = [
            {
                "remaining_time_ms": int(item["clock"]["remaining_time_ms"]),
                "teacher_time_ms": item["teacher_diagnostics"]["teacher_time_ms"],
                "natural_teacher_time_ms": item["teacher_diagnostics"]["natural_teacher_time_ms"],
                "label_status": item["teacher_diagnostics"]["label_status"],
            }
            for item in ordered
            if item["teacher_diagnostics"]["teacher_time_ms"] is not None
        ]
        if len(labeled_points) >= 2:
            comparable_groups += 1
            is_monotonic = all(
                lower_clock["teacher_time_ms"] <= higher_clock["teacher_time_ms"] + 1e-9
                for lower_clock, higher_clock in zip(labeled_points, labeled_points[1:])
            )
            if is_monotonic:
                monotonic_groups += 1
                if len(monotonic_examples) < 3:
                    monotonic_examples.append(_example(fen, labeled_points))
            elif len(violations) < 10:
                violations.append(_example(fen, labeled_points))
            targets = [point["teacher_time_ms"] for point in labeled_points]
            example = _example(fen, labeled_points)
            if max(targets) - min(targets) > 1e-9 and len(changed_examples) < 3:
                changed_examples.append(example)
            elif max(targets) - min(targets) <= 1e-9 and len(unchanged_examples) < 3:
                unchanged_examples.append(example)
        low_points = [
            {
                "remaining_time_ms": int(item["clock"]["remaining_time_ms"]),
                "usable_time_ms": item["teacher_diagnostics"]["usable_time_ms"],
                "natural_teacher_time_ms": item["teacher_diagnostics"]["natural_teacher_time_ms"],
                "teacher_time_ms": item["teacher_diagnostics"]["teacher_time_ms"],
                "label_status": item["teacher_diagnostics"]["label_status"],
                "label_reason": item["teacher"].get("label_reason"),
            }
            for item in ordered
            if item["teacher_diagnostics"]["label_status"] != "labeled"
            and int(item["clock"]["remaining_time_ms"]) <= config.clock_safety_margin_ms
        ]
        if low_points and len(low_clock_examples) < 3:
            low_clock_examples.append(_example(fen, low_points))

    diagnostics_all = [record["teacher_diagnostics"] for record in labeled_records]
    natural_targets = [float(item["natural_teacher_time_ms"]) for item in diagnostics_all if item["natural_teacher_time_ms"] is not None]
    final_targets = [float(item["teacher_time_ms"]) for item in diagnostics_all if item["teacher_time_ms"] is not None]
    capped_records = [item for item in diagnostics_all if item["clock_cap_activated"]]
    reductions = [float(item["cap_amount_ms"]) for item in capped_records]

    completed_depth = [int(record["search"]["completed_depth"]) for record in raw_records]
    unique_probe_records = [group[0] for group in fen_groups.values()]
    unique_probe_depths = [int(record["search"]["completed_depth"]) for record in unique_probe_records]
    search_depth_counts = _count(completed_depth)
    unique_probe_depth_counts = _count(unique_probe_depths)
    teacher_depth_counts = _count(
        int(item["natural_teacher_depth"])
        for item in diagnostics_all
        if item["natural_teacher_depth"] is not None
    )
    labeled_teacher_depth_counts = _count(
        int(item["teacher_depth"])
        for item in diagnostics_all
        if item["teacher_depth"] is not None
    )
    confidence = [float(item["confidence"]) for item in diagnostics_all]
    information_gain = [float(item["information_gain"]) for item in diagnostics_all]
    timeout_count = sum(bool(record["search"]["timed_out"]) for record in raw_records)
    unique_probe_timeout_count = sum(bool(record["search"]["timed_out"]) for record in unique_probe_records)
    insufficient_count = sum(item["label_status"] != "labeled" for item in diagnostics_all)

    edge_cases = _safety_margin_edge_cases(config)
    report = {
        "phase": "2E",
        "target": "teacher_time_ms",
        "dataset": {
            "raw_records": len(raw_records),
            "labeled_records": len(labeled_records),
            "unique_fen_groups": len(fen_groups),
            "clock_values_ms": list(TIME_PRESSURE_CLOCKS_MS),
            "same_fen_clock_variants": _same_fen_variant_summary(fen_groups),
            "safety_margin_ms": config.clock_safety_margin_ms,
        },
        "clock_cap_by_group": per_clock,
        "clock_monotonicity": {
            "comparable_fen_groups": comparable_groups,
            "monotonic_groups": monotonic_groups,
            "violations": comparable_groups - monotonic_groups,
            "monotonicity_rate": monotonic_groups / comparable_groups if comparable_groups else None,
            "comparison_rule": "For labeled records of a FEN, target must be non-decreasing as remaining clock increases.",
            "violation_examples": violations,
        },
        "natural_vs_capped": {
            "natural_target_distribution_ms": _distribution(natural_targets, (0.9,)),
            "final_labeled_target_distribution_ms": _distribution(final_targets, (0.9,)),
            "natural_at_or_below_usable_count": sum(
                item["natural_teacher_time_ms"] is not None
                and item["natural_teacher_time_ms"] <= item["usable_time_ms"]
                for item in diagnostics_all
            ),
            "natural_above_usable_count": len(capped_records),
            "cap_reduction_ms": {
                "average": statistics.fmean(reductions) if reductions else 0.0,
                "maximum": max(reductions, default=0.0),
            },
            "clock_cap_activated_anywhere": bool(capped_records),
        },
        "same_position_examples": {
            "pressure_changed_target": changed_examples,
            "pressure_did_not_change_target": unchanged_examples,
            "too_low_for_meaningful_search": low_clock_examples,
        },
        "safety_margin_edge_cases": edge_cases,
        "search_and_label_health": {
            "completed_depth_distribution": search_depth_counts,
            "unique_probe_count": len(unique_probe_records),
            "unique_probe_completed_depth_distribution": unique_probe_depth_counts,
            "timeout_count": timeout_count,
            "timeout_rate": timeout_count / len(raw_records) if raw_records else 0.0,
            "unique_probe_timeout_count": unique_probe_timeout_count,
            "unique_probe_timeout_rate": unique_probe_timeout_count / len(unique_probe_records) if unique_probe_records else 0.0,
            "labeled_count": len(raw_records) - insufficient_count,
            "insufficient_count": insufficient_count,
            "insufficient_rate": insufficient_count / len(raw_records) if raw_records else 0.0,
            "teacher_depth_distribution": labeled_teacher_depth_counts,
            "natural_teacher_depth_distribution": teacher_depth_counts,
            "confidence_distribution": _distribution(confidence, (0.9,)),
            "information_gain_distribution": _distribution(information_gain, (0.9,)),
        },
        "interpretation_limits": [
            "Clock values are synthetic; one search probe per FEN is reused across clock variants to isolate teacher cap effects from wall-clock timing noise.",
            "Natural targets expose the teacher-selected depth's observed profile time before the existing safety-margin cap.",
            "The safety cap can force a null label when usable time is zero; this is not reported as a negative target.",
            "The teacher remains a deterministic heuristic labeler, not an optimal time policy.",
        ],
    }
    examples = {
        "phase": "2E",
        "top_clock_pressure_changes": changed_examples,
        "top_clock_pressure_no_change": unchanged_examples,
        "positions_too_low_for_search": low_clock_examples,
        "monotonic_violations": violations,
    }
    return report, examples


def generate_phase2e_artifacts(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    games: int = 2,
    sampling_interval: int = 4,
    max_plies: int = 16,
    analysis_budget_ms: int = 1_000,
    max_analysis_depth: int = 6,
    seed: int = 42,
) -> dict[str, str]:
    """Run existing Phase 2A generation and write isolated Phase 2E artifacts."""
    output = Path(output_dir)
    raw_path = output / "time_pressure_dataset.jsonl"
    metadata_path = output / "time_pressure_dataset.metadata.json"
    training_path = output / "time_pressure_training_dataset.jsonl"
    summary_path = output / "time_pressure_teacher_summary.json"
    report_path = output / "phase2e_clock_report.json"
    examples_path = output / "phase2e_clock_examples.json"
    destinations = (raw_path, metadata_path, training_path, summary_path, report_path, examples_path)
    existing = [str(path) for path in destinations if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite Phase 2E artifacts: {', '.join(existing)}")

    # Search is independent of the synthetic clock in Phase 2A. Generate one
    # observation per FEN, then replicate its telemetry across clock variants;
    # this isolates teacher clock-cap behavior from wall-clock measurement noise.
    base_records, metadata, base_summary = build_dataset(
        games=games,
        sampling_interval=sampling_interval,
        game_generation_depth=2,
        max_plies=max_plies,
        analysis_budget_ms=analysis_budget_ms,
        max_analysis_depth=max_analysis_depth,
        remaining_times_ms=(TIME_PRESSURE_CLOCKS_MS[0],),
        seed=seed,
        candidate_count=3,
        evaluation_profile="default",
    )
    raw_records: list[dict[str, Any]] = []
    for base_record in base_records:
        position_id = base_record["sample_id"].split("_T", maxsplit=1)[0]
        board = chess.Board(base_record["fen"])
        for clock_index, clock_ms in enumerate(TIME_PRESSURE_CLOCKS_MS, start=1):
            record = json.loads(json.dumps(base_record, allow_nan=False))
            record["sample_id"] = f"{position_id}_T{clock_index:03d}"
            record["clock"]["remaining_time_ms"] = clock_ms
            record["position_features"] = extract_position_features(board, clock_ms)
            validate_record(record)
            raw_records.append(record)
    generation_summary = DatasetSummary(
        games=base_summary.games,
        sampled_positions=base_summary.sampled_positions,
        clock_variants=len(TIME_PRESSURE_CLOCKS_MS),
        dataset_records=len(raw_records),
        average_legal_moves=base_summary.average_legal_moves,
        average_completed_depth=base_summary.average_completed_depth,
        average_search_time_ms=base_summary.average_search_time_ms,
        timeout_rate=base_summary.timeout_rate,
    )
    metadata["remaining_times_ms"] = list(TIME_PRESSURE_CLOCKS_MS)
    metadata["telemetry_sampling"] = "one Phase 2A probe per FEN, replicated across synthetic clock variants"
    metadata["phase"] = "2E"
    metadata["purpose"] = "synthetic low-clock teacher cap validation"
    metadata["dataset_summary"] = {
        "sampled_positions": generation_summary.sampled_positions,
        "clock_variants": generation_summary.clock_variants,
        "dataset_records": generation_summary.dataset_records,
        "average_completed_depth": generation_summary.average_completed_depth,
        "average_search_time_ms": generation_summary.average_search_time_ms,
        "timeout_rate": generation_summary.timeout_rate,
    }
    write_dataset(raw_records, metadata, raw_path)

    config = TeacherConfig()
    labeled_records = enrich_with_clock_diagnostics(raw_records, config)
    output.mkdir(parents=True, exist_ok=True)
    _write_jsonl(training_path, labeled_records)
    teacher_summary = summarize_labels(labeled_records, config)
    teacher_summary["phase"] = "2E"
    teacher_summary["dataset"] = str(training_path)
    summary_path.write_text(json.dumps(teacher_summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    report, examples = build_clock_report(raw_records, labeled_records, config=config)
    report["generation"] = {
        "seed": seed,
        "games": games,
        "sampling_interval": sampling_interval,
        "max_plies": max_plies,
        "analysis_budget_ms": analysis_budget_ms,
        "max_analysis_depth": max_analysis_depth,
        "telemetry_sampling": "one probe per FEN, reused across synthetic clocks",
        "raw_dataset_sha256": _sha256(raw_path),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    examples_path.write_text(json.dumps(examples, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return {
        "raw_dataset": str(raw_path),
        "metadata": str(metadata_path),
        "training_dataset": str(training_path),
        "teacher_summary": str(summary_path),
        "clock_report": str(report_path),
        "clock_examples": str(examples_path),
    }


def _safety_margin_edge_cases(config: TeacherConfig) -> dict[str, Any]:
    cases: dict[str, Any] = {}
    for name, remaining_ms in (
        ("above_safety_margin", config.clock_safety_margin_ms + 200),
        ("exact_safety_margin", config.clock_safety_margin_ms),
        ("below_safety_margin", max(0, config.clock_safety_margin_ms - 1)),
        ("zero_remaining_time", 0),
    ):
        record = {
            "sample_id": name,
            "fen": "diagnostic-only",
            "clock": {"remaining_time_ms": remaining_ms, "clock_source": "synthetic"},
            "position_features": {},
            "search": {"timed_out": False},
            "depth_profiles": [
                {"depth": 1, "move": "e2e4", "score": 0, "time_ms": 10, "nodes": 10},
                {"depth": 2, "move": "e2e4", "score": 0, "time_ms": 1_200, "nodes": 20},
            ],
        }
        labeled = enrich_with_clock_diagnostics([record], config)[0]
        diag = labeled["teacher_diagnostics"]
        cases[name] = {
            "remaining_time_ms": remaining_ms,
            "usable_time_ms": diag["usable_time_ms"],
            "natural_teacher_time_ms": diag["natural_teacher_time_ms"],
            "teacher_time_ms": diag["teacher_time_ms"],
            "label_status": diag["label_status"],
            "clock_cap_activated": diag["clock_cap_activated"],
            "non_negative_final_target": diag["teacher_time_ms"] is None or diag["teacher_time_ms"] >= 0,
        }
    return cases


def _same_fen_variant_summary(groups: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    counts = [len({int(record["clock"]["remaining_time_ms"]) for record in records}) for records in groups.values()]
    return {
        "fen_groups": len(groups),
        "groups_with_multiple_clocks": sum(count >= 2 for count in counts),
        "groups_with_all_clock_variants": sum(count == len(TIME_PRESSURE_CLOCKS_MS) for count in counts),
    }


def _example(fen: str, points: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {"fen": fen, "clock_targets": list(points)}


def _distribution(values: Sequence[float], percentiles: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"count": 0, "min": None, "median": None, "mean": None, **{f"p{int(p * 100)}": None for p in percentiles}, "max": None}
    ordered = sorted(float(value) for value in values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "median": statistics.median(ordered),
        "mean": statistics.fmean(ordered),
        **{f"p{int(p * 100)}": _percentile(ordered, p) for p in percentiles},
        "max": ordered[-1],
    }


def _percentile(sorted_values: Sequence[float], percentile: float) -> float:
    position = (len(sorted_values) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (position - lower)


def _count(values: Sequence[int] | Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: int(item[0])))


def _write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as destination:
        for record in records:
            destination.write(json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":")))
            destination.write("\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate and report Phase 2E low-clock teacher validation data.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--games", type=int, default=2)
    parser.add_argument("--sampling-interval", type=int, default=4)
    parser.add_argument("--max-plies", type=int, default=16)
    parser.add_argument("--analysis-budget-ms", type=int, default=1_000)
    parser.add_argument("--max-analysis-depth", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    paths = generate_phase2e_artifacts(
        output_dir=args.output_dir,
        games=args.games,
        sampling_interval=args.sampling_interval,
        max_plies=args.max_plies,
        analysis_budget_ms=args.analysis_budget_ms,
        max_analysis_depth=args.max_analysis_depth,
        seed=args.seed,
    )
    report = json.loads(Path(paths["clock_report"]).read_text(encoding="utf-8"))
    print(json.dumps({"artifacts": paths, "dataset": report["dataset"], "natural_vs_capped": report["natural_vs_capped"], "clock_monotonicity": report["clock_monotonicity"], "search_and_label_health": report["search_and_label_health"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
