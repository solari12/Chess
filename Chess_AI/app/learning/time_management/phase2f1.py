"""Offline shadow collection and pre-search probe study for Phase 2F.1.

This module runs real, bounded searches only from an explicit CLI invocation.
It does not change the production search path or train a model.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
from typing import Any, Sequence

import chess
import numpy as np

from app.engine.iterative_search import IterativeSearchResult, iterative_search
from app.engine.search import search_with_time_budget
from app.learning.time_management.evaluate_shadow import load_jsonl
from app.learning.time_management.model_features import FEATURE_NAMES, extract_model_features
from app.learning.time_management.position_generator import generate_positions
from app.learning.time_management.runtime_shadow import (
    SAFETY_MARGIN_MS,
    build_runtime_record,
    load_shadow_model,
)
from app.schemas.chess import TimedAIRequest


ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = ROOT / "data" / "time_management"
DEFAULT_COLLECTION_PATH = DATA_ROOT / "phase2f1_runtime_shadow.jsonl"
DEFAULT_REPORT_PATH = DATA_ROOT / "phase2f1_shadow_report.json"
CLOCK_BUCKETS_MS = (30_000, 10_000, 5_000, 3_000, 2_000, 1_500, 1_000, 750, 500)
PROBE_BUDGETS_MS = (25, 50, 100, 150, 200)


def select_position_fens(*, seed: int, position_count: int) -> list[str]:
    """Generate varied, reproducible FENs using the existing Phase 2A generator."""
    if position_count < 1:
        raise ValueError("position_count must be positive")
    samples = generate_positions(
        games=max(3, math.ceil(position_count / 8)),
        sampling_interval=2,
        game_generation_depth=2,
        max_plies=30,
        seed=seed,
        candidate_count=3,
    )
    fens = list(dict.fromkeys(sample.fen for sample in samples))
    # Evenly sample across the generated game traces while keeping their order
    # deterministic. This avoids using only the opening positions.
    if len(fens) > position_count:
        indices = [round(i * (len(fens) - 1) / (position_count - 1)) for i in range(position_count)] if position_count > 1 else [0]
        fens = [fens[index] for index in indices]
    return fens


def collection_plan(fens: Sequence[str], clock_buckets_ms: Sequence[int] = CLOCK_BUCKETS_MS) -> list[tuple[str, int]]:
    """Return stable, exhaustive FEN/clock pairs for collection metadata/tests."""
    return [(fen, clock_ms) for clock_ms in clock_buckets_ms for fen in fens]


def collect_shadow_searches(
    *,
    fens: Sequence[str],
    search_budget_ms: int,
    max_depth: int,
    log_path: str | Path,
    clock_buckets_ms: Sequence[int] = CLOCK_BUCKETS_MS,
) -> list[dict[str, Any]]:
    """Run one real timed endpoint search for each position and clock pair."""
    if search_budget_ms < 1 or max_depth < 1:
        raise ValueError("search_budget_ms and max_depth must be positive")
    output = Path(log_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("", encoding="utf-8")
    pairs = collection_plan(fens, clock_buckets_ms)
    for fen, remaining_ms in pairs:
        board = chess.Board(fen)
        if not board.is_valid() or board.is_game_over(claim_draw=True):
            continue
        request = TimedAIRequest(
            fen=fen,
            algorithm="alpha-beta",
            time_budget_ms=min(search_budget_ms, remaining_ms),
            remaining_time_ms=remaining_ms,
            max_depth=max_depth,
        )
        search_with_time_budget(request, shadow_log_path=str(output))
    return load_jsonl(output)


def run_probe_experiment(
    *,
    fens: Sequence[str],
    model: Any | None = None,
    model_version: str | None = None,
    clock_buckets_ms: Sequence[int] = CLOCK_BUCKETS_MS,
    probe_budgets_ms: Sequence[int] = PROBE_BUDGETS_MS,
    max_depth: int = 64,
) -> list[dict[str, Any]]:
    """Run offline probes, then infer with the unchanged Phase 2C feature schema."""
    if model is None:
        model, loaded_version = load_shadow_model()
        model_version = model_version or loaded_version
    records: list[dict[str, Any]] = []
    for remaining_ms in clock_buckets_ms:
        for fen in fens:
            for probe_budget in probe_budgets_ms:
                board = chess.Board(fen)
                before_fen = board.fen()
                result = iterative_search(board, time_budget_ms=probe_budget, max_depth=max_depth)
                if board.fen() != before_fen:
                    raise RuntimeError("Offline probe unexpectedly mutated its input board")
                runtime_record = build_runtime_record(
                    board,
                    remaining_time_ms=remaining_ms,
                    result=result,
                    actual_search_budget_ms=probe_budget,
                    max_depth=max_depth,
                )
                features = extract_model_features(runtime_record)
                prediction = np.asarray(model.predict(features.reshape(1, -1)), dtype=np.float64)
                if prediction.shape != (1,) or not np.isfinite(prediction).all():
                    raise ValueError("The Phase 2C model returned an invalid probe prediction")
                elapsed_ms = max(0.0, float(result.time_ms))
                remaining_after_probe = max(float(remaining_ms) - elapsed_ms, 0.0)
                usable_before_probe = max(float(remaining_ms) - SAFETY_MARGIN_MS, 0.0)
                usable_after_probe = max(remaining_after_probe - SAFETY_MARGIN_MS, 0.0)
                raw_prediction_ms = max(float(prediction[0]), 0.0)
                capped_before_probe_ms = min(raw_prediction_ms, usable_before_probe)
                capped_after_probe_ms = min(raw_prediction_ms, usable_after_probe)
                estimated_full_search_budget_ms = min(
                    max(capped_before_probe_ms - elapsed_ms, 0.0), usable_after_probe
                )
                records.append({
                    "fen": fen,
                    "remaining_time_before_probe_ms": remaining_ms,
                    "probe_budget_ms": probe_budget,
                    "probe_elapsed_ms": elapsed_ms,
                    "probe_completed_depth": int(result.completed_depth),
                    "probe_nodes": int(result.nodes),
                    "probe_timed_out": bool(result.timed_out),
                    "remaining_time_after_probe_ms": remaining_after_probe,
                    "raw_prediction_ms": raw_prediction_ms,
                    "post_probe_usable_time_ms": usable_after_probe,
                    "post_probe_shadow_budget_ms": capped_after_probe_ms,
                    "estimated_full_search_budget_ms": estimated_full_search_budget_ms,
                    "probe_clock_cost_percent": elapsed_ms / remaining_ms * 100.0 if remaining_ms else 0.0,
                    "model": "random_forest",
                    "model_version": model_version or "injected-test-model",
                    "feature_names": list(FEATURE_NAMES),
                })
    return records


def build_phase2f1_report(
    shadow_records: Sequence[dict[str, Any]],
    probe_records: Sequence[dict[str, Any]],
    *,
    seed: int,
    generated_fens: Sequence[str],
    search_budget_ms: int,
    max_depth: int = 64,
    probe_budgets_ms: Sequence[int] = PROBE_BUDGETS_MS,
    clock_buckets_ms: Sequence[int] = CLOCK_BUCKETS_MS,
) -> dict[str, Any]:
    """Build a self-contained collection, diagnostics, and probe feasibility report."""
    actual = [float(record["actual_time_ms"]) for record in shadow_records if _finite(record.get("actual_time_ms"))]
    predictions = [float(record["raw_prediction_ms"]) for record in shadow_records if _finite(record.get("raw_prediction_ms"))]
    capped = [float(record["shadow_budget_ms"]) for record in shadow_records if _finite(record.get("shadow_budget_ms"))]
    pairs = [record for record in shadow_records if _finite(record.get("actual_time_ms")) and _finite(record.get("shadow_budget_ms"))]
    bucket_report = {}
    for clock in clock_buckets_ms:
        group = [record for record in shadow_records if record.get("remaining_time_ms") == clock]
        bucket_report[str(clock)] = {
            "sample_count": len(group),
            "median_prediction_ms": _median([float(item["raw_prediction_ms"]) for item in group if _finite(item.get("raw_prediction_ms"))]),
            "median_shadow_budget_ms": _median([float(item["shadow_budget_ms"]) for item in group if _finite(item.get("shadow_budget_ms"))]),
            "median_actual_search_time_ms": _median([float(item["actual_time_ms"]) for item in group if _finite(item.get("actual_time_ms"))]),
            "cap_activation_rate": _rate([bool(item.get("clock_cap_activated")) for item in group]),
            "timeout_rate": _rate([bool(item.get("actual_timeout")) for item in group]),
            "usable_time_ms": max(clock - SAFETY_MARGIN_MS, 0),
        }
    errors = [abs(float(item["shadow_budget_ms"]) - float(item["actual_time_ms"])) for item in pairs]
    over = sum(float(item["shadow_budget_ms"]) > float(item["actual_time_ms"]) for item in pairs)
    under = sum(float(item["shadow_budget_ms"]) < float(item["actual_time_ms"]) for item in pairs)
    comparison = {
        "sample_count": len(pairs),
        "mae_ms": statistics.fmean(errors) if errors else None,
        "median_absolute_error_ms": statistics.median(errors) if errors else None,
        "overprediction_percent": over / len(pairs) * 100 if pairs else None,
        "underprediction_percent": under / len(pairs) * 100 if pairs else None,
        "pearson_correlation": _correlation(
            [float(item["shadow_budget_ms"]) for item in pairs],
            [float(item["actual_time_ms"]) for item in pairs],
        ),
        "comparison_target_note": "Actual runtime is observed engine behavior, not the teacher's ideal target; this is not direct prediction quality.",
    }
    probe_report = {}
    for budget in probe_budgets_ms:
        group = [item for item in probe_records if item.get("probe_budget_ms") == budget]
        probe_report[str(budget)] = {
            "sample_count": len(group),
            "probe_elapsed_ms": _distribution([float(item["probe_elapsed_ms"]) for item in group]),
            "completed_depth": _distribution([float(item["probe_completed_depth"]) for item in group]),
            "nodes": _distribution([float(item["probe_nodes"]) for item in group]),
            "remaining_clock_after_probe_ms": _distribution([float(item["remaining_time_after_probe_ms"]) for item in group]),
            "model_prediction_ms": _distribution([float(item["raw_prediction_ms"]) for item in group]),
            "estimated_full_search_budget_ms": _distribution([float(item["estimated_full_search_budget_ms"]) for item in group]),
            "probe_clock_cost_percent": _distribution([float(item["probe_clock_cost_percent"]) for item in group]),
            "probe_over_10_percent_rate": _rate([float(item["probe_clock_cost_percent"]) > 10 for item in group]),
        }
    total = len(shadow_records)
    enough = total >= 100 and len({record.get("fen") for record in shadow_records}) >= 10 and all(
        bucket_report[str(clock)]["sample_count"] > 0 for clock in clock_buckets_ms
    )
    return {
        "phase": "2F.1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "collection": {
            "total_real_searches": total,
            "unique_fen_count": len({record.get("fen") for record in shadow_records}),
            "generated_fens": list(generated_fens),
            "search_budget_cap_ms": search_budget_ms,
            "max_depth": max_depth,
            "seed": seed,
            "requested_coverage_sufficient": enough,
            "clock_bucket_distribution": {clock: sum(r.get("remaining_time_ms") == clock for r in shadow_records) for clock in clock_buckets_ms},
        },
        "raw_prediction_ms": _distribution(predictions),
        "final_shadow_budget_ms": _distribution(capped),
        "actual_search_time_ms": _distribution(actual),
        "prediction_vs_actual_runtime_diagnostics": comparison,
        "clock_safety": {
            "safety_margin_ms": SAFETY_MARGIN_MS,
            "nonnegative_shadow_budget_count": sum(float(record.get("shadow_budget_ms", -1)) >= 0 for record in shadow_records),
            "cap_activated_count": sum(bool(record.get("clock_cap_activated")) for record in shadow_records),
            "cap_activation_rate": _rate([bool(record.get("clock_cap_activated")) for record in shadow_records]),
            "by_clock_bucket_ms": bucket_report,
        },
        "pre_search_probe_experiment": {
            "production_behavior_changed": False,
            "model_remains_shadow_only": True,
            "budgets_ms": list(probe_budgets_ms),
            "results_by_probe_budget_ms": probe_report,
            "estimated_budget_definition": "min(max(min(max(raw_prediction_ms, 0), max(remaining_before_probe_ms - 300, 0)) - probe_elapsed_ms, 0), max(remaining_after_probe_ms - 300, 0))",
        },
        "limitations": [
            "The model was trained on offline teacher labels. Comparing its shadow recommendation with actual runtime search duration is not a direct measure of prediction quality.",
            "Search duration and telemetry depend on position and hardware. No Teacher was run during this collection, and no model was retrained.",
            "Probe predictions are a feasibility study only. Probe time is deducted from the remaining clock and is never used to control production searches.",
        ],
        "phase2g_recommendation": {
            "decision": "NOT YET",
            "reason": "Collection coverage is sufficient for this phase, but Phase 2G should wait for a direct held-out teacher-label comparison and review of probe cost on fast clocks. Runtime duration is not a teacher target, and this collection intentionally ran no Teacher.",
            "missing_evidence": [
                "A held-out, FEN-and-clock matched comparison against labeled teacher targets.",
                "A pre-search probe cost budget shown to be acceptable at low clock values; the probe study is diagnostic only.",
            ],
            "runtime_model_control_authorized": False,
        },
    }


def _distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "median": None, "mean": None, "p90": None, "max": None}
    ordered = sorted(max(0.0, float(value)) for value in values)
    point = (len(ordered) - 1) * 0.9
    low, high = math.floor(point), math.ceil(point)
    return {"count": len(ordered), "min": ordered[0], "median": statistics.median(ordered), "mean": statistics.fmean(ordered), "p90": ordered[low] + (ordered[high] - ordered[low]) * (point - low), "max": ordered[-1]}


def _median(values: Sequence[float]) -> float | None:
    return statistics.median(values) if values else None


def _rate(values: Sequence[bool]) -> float | None:
    return sum(values) / len(values) if values else None


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 3 or len(set(left)) < 2 or len(set(right)) < 2:
        return None
    return float(np.corrcoef(np.asarray(left), np.asarray(right))[0, 1])


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Collect Phase 2F.1 shadow searches and run offline probe validation.")
    parser.add_argument("--positions", type=int, default=12, help="Number of varied generated FEN positions (12 × 9 clocks = 108 searches).")
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--search-budget-ms", type=int, default=250)
    parser.add_argument("--max-depth", type=int, default=64)
    parser.add_argument("--log", default=str(DEFAULT_COLLECTION_PATH))
    parser.add_argument("--report", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--reuse-existing", action="store_true", help="Reuse the existing shadow JSONL and rerun only the offline probes/report.")
    args = parser.parse_args(argv)
    if args.reuse_existing:
        shadow_records = load_jsonl(args.log)
        fens = list(dict.fromkeys(str(record["fen"]) for record in shadow_records if record.get("fen")))
        if not shadow_records:
            parser.error("--reuse-existing requires a non-empty --log JSONL")
    else:
        fens = select_position_fens(seed=args.seed, position_count=args.positions)
        shadow_records = collect_shadow_searches(
            fens=fens,
            search_budget_ms=args.search_budget_ms,
            max_depth=args.max_depth,
            log_path=args.log,
        )
    probe_fens = fens[: min(4, len(fens))]
    probe_records = run_probe_experiment(fens=probe_fens, max_depth=args.max_depth)
    report = build_phase2f1_report(
        shadow_records,
        probe_records,
        seed=args.seed,
        generated_fens=fens,
        search_budget_ms=args.search_budget_ms,
        max_depth=args.max_depth,
    )
    output = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    probe_path = output.with_name("phase2f1_probe_experiment.jsonl")
    probe_path.write_text("".join(json.dumps(item, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n" for item in probe_records), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    print(f"Shadow JSONL: {args.log}")
    print(f"Probe JSONL: {probe_path}")
    print(f"Phase 2F.1 report: {output}")


if __name__ == "__main__":
    main()
