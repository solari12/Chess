"""Controlled offline runtime comparison for Phase 2G.

This experiment is isolated from the production API. It runs a fixed-budget
baseline and a probe-plus-model policy on the same held-out FEN/clock rows.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
from time import perf_counter
from typing import Any, Sequence

import chess
import numpy as np

from app.engine.iterative_search import IterativeSearchResult, iterative_search
from app.learning.time_management.features import extract_position_features
from app.learning.time_management.model_features import FEATURE_NAMES, extract_model_features
from app.learning.time_management.phase2f2 import load_jsonl
from app.learning.time_management.runtime_shadow import (
    DEFAULT_MODEL_PATH,
    SAFETY_MARGIN_MS,
    load_shadow_model,
)


ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = ROOT / "data" / "time_management"
DEFAULT_HELDOUT_PATH = DATA_ROOT / "phase2f2_heldout_dataset.jsonl"
DEFAULT_RESULTS_PATH = DATA_ROOT / "phase2g_runtime_results.jsonl"
DEFAULT_REPORT_PATH = DATA_ROOT / "phase2g_report.json"
BASELINE_BUDGET_MS = 250
DEFAULT_PROBE_BUDGET_MS = 25
MAX_DEPTH = 64


def safe_budget(
    raw_prediction_ms: float,
    remaining_time_ms: int,
    *,
    probe_elapsed_ms: float = 0.0,
    inference_elapsed_ms: float = 0.0,
    safety_margin_ms: float = SAFETY_MARGIN_MS,
) -> tuple[float, float, bool]:
    """Cap a new search by clock left after probe/inference and the reserve."""
    if not math.isfinite(raw_prediction_ms):
        raise ValueError("raw prediction must be finite")
    if remaining_time_ms < 0 or not math.isfinite(probe_elapsed_ms) or probe_elapsed_ms < 0:
        raise ValueError("clock and probe elapsed time must be non-negative")
    if not math.isfinite(inference_elapsed_ms) or inference_elapsed_ms < 0:
        raise ValueError("inference elapsed time must be non-negative")
    if not math.isfinite(safety_margin_ms) or safety_margin_ms < 0:
        raise ValueError("safety margin must be finite and non-negative")
    usable = max(float(remaining_time_ms) - probe_elapsed_ms - inference_elapsed_ms - safety_margin_ms, 0.0)
    nonnegative_prediction = max(float(raw_prediction_ms), 0.0)
    return min(nonnegative_prediction, usable), usable, float(raw_prediction_ms) > usable or raw_prediction_ms < 0


def _make_model_feature_record(board: chess.Board, remaining_ms: float, result: IterativeSearchResult, budget_ms: float, max_depth: int) -> dict[str, Any]:
    clock_ms = max(0, int(remaining_ms))
    return {
        "fen": board.fen(),
        "clock": {"remaining_time_ms": clock_ms},
        "position_features": extract_position_features(board, clock_ms),
        "search": {
            "completed_depth": result.completed_depth,
            "total_nodes": result.nodes,
            "total_time_ms": result.time_ms,
            "timed_out": result.timed_out,
            "analysis_budget_ms": budget_ms,
            "max_analysis_depth": max_depth,
        },
    }


def _search_metrics(result: IterativeSearchResult) -> dict[str, Any]:
    return {
        "actual_time_ms": float(result.time_ms),
        "completed_depth": int(result.completed_depth),
        "nodes": int(result.nodes),
        "timed_out": bool(result.timed_out),
        "move": result.move.uci() if result.move else None,
        "score": int(result.score),
        "depths_completed": [
            {
                "depth": item.depth,
                "move": item.move,
                "score": item.score,
                "nodes": item.nodes,
                "time_ms": float(item.time_ms),
            }
            for item in result.depths_completed
        ],
    }


def run_case(
    record: dict[str, Any],
    *,
    model: Any,
    model_version: str,
    baseline_budget_ms: int = BASELINE_BUDGET_MS,
    probe_budget_ms: int = DEFAULT_PROBE_BUDGET_MS,
    safety_margin_ms: float = SAFETY_MARGIN_MS,
    max_depth: int = MAX_DEPTH,
) -> dict[str, Any]:
    """Run both isolated branches on the same input FEN and clock."""
    fen = record.get("fen")
    clock = record.get("clock", {}).get("remaining_time_ms")
    if not isinstance(fen, str) or not fen:
        raise ValueError("held-out row requires a non-empty fen")
    if isinstance(clock, bool) or not isinstance(clock, int) or clock < 0:
        raise ValueError("held-out row requires a non-negative integer remaining_time_ms")
    if baseline_budget_ms <= 0 or probe_budget_ms <= 0:
        raise ValueError("baseline and probe budgets must be positive")
    initial_board = chess.Board(fen)
    if not initial_board.is_valid():
        raise ValueError(f"Invalid held-out FEN: {fen}")

    baseline = _run_baseline(
        fen, clock, baseline_budget_ms=baseline_budget_ms,
        safety_margin_ms=safety_margin_ms, max_depth=max_depth,
    )
    model_result = _run_model_branch(
        fen, clock, model=model, model_version=model_version,
        probe_budget_ms=probe_budget_ms,
        safety_margin_ms=safety_margin_ms,
        max_depth=max_depth,
    )
    return {
        "sample_id": record.get("sample_id"),
        "fen": fen,
        "remaining_time_ms": clock,
        "baseline": baseline,
        "model": model_result,
        "input_source": "phase2f2_heldout_dataset",
    }


def _run_baseline(fen: str, remaining_ms: int, *, baseline_budget_ms: int, safety_margin_ms: float, max_depth: int) -> dict[str, Any]:
    usable = max(float(remaining_ms) - safety_margin_ms, 0.0)
    allocated = min(float(baseline_budget_ms), usable)
    common = {
        "requested_budget_ms": baseline_budget_ms,
        "allocated_budget_ms": allocated,
        "usable_time_ms": usable,
        "skipped": allocated <= 0,
        "skipped_reason": "no_usable_time_after_safety_reserve" if allocated <= 0 else None,
        "safety_cap_applied": allocated < baseline_budget_ms,
    }
    if allocated <= 0:
        return {**common, "actual_time_ms": 0.0, "completed_depth": 0, "nodes": 0, "timed_out": False, "move": None, "score": None, "depths_completed": [], "runtime_error": None}
    try:
        board = chess.Board(fen)
        result = iterative_search(board, allocated, max_depth=max_depth)
        return {**common, **_search_metrics(result), "runtime_error": None}
    except Exception as error:
        return {**common, "actual_time_ms": None, "completed_depth": None, "nodes": None, "timed_out": None, "move": None, "score": None, "depths_completed": [], "runtime_error": f"{type(error).__name__}: {error}"}


def _run_model_branch(
    fen: str,
    remaining_ms: int,
    *,
    model: Any,
    model_version: str,
    probe_budget_ms: int,
    safety_margin_ms: float,
    max_depth: int,
) -> dict[str, Any]:
    base = {
        "probe_budget_ms": probe_budget_ms,
        "actual_probe_time_ms": 0.0,
        "predicted_time_ms": None,
        "final_model_budget_ms": 0.0,
        "actual_search_time_ms": 0.0,
        "total_policy_time_ms": 0.0,
        "completed_depth": 0,
        "nodes": 0,
        "timed_out": False,
        "move": None,
        "score": None,
        "depths_completed": [],
        "safety_cap_applied": False,
        "skipped": False,
        "skipped_reason": None,
        "model_inference_latency_ms": None,
        "usable_time_ms": max(float(remaining_ms) - safety_margin_ms, 0.0),
        "total_clock_consumption_ms": 0.0,
        "model_version": model_version,
        "runtime_error": None,
    }
    if remaining_ms <= safety_margin_ms:
        return {**base, "skipped": True, "skipped_reason": "no_usable_time_after_safety_reserve"}

    probe_usable = max(float(remaining_ms) - safety_margin_ms, 0.0)
    if probe_usable <= 0:
        return {**base, "skipped": True, "skipped_reason": "no_usable_time_for_probe_after_safety_reserve"}
    actual_probe_start = perf_counter()
    try:
        probe_board = chess.Board(fen)
        probe_result = iterative_search(probe_board, float(probe_budget_ms), max_depth=max_depth)
        probe_elapsed = max((perf_counter() - actual_probe_start) * 1_000, float(probe_result.time_ms))
    except Exception as error:
        probe_elapsed = max(0.0, (perf_counter() - actual_probe_start) * 1_000)
        return {
            **base,
            "actual_probe_time_ms": probe_elapsed,
            "total_policy_time_ms": probe_elapsed,
            "skipped": True,
            "skipped_reason": "probe_failed",
            "runtime_error": f"{type(error).__name__}: {error}",
        }

    usable_after_probe = max(float(remaining_ms) - probe_elapsed - safety_margin_ms, 0.0)
    updated = {
        **base,
        "actual_probe_time_ms": probe_elapsed,
        "probe_completed_depth": int(probe_result.completed_depth),
        "probe_nodes": int(probe_result.nodes),
        "probe_timed_out": bool(probe_result.timed_out),
        "usable_time_ms": usable_after_probe,
        "total_policy_time_ms": probe_elapsed,
        "total_clock_consumption_ms": probe_elapsed,
    }
    if usable_after_probe <= 0:
        return {**updated, "skipped": True, "skipped_reason": "probe_consumed_safety_reserve"}

    try:
        inference_start = perf_counter()
        inference_board = chess.Board(fen)
        remaining_after_probe = max(float(remaining_ms) - probe_elapsed, 0.0)
        feature_record = _make_model_feature_record(
            inference_board,
            remaining_after_probe,
            probe_result,
            float(probe_budget_ms),
            max_depth,
        )
        features = extract_model_features(feature_record)
        if features.shape != (len(FEATURE_NAMES),):
            raise ValueError("feature vector does not match Phase 2C model schema")
        prediction = np.asarray(model.predict(features.reshape(1, -1)), dtype=np.float64)
        inference_ms = max((perf_counter() - inference_start) * 1_000, 0.0)
        if prediction.shape != (1,) or not np.isfinite(prediction).all():
            raise ValueError("Random Forest returned invalid prediction shape or a non-finite value")
        raw_prediction = float(prediction[0])
        final_budget, usable_after_probe, cap_applied = safe_budget(
            raw_prediction,
            remaining_ms,
            probe_elapsed_ms=probe_elapsed,
            inference_elapsed_ms=inference_ms,
            safety_margin_ms=safety_margin_ms,
        )
        updated.update({
            "predicted_time_ms": raw_prediction,
            "model_inference_latency_ms": inference_ms,
            "usable_time_ms": usable_after_probe,
            "final_model_budget_ms": final_budget,
            "safety_cap_applied": cap_applied,
            "negative_prediction_clamped": raw_prediction < 0,
            "total_clock_consumption_ms": probe_elapsed + inference_ms,
        })
        if final_budget <= 0:
            return {**updated, "skipped": True, "skipped_reason": "model_budget_is_zero_after_safety_cap"}
        search_start = perf_counter()
        search_result = iterative_search(chess.Board(fen), final_budget, max_depth=max_depth)
        actual_search_ms = max((perf_counter() - search_start) * 1_000, float(search_result.time_ms))
        updated.update({
            "actual_search_time_ms": actual_search_ms,
            "total_policy_time_ms": probe_elapsed + actual_search_ms,
            "total_clock_consumption_ms": probe_elapsed + inference_ms + actual_search_ms,
            "completed_depth": int(search_result.completed_depth),
            "nodes": int(search_result.nodes),
            "timed_out": bool(search_result.timed_out),
            "move": search_result.move.uci() if search_result.move else None,
            "score": int(search_result.score),
            "depths_completed": _search_metrics(search_result)["depths_completed"],
        })
        return updated
    except Exception as error:
        return {
            **updated,
            "skipped": True,
            "skipped_reason": "model_or_search_failed",
            "runtime_error": f"{type(error).__name__}: {error}",
        }


def run_experiment(
    records: Sequence[dict[str, Any]],
    *,
    model: Any | None = None,
    model_version: str | None = None,
    baseline_budget_ms: int = BASELINE_BUDGET_MS,
    probe_budget_ms: int = DEFAULT_PROBE_BUDGET_MS,
    safety_margin_ms: float = SAFETY_MARGIN_MS,
    max_depth: int = MAX_DEPTH,
) -> list[dict[str, Any]]:
    """Run one paired baseline/model case for every supplied held-out row."""
    if model is None:
        model, loaded_version = load_shadow_model()
        model_version = model_version or loaded_version
    if not records:
        raise ValueError("held-out dataset is empty")
    return [
        run_case(
            record,
            model=model,
            model_version=model_version or "injected-test-model",
            baseline_budget_ms=baseline_budget_ms,
            probe_budget_ms=probe_budget_ms,
            safety_margin_ms=safety_margin_ms,
            max_depth=max_depth,
        )
        for record in records
    ]


def report_results(records: Sequence[dict[str, Any]], *, safety_margin_ms: float = SAFETY_MARGIN_MS) -> dict[str, Any]:
    """Summarize paired run telemetry and safety without declaring a winner."""
    fen_groups = {str(row.get("fen")) for row in records if row.get("fen")}
    baseline_rows = [row.get("baseline", {}) for row in records]
    model_rows = [row.get("model", {}) for row in records]
    baseline_searched = [row for row in baseline_rows if not row.get("skipped") and not row.get("runtime_error")]
    model_searched = [row for row in model_rows if not row.get("skipped") and not row.get("runtime_error")]
    failures = [row for row in records if row.get("baseline", {}).get("runtime_error") or row.get("model", {}).get("runtime_error")]

    violations = []
    negative_budgets = []
    reserve_violations = []
    skipped_cases = []
    missing_telemetry = []
    for row in records:
        clock = row.get("remaining_time_ms")
        if not isinstance(clock, (int, float)) or isinstance(clock, bool):
            missing_telemetry.append({"sample_id": row.get("sample_id"), "branch": "input", "field": "remaining_time_ms"})
            continue
        allowed_total = max(float(clock) - safety_margin_ms, 0.0)
        for branch_name in ("baseline", "model"):
            branch = row.get(branch_name, {})
            allocated = branch.get("allocated_budget_ms", branch.get("final_model_budget_ms", 0.0))
            prefix = f"{row.get('sample_id') or row.get('fen')}:{branch_name}"
            if isinstance(allocated, (int, float)) and not isinstance(allocated, bool):
                if allocated < 0:
                    negative_budgets.append(prefix)
                allowed_search = max(float(clock) - safety_margin_ms - float(branch.get("actual_probe_time_ms", 0.0)), 0.0)
                if allocated > allowed_search + 1e-9:
                    violations.append({"case": prefix, "allocated_budget_ms": allocated, "allowed_budget_ms": allowed_search})
            else:
                missing_telemetry.append({"sample_id": row.get("sample_id"), "branch": branch_name, "field": "allocated_budget_ms"})
            if branch.get("skipped"):
                skipped_cases.append({"sample_id": row.get("sample_id"), "branch": branch_name, "reason": branch.get("skipped_reason")})
            actual = branch.get("actual_time_ms") if branch_name == "baseline" else branch.get("actual_search_time_ms")
            clock_elapsed = branch.get("actual_time_ms") if branch_name == "baseline" else branch.get("total_clock_consumption_ms", branch.get("total_policy_time_ms"))
            if isinstance(clock_elapsed, (int, float)) and not isinstance(clock_elapsed, bool) and clock_elapsed > allowed_total + 1e-9:
                reserve_violations.append({"case": prefix, "actual_clock_consumption_ms": clock_elapsed, "remaining_after_reserve_ms": allowed_total})
            if not branch.get("skipped") and not branch.get("runtime_error"):
                for field in ("completed_depth", "nodes", "timed_out"):
                    if branch.get(field) is None:
                        missing_telemetry.append({"sample_id": row.get("sample_id"), "branch": branch_name, "field": field})
                if actual is None:
                    missing_telemetry.append({"sample_id": row.get("sample_id"), "branch": branch_name, "field": "actual_search_time_ms"})

    paired = [
        (baseline, model)
        for baseline, model in zip(baseline_rows, model_rows)
        if not baseline.get("skipped") and not baseline.get("runtime_error")
        and not model.get("skipped") and not model.get("runtime_error")
    ]
    comparison = {
        "paired_case_count": len(paired),
        "mean_model_minus_baseline_actual_time_ms": _mean([
            float(model["actual_search_time_ms"]) - float(baseline["actual_time_ms"])
            for baseline, model in paired
        ]),
        "mean_model_minus_baseline_completed_depth": _mean([
            float(model["completed_depth"]) - float(baseline["completed_depth"])
            for baseline, model in paired
        ]),
        "mean_model_minus_baseline_nodes": _mean([
            float(model["nodes"]) - float(baseline["nodes"])
            for baseline, model in paired
        ]),
        "timeout_rate_difference_percentage_points": (
            (_rate([bool(model["timed_out"]) for _, model in paired]) - _rate([bool(baseline["timed_out"]) for baseline, _ in paired])) * 100
            if paired else None
        ),
        "interpretation": "Descriptive paired differences only. No branch is designated a winner; depth, nodes, and elapsed time are not standalone quality measures.",
    }
    all_cases_complete = len(records) > 0 and len(records) == len([row for row in records if row.get("baseline") and row.get("model")])
    safety_ok = not violations and not negative_budgets and not reserve_violations
    telemetry_ok = not missing_telemetry
    model_ok = len(model_rows) == len(records) and all(row.get("runtime_error") is None for row in model_rows)
    recommendation = "GO" if all_cases_complete and safety_ok and telemetry_ok and model_ok and not failures else "NOT_YET"

    report = {
        "phase": "2G",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "test_cases": len(records),
        "fen_groups": len(fen_groups),
        "remaining_time_buckets_ms": _bucket_report(records),
        "baseline": _branch_summary(baseline_rows, baseline_searched, branch="baseline"),
        "model": _branch_summary(model_rows, model_searched, branch="model"),
        "baseline_vs_model_raw_differences": comparison,
        "safety": {
            "safety_margin_ms": safety_margin_ms,
            "safety_violations": len(violations),
            "violations": violations,
            "negative_budgets": len(negative_budgets),
            "negative_budget_cases": negative_budgets,
            "reserve_violations": len(reserve_violations),
            "reserve_violation_cases": reserve_violations,
            "skipped_cases": len(skipped_cases),
            "skipped_details": skipped_cases,
            "missing_telemetry_count": len(missing_telemetry),
            "missing_telemetry": missing_telemetry,
        },
        "runtime_failures": failures,
        "recommendation": {
            "status": recommendation,
            "meaning": "GO means the controlled experiment completed safely with sufficient telemetry. It does not establish model superiority or production readiness and does not authorize production integration.",
            "checks": {
                "all_test_cases_have_both_branches": all_cases_complete,
                "zero_safety_violations": len(violations) == 0,
                "zero_negative_budgets": len(negative_budgets) == 0,
                "zero_reserve_violations": len(reserve_violations) == 0,
                "no_runtime_failures": not failures,
                "telemetry_complete": telemetry_ok,
                "model_branch_operational": model_ok,
            },
        },
        "experiment_configuration": {
            "baseline_requested_budget_ms": BASELINE_BUDGET_MS,
            "model_probe_budget_ms": DEFAULT_PROBE_BUDGET_MS,
            "model_name": "random_forest",
            "model_artifact": str(DEFAULT_MODEL_PATH.relative_to(ROOT)).replace("\\", "/"),
            "feature_schema": list(FEATURE_NAMES),
            "max_depth": MAX_DEPTH,
            "safety_reserve_ms": safety_margin_ms,
            "production_policy_changed": False,
        },
    }
    return report


def _branch_summary(all_rows: Sequence[dict[str, Any]], searched: Sequence[dict[str, Any]], *, branch: str) -> dict[str, Any]:
    if branch == "baseline":
        times = [float(row["actual_time_ms"]) for row in searched if _finite(row.get("actual_time_ms"))]
        budgets = [float(row["allocated_budget_ms"]) for row in all_rows if _finite(row.get("allocated_budget_ms"))]
        predicted_budgets = []
        inference = []
        policy = times
        depths = [float(row["completed_depth"]) for row in searched if _finite(row.get("completed_depth"))]
        nodes = [float(row["nodes"]) for row in searched if _finite(row.get("nodes"))]
    else:
        times = [float(row["actual_search_time_ms"]) for row in searched if _finite(row.get("actual_search_time_ms"))]
        budgets = [float(row["final_model_budget_ms"]) for row in all_rows if _finite(row.get("final_model_budget_ms"))]
        predicted_budgets = [float(row["predicted_time_ms"]) for row in all_rows if _finite(row.get("predicted_time_ms"))]
        inference = [float(row["model_inference_latency_ms"]) for row in all_rows if _finite(row.get("model_inference_latency_ms"))]
        policy = [float(row["total_policy_time_ms"]) for row in searched if _finite(row.get("total_policy_time_ms"))]
        depths = [float(row["completed_depth"]) for row in searched if _finite(row.get("completed_depth"))]
        nodes = [float(row["nodes"]) for row in searched if _finite(row.get("nodes"))]
    return {
        "requested_budget_ms": BASELINE_BUDGET_MS if branch == "baseline" else None,
        "average_allocated_budget_ms": _mean(budgets),
        "average_final_model_budget_ms": _mean(budgets) if branch == "model" else None,
        "average_predicted_budget_ms": _mean(predicted_budgets) if branch == "model" else None,
        "average_probe_time_ms": _mean([float(row["actual_probe_time_ms"]) for row in all_rows if _finite(row.get("actual_probe_time_ms"))]) if branch == "model" else 0.0,
        "average_search_time_ms": _mean(times),
        "average_total_policy_time_ms": _mean(policy),
        "average_completed_depth": _mean(depths),
        "average_nodes": _mean(nodes),
        "timeout_rate": _rate([bool(row["timed_out"]) for row in searched if isinstance(row.get("timed_out"), bool)]),
        "searched_cases": len(searched),
        "skipped_cases": sum(bool(row.get("skipped")) for row in all_rows),
        "runtime_failure_cases": sum(bool(row.get("runtime_error")) for row in all_rows),
        "model_inference_latency_ms": _stats(inference) if branch == "model" else None,
        "safety_cap_applied_count": sum(bool(row.get("safety_cap_applied")) for row in all_rows),
    }


def _bucket_report(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        key = str(record.get("remaining_time_ms"))
        buckets.setdefault(key, []).append(record)
    return {
        clock: {
            "test_cases": len(group),
            "fen_groups": len({row.get("fen") for row in group}),
            "baseline": _branch_summary([row["baseline"] for row in group], [row["baseline"] for row in group if not row["baseline"].get("skipped") and not row["baseline"].get("runtime_error")], branch="baseline"),
            "model": _branch_summary([row["model"] for row in group], [row["model"] for row in group if not row["model"].get("skipped") and not row["model"].get("runtime_error")], branch="model"),
        }
        for clock, group in sorted(buckets.items(), key=lambda item: float(item[0]), reverse=True)
    }


def load_results(path: str | Path) -> list[dict[str, Any]]:
    return load_jsonl(path)


def write_jsonl(path: str | Path, records: Sequence[dict[str, Any]]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n" for row in records), encoding="utf-8")
    return output


def _stats(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {"count": len(values), "mean": statistics.fmean(values), "median": statistics.median(values), "min": min(values), "max": max(values)}


def _mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _rate(values: Sequence[bool]) -> float | None:
    return sum(values) / len(values) if values else None


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run or report the isolated Phase 2G controlled runtime experiment.")
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run", help="Run paired baseline/model searches on the Phase 2F.2 holdout.")
    run_parser.add_argument("--heldout", default=str(DEFAULT_HELDOUT_PATH))
    run_parser.add_argument("--output", default=str(DEFAULT_RESULTS_PATH))
    run_parser.add_argument("--baseline-budget-ms", type=int, default=BASELINE_BUDGET_MS)
    run_parser.add_argument("--probe-budget-ms", type=int, default=DEFAULT_PROBE_BUDGET_MS)
    run_parser.add_argument("--max-depth", type=int, default=MAX_DEPTH)
    report_parser = commands.add_parser("report", help="Summarize a Phase 2G runtime results JSONL.")
    report_parser.add_argument("--results", default=str(DEFAULT_RESULTS_PATH))
    report_parser.add_argument("--output", default=str(DEFAULT_REPORT_PATH))
    args = parser.parse_args(argv)

    if args.command == "run":
        records = load_jsonl(args.heldout)
        model, version = load_shadow_model()
        results = run_experiment(
            records,
            model=model,
            model_version=version,
            baseline_budget_ms=args.baseline_budget_ms,
            probe_budget_ms=args.probe_budget_ms,
            max_depth=args.max_depth,
        )
        write_jsonl(args.output, results)
        print(f"Phase 2G result rows: {len(results)}")
        print(f"JSONL: {args.output}")
    else:
        results = load_results(args.results)
        report = report_results(results)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({
            "test_cases": report["test_cases"],
            "fen_groups": report["fen_groups"],
            "baseline": report["baseline"],
            "model": report["model"],
            "safety": {key: report["safety"][key] for key in ("safety_violations", "negative_budgets", "reserve_violations", "skipped_cases")},
            "recommendation": report["recommendation"],
        }, ensure_ascii=False, indent=2, allow_nan=False))
        print(f"Phase 2G report: {output}")


if __name__ == "__main__":
    main()
