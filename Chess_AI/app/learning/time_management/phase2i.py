"""Small paired offline search comparison for Phase 2I.

Each row runs the existing fixed 250 ms baseline and the frozen model policy
from the same FEN and remaining clock. This is descriptive search telemetry,
not a playing-strength test.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
from time import perf_counter
from typing import Any, Sequence

import chess

from app.engine.evaluation import evaluate
from app.engine.iterative_search import IterativeSearchResult, iterative_search
from app.learning.time_management.phase2f2 import load_jsonl, safety_governor
from app.learning.time_management.runtime_policy import (
    BASELINE_BUDGET_MS,
    SAFETY_MARGIN_MS,
    decide_search_budget,
)
from app.learning.time_management.runtime_shadow import load_shadow_model


ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = ROOT / "data" / "time_management"
DEFAULT_HELDOUT_PATH = DATA_ROOT / "phase2f2_heldout_dataset.jsonl"
DEFAULT_RESULTS_PATH = DATA_ROOT / "phase2i_paired_results.jsonl"
DEFAULT_REPORT_PATH = DATA_ROOT / "phase2i_report.json"
DEFAULT_CASE_LIMIT = 10
MAX_DEPTH = 64


def run_case(
    record: dict[str, Any],
    *,
    model: Any | None = None,
    model_version: str | None = None,
    max_depth: int = MAX_DEPTH,
    baseline_budget_ms: float = BASELINE_BUDGET_MS,
) -> dict[str, Any]:
    """Run both policies on exactly one FEN/clock pair and retain raw metrics."""
    fen = record.get("fen")
    clock = record.get("clock", {}).get("remaining_time_ms")
    if not isinstance(fen, str) or not fen:
        raise ValueError("record requires a non-empty fen")
    if isinstance(clock, bool) or not isinstance(clock, int) or clock < 0:
        raise ValueError("record requires a non-negative integer remaining_time_ms")
    if baseline_budget_ms <= 0:
        raise ValueError("baseline_budget_ms must be positive")
    if not 1 <= max_depth <= MAX_DEPTH:
        raise ValueError(f"max_depth must be between 1 and {MAX_DEPTH}")

    board = chess.Board(fen)
    if not board.is_valid() or board.is_game_over(claim_draw=True):
        raise ValueError("record FEN must describe a valid non-terminal position")

    baseline_budget, _, baseline_capped = safety_governor(
        baseline_budget_ms, clock, SAFETY_MARGIN_MS,
    )
    baseline, baseline_search_ms = _run_search(board, baseline_budget, max_depth)

    decision = decide_search_budget(
        board,
        remaining_time_ms=clock,
        max_depth=max_depth,
        model=model,
    )
    model_result, model_search_ms = _run_search(
        board, decision.final_search_budget_ms, max_depth,
    )
    model_total_ms = decision.policy_latency_ms + model_search_ms

    baseline_evaluation = _evaluate_after_move(board, baseline.move)
    model_evaluation = _evaluate_after_move(board, model_result.move)
    baseline_allowed_budget = max(float(clock) - SAFETY_MARGIN_MS, 0.0)
    model_allowed_budget = max(float(clock) - decision.policy_latency_ms - SAFETY_MARGIN_MS, 0.0)
    return {
        "sample_id": record.get("sample_id"),
        "fen": fen,
        "remaining_time_ms": clock,
        "baseline_move": baseline.move.uci() if baseline.move else None,
        "model_move": model_result.move.uci() if model_result.move else None,
        "baseline_time_ms": baseline_search_ms,
        "model_total_time_ms": model_total_ms,
        "model_search_time_ms": model_search_ms,
        "baseline_search_budget_ms": baseline_budget,
        "model_predicted_time_ms": decision.predicted_time_ms,
        "model_final_search_budget_ms": decision.final_search_budget_ms,
        "probe_time_ms": decision.probe_time_ms,
        "model_load_time_ms": decision.model_load_time_ms,
        "feature_extraction_time_ms": decision.feature_extraction_time_ms,
        "inference_latency_ms": decision.inference_latency_ms,
        "model_policy_time_ms": decision.policy_latency_ms,
        "baseline_completed_depth": baseline.completed_depth,
        "model_completed_depth": model_result.completed_depth,
        "baseline_nodes": baseline.nodes,
        "model_nodes": model_result.nodes,
        "baseline_timed_out": baseline.timed_out,
        "model_timed_out": model_result.timed_out,
        "baseline_evaluation_white_cp": baseline_evaluation,
        "model_evaluation_white_cp": model_evaluation,
        "evaluation_difference_model_minus_baseline_cp": (
            model_evaluation - baseline_evaluation
            if model_evaluation is not None and baseline_evaluation is not None else None
        ),
        "baseline_safety_cap_applied": baseline_capped,
        "model_safety_cap_applied": decision.safety_cap_applied,
        "model_fallback_used": decision.fallback_used,
        "model_fallback_reason": decision.fallback_reason,
        "model_version": model_version or decision.model_version,
        "baseline_safety_violation": baseline_budget > baseline_allowed_budget + 1e-9,
        "model_safety_violation": decision.final_search_budget_ms > model_allowed_budget + 1e-9,
        "model_negative_budget": decision.final_search_budget_ms < 0,
        "quality_note": "Material evaluation is descriptive only; playing-strength improvement is not established.",
    }


def _run_search(board: chess.Board, budget_ms: float, max_depth: int) -> tuple[IterativeSearchResult, float]:
    """Search within a positive budget or return a legal zero-budget move."""
    if budget_ms <= 0:
        move = next(iter(board.legal_moves), None)
        if move is None:
            return IterativeSearchResult(None, 0, 0, 0, 0.0, False, ()), 0.0
        return IterativeSearchResult(move, 0, 0, 0, 0.0, False, ()), 0.0
    started = perf_counter()
    result = iterative_search(board.copy(stack=False), budget_ms, max_depth=max_depth)
    actual_ms = max((perf_counter() - started) * 1_000, float(result.time_ms))
    return result, actual_ms


def _evaluate_after_move(board: chess.Board, move: chess.Move | None) -> int | None:
    if move is None:
        return None
    resulting_board = board.copy(stack=False)
    resulting_board.push(move)
    return evaluate(resulting_board)


def select_records(records: Sequence[dict[str, Any]], limit: int | None = DEFAULT_CASE_LIMIT) -> list[dict[str, Any]]:
    """Choose deterministic, evenly spaced cases while preserving exact pairs."""
    if limit is None or limit <= 0 or len(records) <= limit:
        return list(records)
    if limit == 1:
        return [records[0]]
    indices = [round(index * (len(records) - 1) / (limit - 1)) for index in range(limit)]
    return [records[index] for index in indices]


def run_experiment(
    records: Sequence[dict[str, Any]],
    *,
    model: Any | None = None,
    model_version: str | None = None,
    max_depth: int = MAX_DEPTH,
    baseline_budget_ms: float = BASELINE_BUDGET_MS,
) -> list[dict[str, Any]]:
    if not records:
        raise ValueError("no held-out records were selected")
    if model is None:
        model, loaded_version = load_shadow_model()
        model_version = model_version or loaded_version
    return [
        run_case(
            record,
            model=model,
            model_version=model_version,
            max_depth=max_depth,
            baseline_budget_ms=baseline_budget_ms,
        )
        for record in records
    ]


def summarize(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Summarize paired search telemetry without declaring a winner."""
    return {
        "phase": "2I",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "test_cases": len(records),
        "distinct_fens": len({row["fen"] for row in records}),
        "baseline": {
            "average_search_time_ms": _mean(records, "baseline_time_ms"),
            "average_completed_depth": _mean(records, "baseline_completed_depth"),
            "average_nodes": _mean(records, "baseline_nodes"),
        },
        "model": {
            "average_predicted_time_ms": _mean(records, "model_predicted_time_ms"),
            "average_final_search_budget_ms": _mean(records, "model_final_search_budget_ms"),
            "average_search_time_ms": _mean(records, "model_search_time_ms"),
            "average_total_time_ms": _mean(records, "model_total_time_ms"),
            "average_completed_depth": _mean(records, "model_completed_depth"),
            "average_nodes": _mean(records, "model_nodes"),
            "average_model_load_time_ms": _mean(records, "model_load_time_ms"),
            "average_feature_extraction_time_ms": _mean(records, "feature_extraction_time_ms"),
            "average_inference_latency_ms": _mean(records, "inference_latency_ms"),
            "fallback_cases": sum(bool(row["model_fallback_used"]) for row in records),
        },
        "paired_differences": {
            "model_minus_baseline_search_time_ms": _mean([
                row["model_search_time_ms"] - row["baseline_time_ms"] for row in records
            ]),
            "model_minus_baseline_completed_depth": _mean([
                row["model_completed_depth"] - row["baseline_completed_depth"] for row in records
            ]),
            "model_minus_baseline_nodes": _mean([
                row["model_nodes"] - row["baseline_nodes"] for row in records
            ]),
            "model_minus_baseline_evaluation_white_cp": _mean([
                row["evaluation_difference_model_minus_baseline_cp"] for row in records
            ]),
            "different_move_cases": sum(row["model_move"] != row["baseline_move"] for row in records),
            "interpretation": "Descriptive paired metrics only; material evaluation is not a playing-strength metric.",
        },
        "safety": {
            "baseline_budget_violations": sum(bool(row["baseline_safety_violation"]) for row in records),
            "model_budget_violations": sum(bool(row["model_safety_violation"]) for row in records),
            "negative_model_budgets": sum(bool(row["model_negative_budget"]) for row in records),
        },
        "playing_strength_improvement_established": False,
    }


def _mean(records_or_values: Sequence[dict[str, Any]] | Sequence[float | int | None], field: str | None = None) -> float | None:
    values = [row.get(field) for row in records_or_values if isinstance(row.get(field), (int, float))] if field else [value for value in records_or_values if isinstance(value, (int, float))]
    return statistics.fmean(values) if values else None


def write_jsonl(path: str | Path, records: Sequence[dict[str, Any]]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n" for row in records),
        encoding="utf-8",
    )
    return output


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a small paired Phase 2I offline search comparison.")
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run", help="Run fixed-baseline and model searches on paired held-out positions.")
    run_parser.add_argument("--heldout", default=str(DEFAULT_HELDOUT_PATH))
    run_parser.add_argument("--output", default=str(DEFAULT_RESULTS_PATH))
    run_parser.add_argument("--report", default=str(DEFAULT_REPORT_PATH))
    run_parser.add_argument("--limit", type=int, default=DEFAULT_CASE_LIMIT, help="Maximum cases; 0 runs all rows.")
    run_parser.add_argument("--max-depth", type=int, default=MAX_DEPTH)
    report_parser = commands.add_parser("report", help="Summarize a Phase 2I JSONL results file.")
    report_parser.add_argument("--results", default=str(DEFAULT_RESULTS_PATH))
    report_parser.add_argument("--output", default=str(DEFAULT_REPORT_PATH))
    args = parser.parse_args(argv)

    if args.command == "run":
        source = load_jsonl(args.heldout)
        selected = select_records(source, args.limit)
        results = run_experiment(selected, max_depth=args.max_depth)
        write_jsonl(args.output, results)
        report = summarize(results)
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({"test_cases": len(results), "distinct_fens": report["distinct_fens"], "safety": report["safety"], "results": args.output, "report": args.report}, ensure_ascii=False, indent=2))
    else:
        records = load_jsonl(args.results)
        report = summarize(records)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
