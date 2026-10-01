"""Run iterative search and build descriptive per-position telemetry."""

from collections.abc import Callable
import math

import chess

from app.engine.evaluation import evaluate
from app.engine.iterative_search import iterative_search

Evaluator = Callable[[chess.Board], int]


def collect_search_telemetry(
    board: chess.Board,
    *,
    analysis_budget_ms: int,
    max_analysis_depth: int,
    evaluator: Evaluator = evaluate,
) -> dict:
    """Collect only observed search outputs; no target time is inferred."""
    result = iterative_search(
        board,
        time_budget_ms=analysis_budget_ms,
        max_depth=max_analysis_depth,
        evaluator=evaluator,
    )
    profiles: list[dict] = []
    previous: dict | None = None
    for depth in result.depths_completed:
        profile = {
            "depth": depth.depth,
            "move": depth.move,
            "score": depth.score,
            "nodes": depth.nodes,
            "time_ms": depth.time_ms,
        }
        if previous is None:
            profile.update(
                {
                    "best_move_changed": None,
                    "score_delta": None,
                    "time_growth_ratio": None,
                    "node_growth_ratio": None,
                }
            )
        else:
            profile.update(
                {
                    "best_move_changed": depth.move != previous["move"],
                    "score_delta": depth.score - previous["score"],
                    "time_growth_ratio": _ratio(depth.time_ms, previous["time_ms"]),
                    "node_growth_ratio": _ratio(depth.nodes, previous["nodes"]),
                }
            )
        profiles.append(profile)
        previous = profile

    return {
        "analysis_budget_ms": analysis_budget_ms,
        "max_analysis_depth": max_analysis_depth,
        "completed_depth": result.completed_depth,
        "timed_out": result.timed_out,
        "total_time_ms": result.time_ms,
        "total_nodes": result.nodes,
        "depth_profiles": profiles,
    }


def _ratio(current: float, previous: float) -> float | None:
    if previous == 0:
        return None
    ratio = current / previous
    return ratio if math.isfinite(ratio) else None
