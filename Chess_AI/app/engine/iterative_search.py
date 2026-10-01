"""Time-budgeted iterative deepening around the existing Alpha-Beta search."""

import logging
from dataclasses import dataclass
from time import monotonic
from typing import Callable

import chess

from app.engine.alpha_beta import SearchTimeout, alpha_beta
from app.engine.evaluation import evaluate
from app.engine.time_manager import TIME_SAFETY_MARGIN_MS, deadline_after

MAX_ITERATIVE_DEPTH = 64
logger = logging.getLogger(__name__)
Evaluator = Callable[[chess.Board], int]


@dataclass(frozen=True)
class DepthSearchStats:
    depth: int
    move: str
    score: int
    nodes: int
    time_ms: float


@dataclass(frozen=True)
class IterativeSearchResult:
    move: chess.Move | None
    score: int
    completed_depth: int
    nodes: int
    time_ms: float
    timed_out: bool
    depths_completed: tuple[DepthSearchStats, ...]


def iterative_search(
    board: chess.Board,
    time_budget_ms: float,
    max_depth: int = MAX_ITERATIVE_DEPTH,
    evaluator: Evaluator = evaluate,
    safety_margin_ms: float = TIME_SAFETY_MARGIN_MS,
) -> IterativeSearchResult:
    """Return the deepest fully completed Alpha-Beta result before deadline.

    The first legal move is retained as a fallback if the budget is too small
    to finish depth one. ``time_ms`` in each depth record is cumulative from
    the start of this move search; ``nodes`` is the count for that depth alone.
    """
    if time_budget_ms <= 0:
        raise ValueError("time_budget_ms must be greater than zero")
    if not 1 <= max_depth <= MAX_ITERATIVE_DEPTH:
        raise ValueError(f"max_depth must be between 1 and {MAX_ITERATIVE_DEPTH}")
    if safety_margin_ms < 0:
        raise ValueError("safety_margin_ms cannot be negative")

    started_at = monotonic()
    deadline = deadline_after(started_at, time_budget_ms, safety_margin_ms)
    if board.is_game_over(claim_draw=True):
        return IterativeSearchResult(None, 0, 0, 0, _elapsed_ms(started_at), False, ())

    fallback_move = next(iter(board.legal_moves), None)
    if fallback_move is None:
        return IterativeSearchResult(None, 0, 0, 0, _elapsed_ms(started_at), False, ())

    best_move = fallback_move
    best_score = 0
    completed_depth = 0
    total_nodes = 0
    timed_out = False
    completed: list[DepthSearchStats] = []

    logger.debug(
        "Timed Alpha-Beta search started",
        extra={"event": "time_search_started", "budget_ms": time_budget_ms, "max_depth": max_depth},
    )

    for depth in range(1, max_depth + 1):
        if monotonic() >= deadline:
            timed_out = True
            break
        try:
            result = alpha_beta(board, depth, evaluator=evaluator, deadline=deadline)
        except SearchTimeout as timeout:
            total_nodes += timeout.nodes
            timed_out = True
            logger.debug(
                "Timed Alpha-Beta search interrupted",
                extra={"event": "time_search_timeout", "depth": depth, "completed_depth": completed_depth},
            )
            break

        total_nodes += result.nodes
        if result.move is None:
            break
        best_move = result.move
        best_score = result.score
        completed_depth = depth
        stats = DepthSearchStats(
            depth=depth,
            move=result.move.uci(),
            score=result.score,
            nodes=result.nodes,
            time_ms=_elapsed_ms(started_at),
        )
        completed.append(stats)
        logger.debug(
            "Timed Alpha-Beta depth completed",
            extra={
                "event": "time_search_depth_completed",
                "depth": stats.depth,
                "move": stats.move,
                "score": stats.score,
                "nodes": stats.nodes,
                "elapsed_ms": stats.time_ms,
            },
        )
        if depth < max_depth and monotonic() >= deadline:
            timed_out = True
            break

    result = IterativeSearchResult(
        move=best_move,
        score=best_score,
        completed_depth=completed_depth,
        nodes=total_nodes,
        time_ms=_elapsed_ms(started_at),
        timed_out=timed_out,
        depths_completed=tuple(completed),
    )
    logger.debug(
        "Timed Alpha-Beta search finished",
        extra={
            "event": "time_search_finished",
            "move": result.move.uci() if result.move else None,
            "completed_depth": result.completed_depth,
            "nodes": result.nodes,
            "elapsed_ms": result.time_ms,
            "timed_out": result.timed_out,
        },
    )
    return result


def _elapsed_ms(started_at: float) -> float:
    return max(0.0, (monotonic() - started_at) * 1_000)
