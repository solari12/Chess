"""Common search orchestration for the API."""

import logging
import os

import chess

from app.engine.alpha_beta import alpha_beta
from app.engine.minimax import minimax
from app.engine.iterative_search import IterativeSearchResult, iterative_search
from app.learning.time_management.runtime_shadow import evaluate_and_log_shadow
from app.learning.time_management.runtime_policy import (
    BASELINE_BUDGET_MS,
    SAFETY_MARGIN_MS,
    decide_search_budget,
)
from app.schemas.chess import AIRequest, AIResponse, TimedAIRequest, TimedAIResponse

logger = logging.getLogger(__name__)


def search(request: AIRequest) -> AIResponse:
    """Parse the input position and dispatch to the requested search."""
    try:
        board = chess.Board(request.fen)
    except ValueError as error:
        raise ValueError(f"Invalid FEN: {error}") from error

    if not board.is_valid():
        raise ValueError("FEN does not describe a valid chess position")
    if board.is_game_over(claim_draw=True):
        raise ValueError("The position is terminal and has no legal move")

    # The feature flag is opt-in. Legacy callers without a clock retain the
    # fixed-depth behavior and do not need to change their request payload.
    if (
        _time_management_enabled()
        and request.algorithm in {"alpha-beta", "alpha_beta"}
        and request.remaining_time_ms is not None
    ):
        decision = decide_search_budget(
            board,
            remaining_time_ms=request.remaining_time_ms,
            max_depth=request.depth,
        )
        result = None
        actual_search_time_ms = 0.0
        if decision.final_search_budget_ms > 0:
            result = iterative_search(
                board,
                time_budget_ms=decision.final_search_budget_ms,
                max_depth=request.depth,
            )
            actual_search_time_ms = result.time_ms
        else:
            # At/below the reserve there is no legal search budget. Return an
            # immediate legal move so the existing response contract survives.
            move = next(iter(board.legal_moves), None)
            if move is None:
                raise ValueError("The position is terminal and has no legal move")

        fields = {
            "time_management_enabled": True,
            "predicted_time_ms": decision.predicted_time_ms,
            "final_search_budget_ms": decision.final_search_budget_ms,
            "baseline_budget_ms": BASELINE_BUDGET_MS,
            "safety_reserve_ms": SAFETY_MARGIN_MS,
            "safety_cap_applied": decision.safety_cap_applied,
            "fallback_used": decision.fallback_used,
            "fallback_reason": decision.fallback_reason,
            "probe_time_ms": decision.probe_time_ms,
            "model_load_time_ms": getattr(decision, "model_load_time_ms", 0.0),
            "feature_extraction_time_ms": getattr(decision, "feature_extraction_time_ms", 0.0),
            "inference_latency_ms": decision.inference_latency_ms,
            "policy_latency_ms": decision.policy_latency_ms,
            "model_version": decision.model_version,
            "actual_search_time_ms": actual_search_time_ms,
            "completed_depth": result.completed_depth if result is not None else 0,
            "nodes": result.nodes if result is not None else 0,
            "timed_out": result.timed_out if result is not None else False,
        }
        logger.info("Production time-management decision", extra=fields)
        if result is None:
            return AIResponse(
                move=move.uci(),
                algorithm="alpha-beta",
                depth=request.depth,
                score=0,
                nodes=0,
                time_ms=actual_search_time_ms,
            )
        if result.move is None:
            raise ValueError("The position is terminal and has no legal move")
        return AIResponse(
            move=result.move.uci(),
            algorithm="alpha-beta",
            depth=request.depth,
            score=result.score,
            nodes=result.nodes,
            time_ms=result.time_ms,
        )

    if request.algorithm == "minimax":
        result = minimax(board, request.depth)
        response_algorithm = "minimax"
    else:
        result = alpha_beta(board, request.depth)
        response_algorithm = "alpha-beta"
    if result.move is None:
        raise ValueError("The position is terminal and has no legal move")

    return AIResponse(
        move=result.move.uci(),
        algorithm=response_algorithm,
        depth=request.depth,
        score=result.score,
        nodes=result.nodes,
        time_ms=result.time_ms,
    )


def _time_management_enabled() -> bool:
    return os.getenv("TIME_MANAGEMENT_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}


def search_with_time_budget(
    request: TimedAIRequest,
    *,
    shadow_log_path: str | None = None,
) -> TimedAIResponse:
    """Run iterative deepening and preserve only completed depth results."""
    try:
        board = chess.Board(request.fen)
    except ValueError as error:
        raise ValueError(f"Invalid FEN: {error}") from error

    if not board.is_valid():
        raise ValueError("FEN does not describe a valid chess position")

    # The Chess page currently uses this timed endpoint for alpha-beta. Keep
    # its existing explicit-budget path untouched unless opt-in policy and a
    # real remaining clock are both present.
    if _time_management_enabled() and request.remaining_time_ms is not None:
        decision = decide_search_budget(
            board,
            remaining_time_ms=request.remaining_time_ms,
            max_depth=request.max_depth,
        )
        if decision.final_search_budget_ms > 0:
            result = iterative_search(
                board,
                time_budget_ms=decision.final_search_budget_ms,
                max_depth=request.max_depth,
            )
        else:
            move = next(iter(board.legal_moves), None)
            if move is None:
                raise ValueError("The position is terminal and has no legal move")
            result = IterativeSearchResult(move, 0, 0, 0, 0.0, False, ())
        logger.info(
            "Production time-management decision",
            extra={
                "time_management_enabled": True,
                "predicted_time_ms": decision.predicted_time_ms,
                "final_search_budget_ms": decision.final_search_budget_ms,
                "baseline_budget_ms": BASELINE_BUDGET_MS,
                "safety_reserve_ms": SAFETY_MARGIN_MS,
                "safety_cap_applied": decision.safety_cap_applied,
                "fallback_used": decision.fallback_used,
                "fallback_reason": decision.fallback_reason,
                "probe_time_ms": decision.probe_time_ms,
                "model_load_time_ms": getattr(decision, "model_load_time_ms", 0.0),
                "feature_extraction_time_ms": getattr(decision, "feature_extraction_time_ms", 0.0),
                "inference_latency_ms": decision.inference_latency_ms,
                "policy_latency_ms": decision.policy_latency_ms,
                "model_version": decision.model_version,
                "actual_search_time_ms": result.time_ms,
                "completed_depth": result.completed_depth,
                "nodes": result.nodes,
                "timed_out": result.timed_out,
            },
        )
        return _timed_response(result)

    # Capture the exact input state before search; shadow inference is strictly
    # post-search and is never passed into the engine's budget or move logic.
    search_fen = board.fen()
    result = iterative_search(
        board,
        time_budget_ms=request.time_budget_ms,
        max_depth=request.max_depth,
    )
    try:
        shadow_board = chess.Board(search_fen)
        shadow_kwargs = {} if shadow_log_path is None else {"log_path": shadow_log_path}
        evaluate_and_log_shadow(
            shadow_board,
            remaining_time_ms=(
                request.remaining_time_ms
                if request.remaining_time_ms is not None
                else request.time_budget_ms
            ),
            result=result,
            actual_search_budget_ms=request.time_budget_ms,
            max_depth=request.max_depth,
            **shadow_kwargs,
        )
    except Exception:
        # Shadow failures are observable in logs but cannot fail or alter the
        # already completed engine search response.
        logger.exception("Runtime shadow evaluation failed after completed search")
    return _timed_response(result)


def _timed_response(result: IterativeSearchResult) -> TimedAIResponse:
    return TimedAIResponse(
        move=result.move.uci() if result.move else None,
        algorithm="alpha-beta",
        depth=result.completed_depth,
        score=result.score,
        completed_depth=result.completed_depth,
        nodes=result.nodes,
        time_ms=result.time_ms,
        timed_out=result.timed_out,
        depths_completed=[
            {
                "depth": depth.depth,
                "move": depth.move,
                "score": depth.score,
                "nodes": depth.nodes,
                "time_ms": depth.time_ms,
            }
            for depth in result.depths_completed
        ],
    )
