"""Common search orchestration for the API."""

import logging

import chess

from app.engine.alpha_beta import alpha_beta
from app.engine.minimax import minimax
from app.engine.iterative_search import iterative_search
from app.learning.time_management.runtime_shadow import evaluate_and_log_shadow
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
