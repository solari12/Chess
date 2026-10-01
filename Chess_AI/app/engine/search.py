"""Common search orchestration for the API."""

import chess

from app.engine.alpha_beta import alpha_beta
from app.engine.minimax import minimax
from app.engine.iterative_search import iterative_search
from app.schemas.chess import AIRequest, AIResponse, TimedAIRequest, TimedAIResponse


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


def search_with_time_budget(request: TimedAIRequest) -> TimedAIResponse:
    """Run iterative deepening and preserve only completed depth results."""
    try:
        board = chess.Board(request.fen)
    except ValueError as error:
        raise ValueError(f"Invalid FEN: {error}") from error

    if not board.is_valid():
        raise ValueError("FEN does not describe a valid chess position")

    result = iterative_search(
        board,
        time_budget_ms=request.time_budget_ms,
        max_depth=request.max_depth,
    )
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
