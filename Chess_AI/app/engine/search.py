"""Common search orchestration for the API."""

import chess

from app.engine.minimax import minimax
from app.schemas.chess import AIRequest, AIResponse


def search(request: AIRequest) -> AIResponse:
    """Parse the input position and dispatch to the requested search."""
    try:
        board = chess.Board(request.fen)
    except ValueError as error:
        raise ValueError(f"Invalid FEN: {error}") from error

    if not board.is_valid():
        raise ValueError("FEN does not describe a valid chess position")

    if request.algorithm == "alpha_beta":
        raise NotImplementedError("Alpha-Beta search has not been implemented")

    result = minimax(board, request.depth)
    if result.move is None:
        raise ValueError("The position is terminal and has no legal move")

    return AIResponse(
        move=result.move.uci(),
        algorithm=request.algorithm,
        depth=request.depth,
        score=result.score,
        nodes=result.nodes,
        time_ms=result.time_ms,
    )
