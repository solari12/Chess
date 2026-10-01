"""Pure minimax search without alpha-beta pruning."""

from dataclasses import dataclass
from time import perf_counter

import chess

from app.engine.evaluation import evaluate

MATE_SCORE = 1_000_000


@dataclass(frozen=True)
class SearchResult:
    move: chess.Move | None
    score: int
    nodes: int
    time_ms: float


def _terminal_score(board: chess.Board, ply: int) -> int | None:
    """Return a score for a finished position, or None if it is ongoing."""
    if board.is_checkmate():
        # The side to move has been mated. Prefer faster wins and slower losses.
        return -MATE_SCORE + ply if board.turn == chess.WHITE else MATE_SCORE - ply
    if board.is_game_over(claim_draw=True):
        return 0
    return None


def minimax(board: chess.Board, depth: int) -> SearchResult:
    """Find a move using exhaustive minimax to ``depth`` plies.

    White maximizes the White-positive material score and Black minimizes it.
    Every visited position is counted, including the root and terminal leaves.
    """
    if depth < 1:
        raise ValueError("depth must be at least 1")

    started_at = perf_counter()
    nodes = 0

    def visit(remaining_depth: int, ply: int) -> int:
        nonlocal nodes
        nodes += 1

        terminal = _terminal_score(board, ply)
        if terminal is not None:
            return terminal
        if remaining_depth == 0:
            return evaluate(board)

        if board.turn == chess.WHITE:
            best_score = -MATE_SCORE * 2
            for move in board.legal_moves:
                board.push(move)
                try:
                    best_score = max(best_score, visit(remaining_depth - 1, ply + 1))
                finally:
                    board.pop()
            return best_score

        best_score = MATE_SCORE * 2
        for move in board.legal_moves:
            board.push(move)
            try:
                best_score = min(best_score, visit(remaining_depth - 1, ply + 1))
            finally:
                board.pop()
        return best_score

    best_move: chess.Move | None = None
    root_terminal = _terminal_score(board, 0)
    if root_terminal is not None:
        nodes = 1
        score = root_terminal
    else:
        nodes = 1  # Count the root position.
        candidates = list(board.legal_moves)
        best_score = -MATE_SCORE * 2 if board.turn == chess.WHITE else MATE_SCORE * 2
        for move in candidates:
            board.push(move)
            try:
                score = visit(depth - 1, 1)
            finally:
                board.pop()

            if best_move is None or (
                score > best_score if board.turn == chess.WHITE else score < best_score
            ):
                best_move = move
                best_score = score
        score = best_score

    elapsed_ms = (perf_counter() - started_at) * 1000
    return SearchResult(best_move, score, nodes, elapsed_ms)
