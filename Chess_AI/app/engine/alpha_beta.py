"""Minimax search with alpha-beta pruning and no move-ordering heuristics."""

from dataclasses import dataclass
from time import perf_counter
from typing import Callable

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
    """Match Minimax checkmate-distance and draw scoring exactly."""
    if board.is_checkmate():
        return -MATE_SCORE + ply if board.turn == chess.WHITE else MATE_SCORE - ply
    if board.is_game_over(claim_draw=True):
        return 0
    return None


def alpha_beta(
    board: chess.Board,
    depth: int,
    evaluator: Callable[[chess.Board], int] = evaluate,
) -> SearchResult:
    """Find a move using Minimax with alpha-beta pruning to ``depth`` plies.

    Alpha is the best score White (the maximizing player) can already
    guarantee. Beta is the best score Black (the minimizing player) can
    already guarantee. If alpha reaches or passes beta, the current player
    has found a result that the opponent would already avoid, so exploring
    the remaining siblings cannot change the root's minimax choice.

    Pruning only removes branches that cannot affect the minimax result. The
    algorithm therefore returns the same score and first best move as pure
    Minimax while often visiting fewer nodes. Moves retain python-chess's
    natural legal-move order; no move ordering is applied here.
    """
    if depth < 1:
        raise ValueError("depth must be at least 1")

    started_at = perf_counter()
    nodes = 0

    def visit(remaining_depth: int, ply: int, alpha: float, beta: float) -> int:
        nonlocal nodes
        nodes += 1

        terminal = _terminal_score(board, ply)
        if terminal is not None:
            return terminal
        if remaining_depth == 0:
            return evaluator(board)

        if board.turn == chess.WHITE:
            best_score = -MATE_SCORE * 2
            for move in board.legal_moves:
                board.push(move)
                try:
                    child_score = visit(remaining_depth - 1, ply + 1, alpha, beta)
                    best_score = max(best_score, child_score)
                finally:
                    board.pop()

                alpha = max(alpha, best_score)
                if alpha >= beta:
                    break
            return best_score

        best_score = MATE_SCORE * 2
        for move in board.legal_moves:
            board.push(move)
            try:
                child_score = visit(remaining_depth - 1, ply + 1, alpha, beta)
                best_score = min(best_score, child_score)
            finally:
                board.pop()

            beta = min(beta, best_score)
            if alpha >= beta:
                break
        return best_score

    best_move: chess.Move | None = None
    root_terminal = _terminal_score(board, 0)
    if root_terminal is not None:
        nodes = 1
        score = root_terminal
    else:
        nodes = 1  # Count the root position, as in pure Minimax.
        alpha = float("-inf")
        beta = float("inf")
        best_score = -MATE_SCORE * 2 if board.turn == chess.WHITE else MATE_SCORE * 2

        candidates = list(board.legal_moves)
        for move in candidates:
            board.push(move)
            try:
                score = visit(depth - 1, 1, alpha, beta)
            finally:
                board.pop()

            if best_move is None or (
                score > best_score if board.turn == chess.WHITE else score < best_score
            ):
                best_move = move
                best_score = score

            if board.turn == chess.WHITE:
                alpha = max(alpha, best_score)
            else:
                beta = min(beta, best_score)
            if alpha >= beta:
                break
        score = best_score

    elapsed_ms = (perf_counter() - started_at) * 1000
    return SearchResult(best_move, score, nodes, elapsed_ms)
