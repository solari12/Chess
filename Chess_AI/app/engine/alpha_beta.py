"""Minimax search with alpha-beta pruning and no move-ordering heuristics."""

from dataclasses import dataclass
from time import monotonic, perf_counter
from typing import Callable

import chess

from app.engine.evaluation import evaluate

MATE_SCORE = 1_000_000


class SearchTimeout(Exception):
    """Internal signal used to interrupt a search after its deadline."""

    def __init__(self, nodes: int):
        super().__init__("Alpha-Beta search deadline reached")
        self.nodes = nodes


@dataclass(frozen=True)
class SearchResult:
    move: chess.Move | None
    score: int
    nodes: int
    time_ms: float


def _terminal_score(
    board: chess.Board,
    ply: int,
    repetition_penalty: int = 0,
) -> int | None:
    """Score terminal positions and optionally discourage repeated positions."""
    if board.is_checkmate():
        return -MATE_SCORE + ply if board.turn == chess.WHITE else MATE_SCORE - ply
    if board.is_game_over(claim_draw=repetition_penalty == 0):
        return 0
    if repetition_penalty and board.is_repetition(2):
        # Give the side that would repeat the position a small reason to choose
        # a different line, while still preferring a draw to a material loss.
        return -repetition_penalty if board.turn == chess.WHITE else repetition_penalty
    return None


def alpha_beta(
    board: chess.Board,
    depth: int,
    evaluator: Callable[[chess.Board], int] = evaluate,
    repetition_penalty: int = 0,
    deadline: float | None = None,
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

    def check_deadline() -> None:
        if deadline is not None and monotonic() >= deadline:
            raise SearchTimeout(nodes)

    def visit(remaining_depth: int, ply: int, alpha: float, beta: float) -> int:
        nonlocal nodes
        check_deadline()
        nodes += 1

        terminal = _terminal_score(board, ply, repetition_penalty)
        if terminal is not None:
            return terminal
        if remaining_depth == 0:
            score = evaluator(board)
            check_deadline()
            return score

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
    root_terminal = _terminal_score(board, 0, repetition_penalty)
    if (
        repetition_penalty
        and board.is_repetition(2)
        and not board.is_game_over(claim_draw=False)
    ):
        # Repetition is a search penalty, not a real terminal state: the root
        # still needs to choose a move that can escape the cycle.
        root_terminal = None
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
            check_deadline()
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
        check_deadline()

    elapsed_ms = (perf_counter() - started_at) * 1000
    return SearchResult(best_move, score, nodes, elapsed_ms)
