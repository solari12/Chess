"""Offline generation and sampling of chess positions."""

from collections.abc import Callable
import random

import chess
import chess.pgn

from app.engine.evaluation import evaluate
from app.engine.alpha_beta import alpha_beta
from app.learning.time_management.models import PositionSample

Evaluator = Callable[[chess.Board], int]


def generate_positions(
    *,
    games: int = 10,
    sampling_interval: int = 4,
    game_generation_depth: int = 2,
    max_plies: int = 80,
    seed: int = 42,
    pgn_path: str | None = None,
    candidate_count: int = 3,
    evaluator: Evaluator = evaluate,
) -> list[PositionSample]:
    """Generate engine games and sample them, plus start and optional PGN positions.

    Each engine turn is selected from the best ``candidate_count`` root
    alternatives according to the existing Alpha-Beta evaluator. A seeded RNG
    breaks the choice among those strong candidates to make reproducible but
    varied games; it does not introduce a separate chess engine.
    """
    if games < 0:
        raise ValueError("games cannot be negative")
    if sampling_interval < 1:
        raise ValueError("sampling_interval must be at least 1")
    if game_generation_depth < 1:
        raise ValueError("game_generation_depth must be at least 1")
    if max_plies < 0:
        raise ValueError("max_plies cannot be negative")
    if candidate_count < 1:
        raise ValueError("candidate_count must be at least 1")

    rng = random.Random(seed)
    samples: list[PositionSample] = []
    seen_fens: set[str] = set()

    def add(board: chess.Board, source: str, game_index: int | None, ply: int) -> None:
        fen = board.fen()
        if fen not in seen_fens:
            seen_fens.add(fen)
            samples.append(PositionSample(fen, source, game_index, ply))

    add(chess.Board(), "starting_position", None, 0)

    for game_index in range(1, games + 1):
        board = chess.Board()
        for ply in range(1, max_plies + 1):
            if board.is_game_over(claim_draw=True):
                break
            move = _choose_engine_move(board, game_generation_depth, candidate_count, rng, evaluator)
            if move is None:
                break
            board.push(move)
            if ply % sampling_interval == 0:
                add(board, "engine_game", game_index, ply)

    if pgn_path:
        with open(pgn_path, "r", encoding="utf-8") as pgn_file:
            pgn_game_index = 0
            while game := chess.pgn.read_game(pgn_file):
                pgn_game_index += 1
                if game.errors:
                    raise ValueError(f"PGN game {pgn_game_index} contains errors: {game.errors[0]}")
                board = game.board()
                add(board, "pgn", pgn_game_index, 0)
                for ply, move in enumerate(game.mainline_moves(), start=1):
                    if move not in board.legal_moves:
                        raise ValueError(f"Illegal move in PGN game {pgn_game_index} at ply {ply}")
                    board.push(move)
                    if ply % sampling_interval == 0:
                        add(board, "pgn", pgn_game_index, ply)

    return samples


def _choose_engine_move(
    board: chess.Board,
    depth: int,
    candidate_count: int,
    rng: random.Random,
    evaluator: Evaluator,
) -> chess.Move | None:
    """Score legal root alternatives using the existing search/evaluator."""
    scored: list[tuple[int, chess.Move]] = []
    for move in board.legal_moves:
        board.push(move)
        try:
            if depth == 1:
                score = evaluator(board)
            else:
                score = alpha_beta(board, depth - 1, evaluator=evaluator).score
        finally:
            board.pop()
        scored.append((score, move))

    if not scored:
        return None
    scored.sort(key=lambda item: item[0], reverse=board.turn == chess.WHITE)
    return rng.choice(scored[:candidate_count])[1]
