"""Data structures used by the Phase 2A dataset generator."""

from collections.abc import Callable
from dataclasses import dataclass

import chess


@dataclass(frozen=True)
class PositionSample:
    """A reproducibly identified board position sampled from a source game."""

    fen: str
    source: str
    game_index: int | None
    ply: int


@dataclass(frozen=True)
class DatasetSummary:
    games: int
    sampled_positions: int
    clock_variants: int
    dataset_records: int
    average_legal_moves: float
    average_completed_depth: float
    average_search_time_ms: float
    timeout_rate: float


Evaluator = Callable[[chess.Board], int]


def board_from_sample(sample: PositionSample) -> chess.Board:
    """Reconstruct and validate the board represented by a sample."""
    board = chess.Board(sample.fen)
    if not board.is_valid():
        raise ValueError(f"Sample contains an invalid FEN: {sample.fen}")
    return board
