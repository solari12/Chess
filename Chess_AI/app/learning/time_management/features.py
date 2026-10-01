"""Simple, interpretable chess-position features for dataset records."""

import chess

from app.engine.evaluation import PIECE_VALUES


FEATURE_PIECE_VALUES = {
    chess.PAWN: PIECE_VALUES[chess.PAWN],
    chess.KNIGHT: PIECE_VALUES[chess.KNIGHT],
    chess.BISHOP: PIECE_VALUES[chess.BISHOP],
    chess.ROOK: PIECE_VALUES[chess.ROOK],
    chess.QUEEN: PIECE_VALUES[chess.QUEEN],
}


def extract_position_features(board: chess.Board, remaining_time_ms: int) -> dict[str, int | float | str | bool]:
    """Extract position state and a synthetic clock value without evaluation labels."""
    if remaining_time_ms < 0:
        raise ValueError("remaining_time_ms cannot be negative")

    legal_moves = list(board.legal_moves)
    capture_count = sum(board.is_capture(move) for move in legal_moves)
    checking_move_count = sum(board.gives_check(move) for move in legal_moves)

    piece_counts: dict[str, int] = {}
    material: dict[chess.Color, int] = {}
    for color, color_name in ((chess.WHITE, "white"), (chess.BLACK, "black")):
        material[color] = sum(
            len(board.pieces(piece_type, color)) * value
            for piece_type, value in FEATURE_PIECE_VALUES.items()
        )
        for piece_type, piece_name in (
            (chess.PAWN, "pawn"),
            (chess.KNIGHT, "knight"),
            (chess.BISHOP, "bishop"),
            (chess.ROOK, "rook"),
            (chess.QUEEN, "queen"),
        ):
            piece_counts[f"{piece_name}_count_{color_name}"] = len(board.pieces(piece_type, color))

    king_square = board.king(board.turn)
    king_safety = 0
    if king_square is not None:
        king_safety = sum(
            board.is_attacked_by(not board.turn, square)
            for square in chess.SquareSet(chess.BB_KING_ATTACKS[king_square])
        )

    return {
        "remaining_time_ms": remaining_time_ms,
        "fullmove_number": board.fullmove_number,
        "halfmove_clock": board.halfmove_clock,
        "side_to_move": "white" if board.turn == chess.WHITE else "black",
        "legal_move_count": len(legal_moves),
        "capture_count": capture_count,
        "checking_move_count": checking_move_count,
        "side_in_check": board.is_check(),
        "king_safety_basic_indicator": king_safety,
        "material_white": material[chess.WHITE],
        "material_black": material[chess.BLACK],
        "material_imbalance": material[chess.WHITE] - material[chess.BLACK],
        **piece_counts,
    }
