"""Material-based position evaluation from White's point of view."""

import chess


PIECE_VALUES: dict[int, int] = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20_000,
}


def evaluate(board: chess.Board) -> int:
    """Return material balance in centipawns; positive favors White.

    Terminal outcomes are handled by the search so checkmate can be scored
    according to distance. This function evaluates material only.
    """
    score = 0
    for piece_type, value in PIECE_VALUES.items():
        score += len(board.pieces(piece_type, chess.WHITE)) * value
        score -= len(board.pieces(piece_type, chess.BLACK)) * value
    return score
