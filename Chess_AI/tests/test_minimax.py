import chess
import unittest

from app.api.routes import get_ai_move
from app.engine.evaluation import evaluate
from app.engine.minimax import MATE_SCORE, minimax
from app.schemas.chess import AIRequest


class EvaluationTests(unittest.TestCase):
    def test_material_scores_use_white_positive_convention(self) -> None:
        white_material = chess.Board("7k/8/8/8/8/8/P7/K7 w - - 0 1")
        black_material = chess.Board("7k/8/8/8/8/8/p7/K7 w - - 0 1")

        self.assertEqual(evaluate(white_material), 100)
        self.assertEqual(evaluate(black_material), -100)

    def test_all_piece_values_are_included(self) -> None:
        board = chess.Board("7k/8/8/8/8/8/8/KQRBNP2 w - - 0 1")
        self.assertEqual(evaluate(board), 2_150)


class MinimaxTests(unittest.TestCase):
    def test_returns_legal_move_and_counts_nodes(self) -> None:
        board = chess.Board()

        result = minimax(board, depth=2)

        self.assertIsNotNone(result.move)
        self.assertIn(result.move, board.legal_moves)
        self.assertGreater(result.nodes, 1)
        self.assertGreaterEqual(result.time_ms, 0)
        self.assertEqual(board.fen(), chess.STARTING_FEN)

    def test_checkmate_is_a_terminal_win(self) -> None:
        board = chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")

        result = minimax(board, depth=3)

        self.assertTrue(board.is_checkmate())
        self.assertIsNone(result.move)
        self.assertEqual(result.score, MATE_SCORE)
        self.assertEqual(result.nodes, 1)

    def test_stalemate_is_a_terminal_draw(self) -> None:
        board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")

        result = minimax(board, depth=3)

        self.assertTrue(board.is_stalemate())
        self.assertIsNone(result.move)
        self.assertEqual(result.score, 0)
        self.assertEqual(result.nodes, 1)


class ApiTests(unittest.TestCase):
    def test_minimax_endpoint_returns_legal_uci_move(self) -> None:
        request = AIRequest(
            fen=chess.STARTING_FEN,
            algorithm="minimax",
            depth=2,
        )

        response = get_ai_move(request)
        board = chess.Board(request.fen)

        move = chess.Move.from_uci(response.move)
        self.assertIn(move, board.legal_moves)
        self.assertEqual(response.algorithm, "minimax")
        self.assertEqual(response.depth, 2)
        self.assertGreater(response.nodes, 1)


if __name__ == "__main__":
    unittest.main()
