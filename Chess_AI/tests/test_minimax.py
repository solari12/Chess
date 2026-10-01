import unittest

import chess

from app.api.routes import get_ai_move
from app.engine.alpha_beta import MATE_SCORE as ALPHA_BETA_MATE_SCORE, alpha_beta
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

    def test_rejects_non_positive_depth(self) -> None:
        with self.assertRaises(ValueError):
            minimax(chess.Board(), 0)

    def test_checkmate_is_a_terminal_win(self) -> None:
        board = chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")
        original_fen = board.fen()

        result = minimax(board, depth=3)

        self.assertTrue(board.is_checkmate())
        self.assertIsNone(result.move)
        self.assertEqual(result.score, MATE_SCORE)
        self.assertEqual(result.nodes, 1)
        self.assertEqual(board.fen(), original_fen)

    def test_stalemate_is_a_terminal_draw(self) -> None:
        board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
        original_fen = board.fen()

        result = minimax(board, depth=3)

        self.assertTrue(board.is_stalemate())
        self.assertIsNone(result.move)
        self.assertEqual(result.score, 0)
        self.assertEqual(result.nodes, 1)
        self.assertEqual(board.fen(), original_fen)


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

    def test_alpha_beta_endpoint_returns_legal_uci_move(self) -> None:
        request = AIRequest(
            fen=chess.STARTING_FEN,
            algorithm="alpha-beta",
            depth=2,
        )

        response = get_ai_move(request)
        board = chess.Board(request.fen)

        move = chess.Move.from_uci(response.move)
        self.assertIn(move, board.legal_moves)
        self.assertEqual(response.algorithm, "alpha-beta")
        self.assertEqual(response.depth, 2)

    def test_legacy_alpha_beta_algorithm_spelling_is_accepted(self) -> None:
        response = get_ai_move(
            AIRequest(fen=chess.STARTING_FEN, algorithm="alpha_beta", depth=1)
        )
        self.assertEqual(response.algorithm, "alpha-beta")


class AlphaBetaTests(unittest.TestCase):
    def test_matches_minimax_score_and_first_best_move_with_fewer_nodes(self) -> None:
        board = chess.Board()
        original_fen = board.fen()

        minimax_result = minimax(board, depth=2)
        alpha_beta_result = alpha_beta(board, depth=2)

        self.assertEqual(alpha_beta_result.score, minimax_result.score)
        self.assertEqual(alpha_beta_result.move, minimax_result.move)
        self.assertLessEqual(alpha_beta_result.nodes, minimax_result.nodes)
        self.assertLess(alpha_beta_result.nodes, minimax_result.nodes)
        self.assertGreaterEqual(alpha_beta_result.time_ms, 0)
        self.assertEqual(board.fen(), original_fen)

    def test_checkmate_is_scored_like_minimax(self) -> None:
        board = chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")
        original_fen = board.fen()

        expected = minimax(board, depth=3)
        result = alpha_beta(board, depth=3)

        self.assertIsNone(result.move)
        self.assertEqual(result.score, ALPHA_BETA_MATE_SCORE)
        self.assertEqual(result.score, expected.score)
        self.assertEqual(result.nodes, expected.nodes)
        self.assertEqual(board.fen(), original_fen)

    def test_stalemate_is_scored_like_minimax(self) -> None:
        board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
        original_fen = board.fen()

        expected = minimax(board, depth=3)
        result = alpha_beta(board, depth=3)

        self.assertTrue(board.is_stalemate())
        self.assertIsNone(result.move)
        self.assertEqual(result.score, expected.score)
        self.assertEqual(result.score, 0)
        self.assertEqual(result.nodes, expected.nodes)
        self.assertEqual(board.fen(), original_fen)

    def test_rejects_non_positive_depth(self) -> None:
        with self.assertRaises(ValueError):
            alpha_beta(chess.Board(), 0)


if __name__ == "__main__":
    unittest.main()
