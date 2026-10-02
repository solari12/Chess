import unittest
from unittest.mock import patch

import chess
from pydantic import ValidationError

from app.api.routes import get_timed_ai_move
from app.engine.alpha_beta import SearchResult, SearchTimeout, alpha_beta
from app.engine.iterative_search import iterative_search
from app.schemas.chess import TimedAIRequest
from app.engine.time_manager import deadline_after


class IterativeSearchTests(unittest.TestCase):
    def test_completes_multiple_depths_when_budget_allows(self) -> None:
        result = iterative_search(chess.Board(), time_budget_ms=1_000, max_depth=3)

        self.assertIsNotNone(result.move)
        self.assertIn(result.move, chess.Board().legal_moves)
        self.assertGreaterEqual(result.completed_depth, 1)
        self.assertEqual(result.completed_depth, len(result.depths_completed))

    def test_tiny_budget_returns_a_legal_fallback_without_escaping_timeout(self) -> None:
        board = chess.Board()
        result = iterative_search(board, time_budget_ms=0.000000001, max_depth=64)

        self.assertIsNotNone(result.move)
        self.assertIn(result.move, board.legal_moves)
        self.assertEqual(result.completed_depth, 0)
        self.assertTrue(result.timed_out)
        self.assertEqual(result.depths_completed, ())

    def test_timed_out_depth_is_not_recorded_or_returned(self) -> None:
        first_move = chess.Move.from_uci("e2e4")
        completed_depth = SearchResult(first_move, 17, 21, 1.0)
        with patch(
            "app.engine.iterative_search.alpha_beta",
            side_effect=[completed_depth, SearchTimeout(nodes=13)],
        ):
            result = iterative_search(chess.Board(), time_budget_ms=1_000, max_depth=4)

        self.assertEqual(result.move, first_move)
        self.assertEqual(result.score, 17)
        self.assertEqual(result.completed_depth, 1)
        self.assertEqual(result.nodes, 34)
        self.assertTrue(result.timed_out)
        self.assertEqual([entry.depth for entry in result.depths_completed], [1])

    def test_per_depth_statistics_include_all_required_fields(self) -> None:
        result = iterative_search(chess.Board(), time_budget_ms=1_000, max_depth=2)

        self.assertEqual([entry.depth for entry in result.depths_completed], [1, 2])
        for entry in result.depths_completed:
            self.assertTrue(entry.move)
            self.assertIsInstance(entry.score, int)
            self.assertGreater(entry.nodes, 0)
            self.assertGreaterEqual(entry.time_ms, 0)
        self.assertEqual(result.nodes, sum(entry.nodes for entry in result.depths_completed))

    def test_max_depth_is_a_hard_ceiling(self) -> None:
        move = chess.Move.from_uci("e2e4")
        completed = SearchResult(move, 0, 10, 0.1)
        with patch("app.engine.iterative_search.alpha_beta", return_value=completed) as search_mock:
            result = iterative_search(chess.Board(), time_budget_ms=60_000, max_depth=2)

        self.assertEqual(result.completed_depth, 2)
        self.assertEqual(search_mock.call_count, 2)
        self.assertFalse(result.timed_out)

    def test_custom_evaluator_is_passed_to_alpha_beta(self) -> None:
        seen_positions = []

        def custom_evaluator(board: chess.Board) -> int:
            seen_positions.append(board.fen())
            return len(board.piece_map())

        result = iterative_search(
            chess.Board(), time_budget_ms=1_000, max_depth=1, evaluator=custom_evaluator
        )

        self.assertEqual(result.completed_depth, 1)
        self.assertTrue(seen_positions)

    def test_repetition_penalty_is_forwarded_to_alpha_beta(self) -> None:
        completed = SearchResult(chess.Move.from_uci("e2e4"), 0, 10, 0.1)
        with patch("app.engine.iterative_search.alpha_beta", return_value=completed) as search_mock:
            result = iterative_search(
                chess.Board(),
                time_budget_ms=1_000,
                max_depth=1,
                repetition_penalty=12,
            )

        self.assertEqual(result.completed_depth, 1)
        self.assertEqual(search_mock.call_args.kwargs["repetition_penalty"], 12)

    def test_terminal_position_returns_no_move_without_search(self) -> None:
        checkmate = chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")
        result = iterative_search(checkmate, time_budget_ms=2_000, max_depth=64)

        self.assertIsNone(result.move)
        self.assertEqual(result.completed_depth, 0)
        self.assertEqual(result.depths_completed, ())
        self.assertFalse(result.timed_out)

    def test_stalemate_is_terminal_without_iterative_search(self) -> None:
        stalemate = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
        result = iterative_search(stalemate, time_budget_ms=2_000, max_depth=64)

        self.assertTrue(stalemate.is_stalemate())
        self.assertIsNone(result.move)
        self.assertEqual(result.completed_depth, 0)

    def test_deadline_keeps_an_explicit_safety_margin(self) -> None:
        self.assertAlmostEqual(deadline_after(10.0, 2_000), 11.95)
        self.assertAlmostEqual(deadline_after(10.0, 1.0), 10.0009)

    def test_alpha_beta_deadline_raises_internal_timeout(self) -> None:
        with patch("app.engine.alpha_beta.monotonic", return_value=2.0):
            with self.assertRaises(SearchTimeout):
                alpha_beta(chess.Board(), depth=4, deadline=1.0)


class TimedSearchAPITests(unittest.TestCase):
    def test_timed_endpoint_returns_a_legal_fallback_and_depth_metadata(self) -> None:
        request = TimedAIRequest(
            fen=chess.STARTING_FEN,
            algorithm="alpha-beta",
            time_budget_ms=1,
            max_depth=64,
        )
        response = get_timed_ai_move(request)

        self.assertIsNotNone(response.move)
        self.assertIn(chess.Move.from_uci(response.move), chess.Board().legal_moves)
        self.assertEqual(response.depth, response.completed_depth)
        self.assertIsInstance(response.depths_completed, list)

    def test_time_request_rejects_ambiguous_fixed_depth_field(self) -> None:
        with self.assertRaises(ValidationError):
            TimedAIRequest(
                fen=chess.STARTING_FEN,
                algorithm="alpha-beta",
                time_budget_ms=100,
                depth=3,
            )
