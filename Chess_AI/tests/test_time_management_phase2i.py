from __future__ import annotations

import unittest
from unittest.mock import patch

import chess

from app.engine.iterative_search import IterativeSearchResult
from app.learning.time_management.phase2i import run_case, select_records, summarize
from app.learning.time_management.runtime_policy import TimeManagementDecision


class Phase2IPairedComparisonTests(unittest.TestCase):
    def test_run_case_pairs_same_position_and_reports_search_metrics(self) -> None:
        board = chess.Board()
        baseline_move = chess.Move.from_uci("e2e4")
        model_move = chess.Move.from_uci("d2d4")
        self.assertIn(baseline_move, board.legal_moves)
        self.assertIn(model_move, board.legal_moves)
        search_results = iter((
            IterativeSearchResult(baseline_move, 15, 2, 100, 250.0, True, ()),
            IterativeSearchResult(model_move, 20, 3, 140, 180.0, True, ()),
        ))
        decision = TimeManagementDecision(
            predicted_time_ms=200.0,
            final_search_budget_ms=200.0,
            safety_cap_applied=False,
            fallback_used=False,
            fallback_reason=None,
            probe_time_ms=25.0,
            inference_latency_ms=2.0,
            policy_latency_ms=30.0,
            model_version="test-model",
            model_load_time_ms=0.0,
            feature_extraction_time_ms=1.0,
        )
        record = {
            "sample_id": "heldout-1",
            "fen": board.fen(),
            "clock": {"remaining_time_ms": 30_000},
        }
        with (
            patch("app.learning.time_management.phase2i.iterative_search", side_effect=lambda *_args, **_kwargs: next(search_results)),
            patch("app.learning.time_management.phase2i.decide_search_budget", return_value=decision) as decide,
        ):
            result = run_case(record, model=object(), model_version="test-model", max_depth=6)

        decide.assert_called_once()
        self.assertEqual(decide.call_args.kwargs["remaining_time_ms"], 30_000)
        self.assertEqual(result["fen"], record["fen"])
        self.assertEqual(result["remaining_time_ms"], 30_000)
        self.assertEqual(result["baseline_move"], "e2e4")
        self.assertEqual(result["model_move"], "d2d4")
        self.assertEqual(result["baseline_time_ms"], 250.0)
        self.assertEqual(result["model_total_time_ms"], 210.0)
        self.assertEqual(result["model_search_time_ms"], 180.0)
        self.assertEqual(result["model_predicted_time_ms"], 200.0)
        self.assertEqual(result["model_load_time_ms"], 0.0)
        self.assertFalse(result["baseline_safety_violation"])
        self.assertFalse(result["model_safety_violation"])
        self.assertFalse(result["model_negative_budget"])
        self.assertIsNotNone(result["evaluation_difference_model_minus_baseline_cp"])
        self.assertEqual(board.fen(), record["fen"], "paired searches must not mutate the source position")

    def test_select_records_is_deterministic_and_spans_dataset(self) -> None:
        records = [{"sample_id": str(index)} for index in range(9)]
        selected = select_records(records, 4)
        self.assertEqual([row["sample_id"] for row in selected], ["0", "3", "5", "8"])
        self.assertEqual(select_records(records, 0), records)

    def test_summary_reports_descriptive_metrics_and_safety(self) -> None:
        rows = [
            {
                "fen": "fen-a", "baseline_time_ms": 250, "baseline_completed_depth": 2,
                "baseline_nodes": 100, "model_predicted_time_ms": 200,
                "model_final_search_budget_ms": 190, "model_search_time_ms": 180,
                "model_total_time_ms": 190, "model_completed_depth": 3, "model_nodes": 140,
                "model_load_time_ms": 0, "feature_extraction_time_ms": 1,
                "inference_latency_ms": 2, "model_fallback_used": False,
                "baseline_safety_violation": False, "model_safety_violation": False,
                "model_negative_budget": False, "evaluation_difference_model_minus_baseline_cp": 5,
                "model_move": "d2d4", "baseline_move": "e2e4",
            },
        ]
        report = summarize(rows)
        self.assertFalse(report["playing_strength_improvement_established"])
        self.assertEqual(report["test_cases"], 1)
        self.assertEqual(report["paired_differences"]["different_move_cases"], 1)
        self.assertEqual(report["safety"]["model_budget_violations"], 0)


if __name__ == "__main__":
    unittest.main()
