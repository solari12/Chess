"""Tests for the isolated Phase 2G paired runtime experiment."""

import unittest
from unittest.mock import patch

import chess
import numpy as np

from app.engine.iterative_search import DepthSearchStats, IterativeSearchResult
from app.learning.time_management.model_features import FEATURE_NAMES
from app.learning.time_management.phase2g import (
    report_results,
    run_case,
    safe_budget,
)


class TestModel:
    n_features_in_ = len(FEATURE_NAMES)

    def __init__(self, prediction=120.0):
        self.prediction = prediction
        self.features = []

    def predict(self, features):
        self.features.append(np.array(features, copy=True))
        return np.array([self.prediction], dtype=np.float64)


def search_result(*, elapsed=20.0, depth=1, nodes=60, timed_out=False):
    move = chess.Move.from_uci("e2e4")
    return IterativeSearchResult(
        move=move,
        score=4,
        completed_depth=depth,
        nodes=nodes,
        time_ms=elapsed,
        timed_out=timed_out,
        depths_completed=(DepthSearchStats(depth, move.uci(), 4, nodes, elapsed),),
    )


class Phase2GTests(unittest.TestCase):
    def test_safety_budget_accounts_for_probe_clock_and_reserve(self):
        budget, usable, capped = safe_budget(500, 1_000, probe_elapsed_ms=75)
        self.assertEqual(usable, 625)
        self.assertEqual(budget, 500)
        self.assertFalse(capped)
        budget, usable, capped = safe_budget(900, 1_000, probe_elapsed_ms=75)
        self.assertEqual((budget, usable, capped), (625, 625, True))
        budget, usable, capped = safe_budget(2_000, 2_000, probe_elapsed_ms=100, inference_elapsed_ms=50)
        self.assertEqual((budget, usable, capped), (1_550, 1_550, True))

    def test_zero_usable_clock_never_allocates_negative_budget(self):
        for clock in (300, 299, 0):
            with self.subTest(clock=clock):
                self.assertEqual(safe_budget(500, clock), (0.0, 0.0, True))
        self.assertEqual(safe_budget(-3, 500), (0.0, 200.0, True))

    def test_both_branches_use_same_input_and_model_uses_probe_clock(self):
        model = TestModel(110.0)
        probe = search_result(elapsed=30.0, depth=1, nodes=48, timed_out=True)
        model_search = search_result(elapsed=70.0, depth=2, nodes=130)
        baseline = search_result(elapsed=225.0, depth=3, nodes=500)
        record = {"sample_id": "heldout-1", "fen": chess.STARTING_FEN, "clock": {"remaining_time_ms": 5_000}}
        with patch("app.learning.time_management.phase2g.iterative_search", side_effect=[baseline, probe, model_search]) as search:
            output = run_case(record, model=model, model_version="test")
        self.assertEqual(output["baseline"]["allocated_budget_ms"], 250)
        self.assertEqual(output["model"]["probe_budget_ms"], 25)
        self.assertGreaterEqual(output["model"]["actual_probe_time_ms"], 30)
        self.assertEqual(output["model"]["predicted_time_ms"], 110)
        self.assertEqual(output["model"]["final_model_budget_ms"], 110)
        self.assertAlmostEqual(
            output["model"]["usable_time_ms"],
            5_000 - output["model"]["actual_probe_time_ms"] - output["model"]["model_inference_latency_ms"] - 300,
        )
        self.assertAlmostEqual(output["model"]["total_policy_time_ms"], output["model"]["actual_probe_time_ms"] + 70.0)
        self.assertEqual(output["baseline"]["move"], "e2e4")
        self.assertEqual(output["model"]["completed_depth"], 2)
        self.assertEqual(search.call_count, 3)
        features = model.features[0][0]
        self.assertEqual(features.shape, (len(FEATURE_NAMES),))
        self.assertAlmostEqual(features[FEATURE_NAMES.index("remaining_time_ms")], 5_000 - output["model"]["actual_probe_time_ms"])

    def test_model_branch_skips_at_or_below_safety_margin(self):
        model = TestModel()
        for clock in (300, 299, 0):
            with self.subTest(clock=clock):
                record = {"sample_id": str(clock), "fen": chess.STARTING_FEN, "clock": {"remaining_time_ms": clock}}
                with patch("app.learning.time_management.phase2g.iterative_search") as search:
                    output = run_case(record, model=model, model_version="test")
                self.assertTrue(output["model"]["skipped"])
                self.assertEqual(output["model"]["skipped_reason"], "no_usable_time_after_safety_reserve")
                self.assertEqual(output["model"]["final_model_budget_ms"], 0)
                self.assertEqual(output["model"]["actual_probe_time_ms"], 0)
                search.assert_not_called()

    def test_budget_never_exceeds_remaining_clock_after_probe_and_margin(self):
        model = TestModel(20_000)
        with patch("app.learning.time_management.phase2g.iterative_search", side_effect=[search_result(elapsed=10), search_result(elapsed=12), search_result(elapsed=1)]) as search:
            output = run_case(
                {"fen": chess.STARTING_FEN, "clock": {"remaining_time_ms": 1_000}},
                model=model,
                model_version="test",
            )
        model_branch = output["model"]
        self.assertLessEqual(model_branch["final_model_budget_ms"], 1_000 - model_branch["actual_probe_time_ms"] - model_branch["model_inference_latency_ms"] - 300)
        self.assertGreaterEqual(model_branch["final_model_budget_ms"], 0)
        self.assertEqual(model_branch["final_model_budget_ms"], model_branch["usable_time_ms"])
        self.assertTrue(model_branch["safety_cap_applied"])
        self.assertEqual(search.call_count, 3)

    def test_non_finite_model_prediction_is_recorded_as_failure(self):
        model = TestModel(float("nan"))
        with patch("app.learning.time_management.phase2g.iterative_search", return_value=search_result(elapsed=10)):
            output = run_case(
                {"fen": chess.STARTING_FEN, "clock": {"remaining_time_ms": 5_000}},
                model=model,
                model_version="test",
            )
        self.assertTrue(output["model"]["skipped"])
        self.assertEqual(output["model"]["skipped_reason"], "model_or_search_failed")
        self.assertIn("non-finite", output["model"]["runtime_error"])

    def test_report_has_paired_differences_safety_and_recommendation(self):
        model = TestModel(100.0)
        with patch("app.learning.time_management.phase2g.iterative_search", side_effect=[search_result(elapsed=230), search_result(elapsed=25), search_result(elapsed=80)]):
            record = run_case(
                {"sample_id": "1", "fen": chess.STARTING_FEN, "clock": {"remaining_time_ms": 5_000}},
                model=model,
                model_version="test",
            )
        report = report_results([record])
        self.assertEqual(report["test_cases"], 1)
        self.assertEqual(report["fen_groups"], 1)
        self.assertEqual(report["baseline"]["average_search_time_ms"], 230)
        self.assertEqual(report["model"]["average_predicted_budget_ms"], 100)
        self.assertEqual(report["baseline_vs_model_raw_differences"]["paired_case_count"], 1)
        self.assertEqual(report["safety"]["safety_violations"], 0)
        self.assertEqual(report["safety"]["negative_budgets"], 0)
        self.assertEqual(report["safety"]["reserve_violations"], 0)
        self.assertEqual(report["recommendation"]["status"], "GO")
        self.assertNotIn("winner", report["baseline_vs_model_raw_differences"])


if __name__ == "__main__":
    unittest.main()
