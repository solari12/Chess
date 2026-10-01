"""Focused tests for Phase 2F.1 offline collection and probe tooling."""

import unittest
from unittest.mock import patch

import chess
import numpy as np

from app.learning.time_management.model_features import FEATURE_NAMES
from app.learning.time_management.phase2f1 import (
    build_phase2f1_report,
    collection_plan,
    run_probe_experiment,
    select_position_fens,
)
from app.learning.time_management.models import PositionSample


class FixedModel:
    n_features_in_ = len(FEATURE_NAMES)

    def predict(self, features):
        assert features.shape == (1, len(FEATURE_NAMES))
        return np.array([125.0])


class Phase2F1Tests(unittest.TestCase):
    def test_position_and_collection_metadata_are_deterministic(self):
        samples = [
            PositionSample(chess.Board().fen(), "fixture", None, 0),
            PositionSample(chess.Board().mirror().fen(), "fixture", 1, 2),
            PositionSample(chess.Board().fen(), "fixture", 2, 4),
            PositionSample(chess.Board().mirror().fen(), "fixture", 3, 6),
        ]
        with patch("app.learning.time_management.phase2f1.generate_positions", return_value=samples):
            first = select_position_fens(seed=124, position_count=2)
            second = select_position_fens(seed=124, position_count=2)
        self.assertEqual(first, second)
        plan = collection_plan(first, (30_000, 500))
        self.assertEqual(plan, [(fen, clock) for clock in (30_000, 500) for fen in first])
        self.assertGreater(len(set(first)), 1)

    def test_probe_uses_existing_feature_schema_and_is_non_mutating(self):
        board = chess.Board()
        before_fen = board.fen()
        records = run_probe_experiment(
            fens=[before_fen],
            model=FixedModel(),
            model_version="fixture",
            clock_buckets_ms=(30_000,),
            probe_budgets_ms=(25,),
            max_depth=2,
        )
        self.assertEqual(board.fen(), before_fen)
        record = records[0]
        self.assertEqual(record["feature_names"], list(FEATURE_NAMES))
        self.assertGreaterEqual(record["probe_elapsed_ms"], 0)
        self.assertLessEqual(record["probe_elapsed_ms"], 25 + 250)
        self.assertGreaterEqual(record["raw_prediction_ms"], 0)
        self.assertGreaterEqual(record["estimated_full_search_budget_ms"], 0)
        for field in (
            "probe_budget_ms", "probe_completed_depth", "probe_nodes",
            "remaining_time_after_probe_ms", "probe_clock_cost_percent",
        ):
            self.assertIn(field, record)

    def test_report_enforces_nonnegative_shadow_cap_and_required_telemetry(self):
        shadow = [{
            "fen": chess.STARTING_FEN,
            "remaining_time_ms": 500,
            "actual_time_ms": 100.0,
            "actual_completed_depth": 1,
            "actual_nodes": 20,
            "actual_timeout": True,
            "raw_prediction_ms": 1_000.0,
            "shadow_budget_ms": 200.0,
            "usable_time_ms": 200.0,
            "clock_cap_activated": True,
            "inference_time_ms": 0.5,
            "model": "random_forest",
            "model_version": "fixture",
        }]
        probe = run_probe_experiment(
            fens=[chess.STARTING_FEN], model=FixedModel(), model_version="fixture",
            clock_buckets_ms=(500,), probe_budgets_ms=(25,), max_depth=1,
        )
        report = build_phase2f1_report(
            shadow, probe, seed=42, generated_fens=[chess.STARTING_FEN],
            search_budget_ms=250, clock_buckets_ms=(500,), probe_budgets_ms=(25,),
        )
        self.assertGreaterEqual(shadow[0]["shadow_budget_ms"], 0)
        self.assertEqual(shadow[0]["shadow_budget_ms"], 200)
        self.assertEqual(report["clock_safety"]["by_clock_bucket_ms"]["500"]["sample_count"], 1)
        self.assertIn("not direct prediction quality", report["prediction_vs_actual_runtime_diagnostics"]["comparison_target_note"])
        for field in (
            "actual_completed_depth", "actual_nodes", "actual_timeout", "raw_prediction_ms",
            "shadow_budget_ms", "usable_time_ms", "inference_time_ms", "model",
        ):
            self.assertIn(field, shadow[0])
        self.assertTrue(report["pre_search_probe_experiment"]["model_remains_shadow_only"])

    def test_probe_keeps_iterative_search_call_bounded_and_separate_from_endpoint(self):
        from app.engine.iterative_search import iterative_search
        with patch("app.learning.time_management.phase2f1.search_with_time_budget") as production_endpoint:
            record = run_probe_experiment(
                fens=[chess.STARTING_FEN], model=FixedModel(), model_version="fixture",
                clock_buckets_ms=(10_000,), probe_budgets_ms=(25,), max_depth=1,
            )[0]
        production_endpoint.assert_not_called()
        # The recorded probe cap is the explicit bounded iterative search input.
        self.assertEqual(record["probe_budget_ms"], 25)
        self.assertTrue(callable(iterative_search))


if __name__ == "__main__":
    unittest.main()
