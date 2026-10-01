"""Tests for Phase 2D multi-seed grouped evaluation and diagnostics."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from app.learning.time_management.evaluate_models import (
    CLOCKS_MS,
    FEATURE_GROUPS,
    evaluate_dataset,
    write_evaluation_artifacts,
)
from app.learning.time_management.model_features import FEATURE_NAMES


def records_for_groups(group_count=16):
    records = []
    for group in range(group_count):
        for clock in CLOCKS_MS:
            records.append(
                {
                    "sample_id": f"P{group:03d}_T{clock}",
                    "fen": f"fen-{group}",
                    "clock": {"remaining_time_ms": clock, "clock_source": "synthetic"},
                    "position_features": {
                        "remaining_time_ms": clock,
                        "legal_move_count": 15 + group % 12,
                        "capture_count": group % 4,
                        "checking_move_count": group % 3,
                        "side_in_check": bool(group % 2),
                        "fullmove_number": 3 + group,
                        "halfmove_clock": group % 8,
                        "material_white": 4_000 + group * 15,
                        "material_black": 3_800,
                        "material_imbalance": 200 + group * 15,
                        "pawn_count_white": 7 - group % 3,
                        "pawn_count_black": 8,
                        "knight_count_white": 1 + group % 2,
                        "knight_count_black": 2,
                        "bishop_count_white": 2,
                        "bishop_count_black": 1 + group % 2,
                        "rook_count_white": 2,
                        "rook_count_black": 2,
                        "queen_count_white": 1,
                        "queen_count_black": 1,
                        "king_safety_basic_indicator": group % 4,
                        "side_to_move": "white" if group % 2 else "black",
                    },
                    "search": {
                        "completed_depth": 2 + group % 3,
                        "total_nodes": 500 + group * 100,
                        "total_time_ms": 980,
                        "timed_out": True,
                        "analysis_budget_ms": 1_000,
                        "max_analysis_depth": 6,
                    },
                    "teacher": {
                        "label_status": "labeled",
                        "teacher_time_ms": float(100 + group * 20),
                        "teacher_depth": 3,
                    },
                }
            )
    return records


class Phase2DEvaluationTests(unittest.TestCase):
    def test_probe_and_ablation_feature_groups_match_specification(self):
        probe_fields = {
            "completed_depth",
            "total_nodes",
            "total_search_time_ms",
            "timed_out",
            "analysis_budget_ms",
            "max_analysis_depth",
        }
        self.assertFalse(probe_fields.intersection(FEATURE_GROUPS["no_probe"]))
        self.assertEqual(len(FEATURE_GROUPS["no_probe"]), len(FEATURE_NAMES) - len(probe_fields))
        self.assertEqual(
            set(FEATURE_GROUPS["minimal"]),
            {"remaining_time_ms", "legal_move_count", "material_imbalance", "side_to_move_is_white", "side_in_check"},
        )
        self.assertNotIn("completed_depth", FEATURE_GROUPS["full_minus_completed_depth"])
        self.assertEqual(len(FEATURE_GROUPS["full_minus_completed_depth"]), len(FEATURE_NAMES) - 1)

    def test_fixed_seed_grouped_evaluation_is_deterministic_and_complete(self):
        records = records_for_groups()
        first, first_errors = evaluate_dataset(records, seeds=(42, 43), include_learning_curve=False)
        second, second_errors = evaluate_dataset(records, seeds=(42, 43), include_learning_curve=False)

        self.assertEqual(first, second)
        self.assertEqual(first_errors, second_errors)
        self.assertEqual(len(first["seed_results"]), 2)
        for seed_result in first["seed_results"]:
            self.assertEqual(seed_result["split"]["fen_group_overlap"], 0)
            self.assertEqual(seed_result["split"]["train_fen_groups"] + seed_result["split"]["test_fen_groups"], 16)
            for clock in (str(value) for value in CLOCKS_MS):
                self.assertIn(clock, seed_result["full_feature_clock_errors"]["random_forest"])
                self.assertEqual(seed_result["full_feature_clock_errors"]["random_forest"][clock]["sample_count"], 4)
        self.assertEqual(len(first_errors["random_forest"]), 20)

    def test_metrics_predictions_and_error_case_fields_are_valid(self):
        report, errors = evaluate_dataset(records_for_groups(), seeds=(42,), include_learning_curve=False)
        result = report["seed_results"][0]
        for group_results in result["feature_groups"].values():
            for model_result in group_results.values():
                if "metrics" in model_result:
                    self.assertTrue(np.isfinite(list(model_result["metrics"].values())).all())
        required = {
            "sample_id", "fen", "remaining_time_ms", "teacher_time_ms", "prediction",
            "absolute_error", "completed_depth", "total_nodes", "legal_move_count",
        }
        self.assertTrue(required.issubset(errors["random_forest"][0]))

    def test_clock_target_distributions_are_reported_for_all_clock_values(self):
        report, _ = evaluate_dataset(records_for_groups(), seeds=(42,), include_learning_curve=False)
        self.assertEqual(set(report["target_distribution_by_clock"]), {str(value) for value in CLOCKS_MS})
        for distribution in report["target_distribution_by_clock"].values():
            for metric in ("count", "min", "median", "mean", "p75", "p90", "max"):
                self.assertIn(metric, distribution)

    def test_report_and_error_artifact_write_valid_json(self):
        report, errors = evaluate_dataset(records_for_groups(), seeds=(42,), include_learning_curve=False)
        with tempfile.TemporaryDirectory() as directory:
            paths = write_evaluation_artifacts(report, errors, output_root=directory)
            loaded_report = json.loads(Path(paths["evaluation"]).read_text(encoding="utf-8"))
            loaded_errors = json.loads(Path(paths["error_cases"]).read_text(encoding="utf-8"))
        self.assertEqual(loaded_report["phase"], "2D")
        self.assertIn("random_forest", loaded_errors)


if __name__ == "__main__":
    unittest.main()
