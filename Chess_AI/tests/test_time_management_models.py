"""Tests for Phase 2C feature extraction, grouped training, and artifacts."""

import json
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np

from app.learning.time_management.model_features import (
    FEATURE_NAMES,
    FORBIDDEN_FEATURE_FIELDS,
    extract_model_features,
)
from app.learning.time_management.train_models import (
    build_feature_matrix,
    save_artifacts,
    train_and_evaluate,
)


def make_record(group: int, clock_ms: int, target_ms: float):
    return {
        "sample_id": f"P{group:03d}_T{clock_ms}",
        "fen": f"fen-position-{group}",
        "clock": {"remaining_time_ms": clock_ms, "clock_source": "synthetic"},
        "position_features": {
            "remaining_time_ms": clock_ms,
            "legal_move_count": 20 + group % 10,
            "capture_count": group % 4,
            "checking_move_count": group % 3,
            "side_in_check": bool(group % 2),
            "fullmove_number": 1 + group,
            "halfmove_clock": group % 5,
            "material_white": 4_000 + group * 10,
            "material_black": 4_000,
            "material_imbalance": group * 10,
            "pawn_count_white": 8 - group % 4,
            "pawn_count_black": 8,
            "knight_count_white": 2,
            "knight_count_black": 2,
            "bishop_count_white": 2,
            "bishop_count_black": 2,
            "rook_count_white": 2,
            "rook_count_black": 2,
            "queen_count_white": 1,
            "queen_count_black": 1,
            "king_safety_basic_indicator": group % 4,
            "side_to_move": "white" if group % 2 else "black",
        },
        "search": {
            "analysis_budget_ms": 1_000,
            "max_analysis_depth": 6,
            "completed_depth": 3 + group % 2,
            "total_nodes": 1_000 + group * 100,
            "total_time_ms": 990,
            "timed_out": True,
        },
        "depth_profiles": [{"depth": 1, "move": "e2e4", "score": group, "nodes": 20, "time_ms": 5}],
        "teacher": {
            "label_status": "labeled",
            "teacher_time_ms": target_ms,
            "teacher_depth": 3,
            "teacher_confidence": 0.5,
            "information_gain_score": 0.2,
            "best_move_change_count": 2,
            "max_score_delta": 300,
            "average_score_delta": 100,
            "last_significant_depth": 3,
            "search_stability": 0.8,
        },
    }


class TimeManagementModelFeatureTests(unittest.TestCase):
    def test_features_have_deterministic_order_and_numeric_shape(self):
        record = make_record(1, 30_000, 240)
        first = extract_model_features(record)
        second = extract_model_features(record)

        self.assertEqual(first.shape, (len(FEATURE_NAMES),))
        self.assertTrue(np.isfinite(first).all())
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first[FEATURE_NAMES.index("side_to_move_is_white")], 1.0)
        self.assertEqual(FEATURE_NAMES[0], "remaining_time_ms")

    def test_teacher_target_and_teacher_derived_fields_do_not_leak(self):
        record = make_record(1, 30_000, 240)
        changed = json.loads(json.dumps(record))
        changed["teacher"].update(
            {
                "teacher_time_ms": 99_999,
                "teacher_depth": 6,
                "teacher_confidence": 1.0,
                "information_gain_score": 1.0,
                "best_move_change_count": 99,
                "search_stability": 0,
            }
        )

        np.testing.assert_array_equal(extract_model_features(record), extract_model_features(changed))
        self.assertFalse(FORBIDDEN_FEATURE_FIELDS.intersection(FEATURE_NAMES))
        self.assertNotIn("teacher_time_ms", FEATURE_NAMES)

    def test_clock_mismatch_is_rejected(self):
        record = make_record(1, 30_000, 240)
        record["position_features"]["remaining_time_ms"] = 10_000
        with self.assertRaisesRegex(ValueError, "remaining_time_ms values differ"):
            extract_model_features(record)


class TimeManagementTrainingTests(unittest.TestCase):
    @staticmethod
    def records():
        return [
            make_record(group, clock, 100 + group * 9 + clock / 10_000)
            for group in range(20)
            for clock in (300_000, 120_000, 30_000, 10_000, 3_000)
        ]

    def test_group_split_training_metrics_and_clock_errors(self):
        records = self.records()
        result = train_and_evaluate(records, random_state=17)
        evaluation = result["evaluation"]

        self.assertEqual(evaluation["total_fen_groups"], 20)
        self.assertGreater(evaluation["train_fen_groups"], evaluation["test_fen_groups"])
        self.assertEqual(evaluation["fen_group_overlap"], 0)
        self.assertEqual(evaluation["train_records"] + evaluation["test_records"], len(records))
        self.assertEqual(set(evaluation["metrics"]), {"median_baseline", "gradient_boosting", "random_forest"})
        for model_result in evaluation["metrics"].values():
            self.assertTrue(all(np.isfinite(list(model_result["metrics"].values()))))
            self.assertTrue(np.isfinite(list(model_result["prediction_distribution"].values())).all())
            self.assertGreater(sum(group["sample_count"] for group in model_result["clock_group_errors"].values()), 0)
        self.assertTrue(evaluation["clock_is_a_feature"])

    def test_model_artifacts_are_saved_and_can_be_loaded(self):
        records = self.records()
        result = train_and_evaluate(records, random_state=4)
        with tempfile.TemporaryDirectory() as directory:
            paths = save_artifacts(result, output_root=directory, random_state=4)
            reloaded_gb = joblib.load(paths["gradient_boosting"])
            reloaded_rf = joblib.load(paths["random_forest"])
            metadata = json.loads(Path(paths["metadata"]).read_text(encoding="utf-8"))
            features = json.loads(Path(paths["features"]).read_text(encoding="utf-8"))
            evaluation = json.loads(Path(paths["evaluation"]).read_text(encoding="utf-8"))
            sample_x = build_feature_matrix(records[:2])

        self.assertEqual(reloaded_gb.predict(sample_x).shape, (2,))
        self.assertEqual(reloaded_rf.predict(sample_x).shape, (2,))
        self.assertEqual(metadata["split_method"], "grouped_by_fen")
        self.assertEqual(features["feature_names"], list(FEATURE_NAMES))
        self.assertIn("gradient_boosting", evaluation["metrics"])


if __name__ == "__main__":
    unittest.main()
