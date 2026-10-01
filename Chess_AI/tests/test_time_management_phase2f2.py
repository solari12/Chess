"""Held-out Phase 2F.2 tests; none of these tests fit or mutate a model."""

import copy
import unittest

import chess
import numpy as np

from app.engine.iterative_search import DepthSearchStats, IterativeSearchResult
from app.learning.time_management.model_features import FEATURE_NAMES, extract_model_features
from app.learning.time_management.phase2f2 import (
    build_phase2f2_report,
    grouped_heldout_split,
    predict_heldout,
    regression_metrics,
    safety_governor,
)
from app.learning.time_management.runtime_shadow import build_runtime_record, load_shadow_model


class FixedModel:
    n_features_in_ = len(FEATURE_NAMES)

    def __init__(self, value=120.0):
        self.value = value

    def predict(self, features):
        if features.shape[1] != len(FEATURE_NAMES):
            raise AssertionError("Feature schema mismatch")
        return np.asarray([self.value], dtype=np.float64)


def make_record(index: int, clock: int = 3_000, target: float | None = None) -> dict:
    board = chess.Board()
    for _ in range(index % 4):
        board.push(next(iter(board.legal_moves)))
    search_result = IterativeSearchResult(
        move=next(iter(board.legal_moves), None),
        score=0,
        completed_depth=2,
        nodes=45 + index,
        time_ms=75.0 + index,
        timed_out=False,
        depths_completed=(DepthSearchStats(1, "e2e4", 0, 20, 20.0),),
    )
    raw = build_runtime_record(
        board,
        remaining_time_ms=clock,
        result=search_result,
        actual_search_budget_ms=1_000,
        max_depth=6,
    )
    raw.update({"sample_id": f"sample-{index:02d}", "source": "fixture"})
    raw["search"]["analysis_budget_ms"] = 1_000
    raw["search"]["max_analysis_depth"] = 6
    raw["teacher"] = {
        "teacher_time_ms": float(100 + index if target is None else target),
        "teacher_depth": 2,
        "teacher_confidence": 0.5,
        "information_gain_score": 0.1,
        "label_status": "labeled",
        "depth_transitions": [{"teacher_summary": 1}],
    }
    return raw


class Phase2F2Tests(unittest.TestCase):
    def test_grouped_fen_split_is_deterministic_and_has_zero_overlap(self):
        records = [make_record(group * 2 + clock_index, clock) for group in range(8) for clock_index, clock in enumerate((3_000, 10_000))]
        train_a, holdout_a, audit_a = grouped_heldout_split(records, test_size=0.25, random_state=31)
        train_b, holdout_b, audit_b = grouped_heldout_split(records, test_size=0.25, random_state=31)
        train_fens = {row["fen"] for row in train_a}
        holdout_fens = {row["fen"] for row in holdout_a}
        self.assertFalse(train_fens & holdout_fens)
        self.assertEqual(audit_a["fen_overlap_count"], 0)
        self.assertEqual([row["sample_id"] for row in holdout_a], [row["sample_id"] for row in holdout_b])
        self.assertEqual(audit_a, audit_b)

    def test_saved_phase2c_model_loads_and_schema_matches(self):
        model, version = load_shadow_model()
        self.assertTrue(callable(model.predict))
        self.assertEqual(model.n_features_in_, len(FEATURE_NAMES))
        self.assertTrue(version.startswith("phase2c_"))
        self.assertEqual(extract_model_features(make_record(0)).shape, (len(FEATURE_NAMES),))

    def test_teacher_target_and_derived_fields_never_enter_feature_vector(self):
        original = make_record(1)
        modified = copy.deepcopy(original)
        modified["teacher"].update({
            "teacher_time_ms": 999_999,
            "teacher_depth": 99,
            "teacher_confidence": 0.99,
            "information_gain_score": 42,
            "label_status": "insufficient_data",
            "depth_transitions": [{"anything": "else"}],
        })
        self.assertTrue(np.array_equal(extract_model_features(original), extract_model_features(modified)))
        self.assertTrue({"teacher_time_ms", "teacher_depth", "teacher_confidence", "information_gain_score", "label_status"}.isdisjoint(FEATURE_NAMES))

    def test_predictions_are_deterministic_and_nonnegative(self):
        heldout = [make_record(0), make_record(1)]
        train = [make_record(2, target=80), make_record(3, target=140)]
        first = predict_heldout(heldout, train, model=FixedModel(125), model_version="fixture")
        second = predict_heldout(heldout, train, model=FixedModel(125), model_version="fixture")
        self.assertEqual([row["prediction_ms"] for row in first], [row["prediction_ms"] for row in second])
        self.assertTrue(all(row["prediction_ms"] >= 0 for row in first))

    def test_negative_model_output_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "negative prediction"):
            predict_heldout([make_record(0)], [make_record(1)], model=FixedModel(-2))

    def test_baseline_metrics_use_exact_same_heldout_records(self):
        training = [make_record(10, target=100), make_record(11, target=200)]
        heldout = [make_record(0, target=100), make_record(1, target=200)]
        predictions = predict_heldout(heldout, training, model=FixedModel(140))
        report = build_phase2f2_report(heldout, predictions, [], training, {"fen_overlap_count": 0, "heldout_fen_group_count": 2})
        metrics = report["heldout_metrics"]
        self.assertEqual(metrics["model"]["sample_count"], len(heldout))
        self.assertEqual(metrics["training_median_baseline"]["sample_count"], len(heldout))
        self.assertEqual(metrics["training_median_baseline"]["mae_ms"], regression_metrics([100, 200], [150, 150])["mae_ms"])

    def test_safety_governor_at_margin_below_margin_and_zero(self):
        self.assertEqual(safety_governor(900, 300), (0, 0, True))
        self.assertEqual(safety_governor(900, 299), (0, 0, True))
        self.assertEqual(safety_governor(900, 0), (0, 0, True))
        self.assertEqual(safety_governor(100, 500), (100, 200, False))
        self.assertEqual(safety_governor(900, 500), (200, 200, True))

    def test_report_has_required_sections_and_deterministic_top_errors(self):
        training = [make_record(12, target=120), make_record(13, target=180)]
        heldout = [make_record(i, clock=(3_000 if i < 3 else 10_000), target=100 + i * 20) for i in range(25)]
        predictions = predict_heldout(heldout, training, model=FixedModel(140))
        report = build_phase2f2_report(heldout, predictions, [], training, {"fen_overlap_count": 0, "heldout_fen_group_count": len({row['fen'] for row in heldout})})
        top_again = build_phase2f2_report(heldout, predictions, [], training, {"fen_overlap_count": 0, "heldout_fen_group_count": len({row['fen'] for row in heldout})})["top_20_absolute_errors"]
        self.assertEqual(report["top_20_absolute_errors"], top_again)
        for section in (
            "heldout_metrics", "error_by_clock_bucket_ms", "error_by_teacher_depth",
            "error_by_teacher_confidence", "top_20_absolute_errors", "leakage_audit",
            "probe_cost_analysis", "runtime_safety_simulation", "evidence_table", "phase2g_recommendation",
        ):
            self.assertIn(section, report)
        self.assertEqual(len(report["top_20_absolute_errors"]), 20)
        self.assertTrue(report["leakage_audit"]["leakage_check_passed"])


if __name__ == "__main__":
    unittest.main()
