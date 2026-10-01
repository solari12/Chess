"""Phase 2F runtime shadow inference, safety, and logging tests."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import chess
import numpy as np

from app.engine.iterative_search import DepthSearchStats, IterativeSearchResult
from app.engine.search import search_with_time_budget
from app.learning.time_management.evaluate_shadow import evaluate_shadow_records
from app.learning.time_management.features import extract_position_features
from app.learning.time_management.model_features import FEATURE_NAMES, extract_model_features
from app.learning.time_management.runtime_shadow import (
    DEFAULT_METADATA_PATH,
    DEFAULT_MODEL_PATH,
    ShadowModelError,
    _load_cached_model,
    append_shadow_record,
    build_runtime_record,
    evaluate_and_log_shadow,
    infer_shadow_record,
    load_shadow_model,
)
from app.schemas.chess import TimedAIRequest


def fixture_result() -> IterativeSearchResult:
    return IterativeSearchResult(
        move=chess.Move.from_uci("e2e4"),
        score=14,
        completed_depth=2,
        nodes=72,
        time_ms=81.5,
        timed_out=True,
        depths_completed=(DepthSearchStats(1, "e2e4", 10, 20, 20.0),),
    )


class ConstantModel:
    n_features_in_ = len(FEATURE_NAMES)

    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, features):
        return np.asarray([self.value], dtype=np.float64)


class RuntimeShadowTests(unittest.TestCase):
    def test_phase2c_random_forest_artifact_loads_with_training_schema(self):
        model, version = load_shadow_model()

        self.assertTrue(callable(model.predict))
        self.assertEqual(model.n_features_in_, len(FEATURE_NAMES))
        self.assertTrue(version.startswith("phase2c_"))
        self.assertTrue(DEFAULT_MODEL_PATH.is_file())
        self.assertTrue(DEFAULT_METADATA_PATH.is_file())

    def test_runtime_feature_vector_matches_model_feature_schema(self):
        board = chess.Board()
        result = fixture_result()
        record = build_runtime_record(
            board,
            remaining_time_ms=20_000,
            result=result,
            actual_search_budget_ms=2_000,
            max_depth=64,
        )
        vector = extract_model_features(record)
        saved_feature_names = json.loads(Path("data/time_management/model_features.json").read_text())["feature_names"]

        self.assertEqual(tuple(saved_feature_names), FEATURE_NAMES)
        self.assertEqual(vector.shape, (len(saved_feature_names),))
        self.assertTrue(np.isfinite(vector).all())

    def test_inference_is_deterministic_and_prediction_is_numeric(self):
        board = chess.Board()
        result = fixture_result()
        first = infer_shadow_record(
            board,
            remaining_time_ms=20_000,
            result=result,
            actual_search_budget_ms=2_000,
            max_depth=64,
        )
        second = infer_shadow_record(
            board,
            remaining_time_ms=20_000,
            result=result,
            actual_search_budget_ms=2_000,
            max_depth=64,
        )

        self.assertAlmostEqual(first["raw_prediction_ms"], second["raw_prediction_ms"], places=8)
        self.assertIsInstance(first["raw_prediction_ms"], float)
        self.assertGreaterEqual(first["raw_prediction_ms"], 0)

    def test_clock_cap_is_applied_only_to_shadow_recommendation(self):
        record = infer_shadow_record(
            chess.Board(),
            remaining_time_ms=500,
            result=fixture_result(),
            actual_search_budget_ms=400,
            max_depth=4,
            model=ConstantModel(1_000),
            model_version="test-model",
        )

        self.assertEqual(record["usable_time_ms"], 200)
        self.assertEqual(record["shadow_budget_ms"], 200)
        self.assertTrue(record["clock_cap_activated"])
        self.assertEqual(record["actual_search_budget_ms"], 400)

    def test_zero_and_low_clock_never_produce_negative_shadow_budget(self):
        for remaining_ms in (100, 0):
            with self.subTest(remaining_ms=remaining_ms):
                record = infer_shadow_record(
                    chess.Board(),
                    remaining_time_ms=remaining_ms,
                    result=fixture_result(),
                    actual_search_budget_ms=1,
                    max_depth=4,
                    model=ConstantModel(1_000),
                )
                self.assertEqual(record["usable_time_ms"], 0)
                self.assertEqual(record["shadow_budget_ms"], 0)
                self.assertGreaterEqual(record["shadow_budget_ms"], 0)

    def test_shadow_inference_does_not_mutate_board_state(self):
        board = chess.Board()
        before = board.fen()

        infer_shadow_record(
            board,
            remaining_time_ms=30_000,
            result=fixture_result(),
            actual_search_budget_ms=2_000,
            max_depth=64,
        )

        self.assertEqual(board.fen(), before)
        self.assertEqual(len(board.move_stack), 0)

    def test_missing_and_malformed_artifacts_fail_with_clear_errors(self):
        _load_cached_model.cache_clear()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            missing = root / "missing.joblib"
            metadata = root / "metadata.json"
            metadata.write_text(json.dumps({"feature_names": list(FEATURE_NAMES)}), encoding="utf-8")
            with self.assertRaisesRegex(ShadowModelError, "artifact is missing"):
                load_shadow_model(missing, metadata)

            malformed = root / "malformed.joblib"
            malformed.write_text("not a joblib model", encoding="utf-8")
            with self.assertRaisesRegex(ShadowModelError, "Could not load Phase 2C model artifact"):
                load_shadow_model(malformed, metadata)
        _load_cached_model.cache_clear()

    def test_completed_search_appends_exactly_one_jsonl_record(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime_shadow.jsonl"
            record = evaluate_and_log_shadow(
                chess.Board(),
                remaining_time_ms=30_000,
                result=fixture_result(),
                actual_search_budget_ms=2_000,
                max_depth=64,
                log_path=path,
            )
            lines = path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(len(lines), 1)
        self.assertEqual(json.loads(lines[0]), record)
        for field in (
            "timestamp", "fen", "remaining_time_ms", "actual_time_ms", "actual_completed_depth",
            "actual_nodes", "actual_timeout", "model", "raw_prediction_ms", "shadow_budget_ms",
            "usable_time_ms", "inference_time_ms",
        ):
            self.assertIn(field, record)

    def test_engine_move_and_actual_budget_are_identical_with_shadow_logging(self):
        request = TimedAIRequest(
            fen=chess.STARTING_FEN,
            algorithm="alpha-beta",
            time_budget_ms=1_234,
            remaining_time_ms=8_000,
            max_depth=5,
        )
        fixed_result = fixture_result()
        with patch("app.engine.search.iterative_search", return_value=fixed_result) as search_without, patch(
            "app.engine.search.evaluate_and_log_shadow", return_value=None
        ):
            without_shadow = search_with_time_budget(request)
            without_args = search_without.call_args
        with patch("app.engine.search.iterative_search", return_value=fixed_result) as search_with, patch(
            "app.engine.search.evaluate_and_log_shadow", return_value={"shadow_budget_ms": 7}
        ) as shadow_mock:
            with_shadow = search_with_time_budget(request)
            with_args = search_with.call_args

        self.assertEqual(without_shadow, with_shadow)
        self.assertEqual(without_args.args[1:], with_args.args[1:])
        self.assertEqual(without_args.kwargs, with_args.kwargs)
        self.assertEqual(with_args.kwargs["time_budget_ms"], request.time_budget_ms)
        self.assertEqual(with_args.kwargs["max_depth"], request.max_depth)
        self.assertEqual(shadow_mock.call_count, 1)

    def test_real_search_output_is_equivalent_with_shadow_enabled(self):
        request = TimedAIRequest(
            fen=chess.STARTING_FEN,
            algorithm="alpha-beta",
            time_budget_ms=5_000,
            remaining_time_ms=30_000,
            max_depth=2,
        )
        with tempfile.TemporaryDirectory() as directory:
            with patch("app.engine.search.evaluate_and_log_shadow", return_value=None):
                without_shadow = search_with_time_budget(request)
            with_shadow = search_with_time_budget(
                request,
                shadow_log_path=str(Path(directory) / "runtime_shadow.jsonl"),
            )

        self.assertEqual(without_shadow.move, with_shadow.move)
        self.assertEqual(without_shadow.depth, with_shadow.depth)
        self.assertEqual(without_shadow.nodes, with_shadow.nodes)
        self.assertEqual(without_shadow.timed_out, with_shadow.timed_out)

    def test_shadow_report_contains_required_metrics_and_clock_buckets(self):
        records = [
            {
                "remaining_time_ms": 30_000,
                "raw_prediction_ms": 400.0,
                "shadow_budget_ms": 400.0,
                "actual_time_ms": 250.0,
                "clock_cap_activated": False,
                "inference_time_ms": 0.3,
                "model_version": "phase2c_test",
                "fen": chess.STARTING_FEN,
            },
            {
                "remaining_time_ms": 500,
                "raw_prediction_ms": 400.0,
                "shadow_budget_ms": 200.0,
                "actual_time_ms": 250.0,
                "clock_cap_activated": True,
                "inference_time_ms": 0.5,
                "model_version": "phase2c_test",
                "fen": chess.STARTING_FEN,
            },
        ]

        report = evaluate_shadow_records(records)

        self.assertEqual(report["shadow_record_count"], 2)
        self.assertEqual(report["prediction_vs_actual"]["mae_ms"], 100)
        self.assertEqual(report["clock_cap"]["activation_rate"], 0.5)
        self.assertEqual(report["recommendation_by_clock_bucket"]["30s_plus"]["record_count"], 1)
        self.assertEqual(report["recommendation_by_clock_bucket"]["under_3s"]["record_count"], 1)
        self.assertEqual(report["prediction_vs_teacher"]["matched_record_count"], 0)


if __name__ == "__main__":
    unittest.main()
