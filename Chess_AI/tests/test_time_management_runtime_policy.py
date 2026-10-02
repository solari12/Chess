"""Tests for opt-in production Time Management integration."""

from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import chess
import numpy as np

from app.api.routes import get_ai_move
from app.engine.search import search, search_with_time_budget
from app.learning.time_management.model_features import FEATURE_NAMES
from app.learning.time_management.runtime_policy import decide_search_budget
from app.learning.time_management.runtime_shadow import _load_cached_model, load_shadow_model_with_timing
from app.schemas.chess import AIRequest, AIResponse, TimedAIRequest


def fake_result(elapsed: float = 10.0):
    move = chess.Move.from_uci("e2e4")
    return SimpleNamespace(
        move=move,
        score=12,
        completed_depth=2,
        nodes=45,
        time_ms=elapsed,
        timed_out=True,
        depths_completed=(),
    )


class ConstantModel:
    n_features_in_ = len(FEATURE_NAMES)

    def __init__(self, prediction: float):
        self.prediction = prediction
        self.features = None

    def predict(self, features):
        self.features = np.array(features, copy=True)
        return np.array([self.prediction])


class RuntimePolicyTests(unittest.TestCase):
    def test_frozen_model_load_is_measured_once_and_reused(self):
        _load_cached_model.cache_clear()
        try:
            cold_model, cold_version, cold_load_ms = load_shadow_model_with_timing()
            warm_model, warm_version, warm_load_ms = load_shadow_model_with_timing()
            self.assertIs(cold_model, warm_model)
            self.assertEqual(cold_version, warm_version)
            self.assertGreater(cold_load_ms, 0.0)
            self.assertEqual(warm_load_ms, 0.0)
            self.assertEqual(_load_cached_model.cache_info().misses, 1)
        finally:
            _load_cached_model.cache_clear()

    def test_model_prediction_produces_a_valid_budget_and_canonical_features(self):
        model = ConstantModel(120.0)
        probe = unittest.mock.Mock(return_value=fake_result())
        decision = decide_search_budget(
            chess.Board(), remaining_time_ms=5_000, max_depth=4, model=model, search_fn=probe,
        )
        self.assertEqual(decision.predicted_time_ms, 120.0)
        self.assertGreater(decision.final_search_budget_ms, 0)
        self.assertLessEqual(
            decision.final_search_budget_ms,
            5_000 - decision.probe_time_ms - decision.inference_latency_ms - 300,
        )
        self.assertFalse(decision.fallback_used)
        self.assertEqual(model.features.shape, (1, len(FEATURE_NAMES)))
        self.assertEqual(probe.call_args.args[1], 25.0)
        self.assertEqual(probe.call_args.kwargs["max_depth"], 64)

    def test_model_prediction_is_capped_by_safety_reserve(self):
        model = ConstantModel(10_000.0)
        with patch("app.learning.time_management.runtime_policy.iterative_search", return_value=fake_result()):
            decision = decide_search_budget(
                chess.Board(), remaining_time_ms=1_000, max_depth=3, model=model,
            )
        self.assertTrue(decision.safety_cap_applied)
        self.assertLessEqual(
            decision.final_search_budget_ms,
            max(1_000 - decision.policy_latency_ms - 300, 0),
        )

    def test_prediction_and_fallback_budgets_never_become_negative(self):
        model = ConstantModel(-1.0)
        with patch("app.learning.time_management.runtime_policy.iterative_search", return_value=fake_result()):
            decision = decide_search_budget(
                chess.Board(), remaining_time_ms=500, max_depth=3, model=model,
            )
        self.assertTrue(decision.fallback_used)
        self.assertEqual(decision.fallback_reason, "non_positive_prediction")
        self.assertGreaterEqual(decision.final_search_budget_ms, 0)
        self.assertLessEqual(decision.final_search_budget_ms, 500 - decision.policy_latency_ms - 300)

    def test_clock_at_or_below_reserve_skips_probe_and_has_zero_budget(self):
        for clock in (300, 299, 0):
            with self.subTest(clock=clock), patch(
                "app.learning.time_management.runtime_policy.iterative_search"
            ) as probe:
                decision = decide_search_budget(
                    chess.Board(), remaining_time_ms=clock, max_depth=2, model=ConstantModel(100.0),
                )
            probe.assert_not_called()
            self.assertEqual(decision.final_search_budget_ms, 0)
            self.assertEqual(decision.fallback_reason, "insufficient_remaining_clock")

    def test_model_loading_failure_uses_capped_250_ms_fallback(self):
        with patch("app.learning.time_management.runtime_policy.iterative_search", return_value=fake_result()):
            decision = decide_search_budget(
                chess.Board(), remaining_time_ms=2_000, max_depth=5,
                model_loader=unittest.mock.Mock(side_effect=OSError("missing artifact")),
            )
        self.assertTrue(decision.fallback_used)
        self.assertIn("model_or_feature_failure", decision.fallback_reason)
        self.assertEqual(decision.final_search_budget_ms, 250.0)

    def test_feature_extraction_failure_uses_fallback(self):
        with patch("app.learning.time_management.runtime_policy.iterative_search", return_value=fake_result()), patch(
            "app.learning.time_management.runtime_policy.extract_position_features",
            side_effect=ValueError("feature failure"),
        ):
            decision = decide_search_budget(
                chess.Board(), remaining_time_ms=2_000, max_depth=5, model=ConstantModel(90.0),
            )
        self.assertTrue(decision.fallback_used)
        self.assertEqual(decision.final_search_budget_ms, 250.0)

    def test_non_finite_model_output_uses_fallback(self):
        model = ConstantModel(float("nan"))
        with patch("app.learning.time_management.runtime_policy.iterative_search", return_value=fake_result()):
            decision = decide_search_budget(
                chess.Board(), remaining_time_ms=2_000, max_depth=5, model=model,
            )
        self.assertTrue(decision.fallback_used)
        self.assertEqual(decision.final_search_budget_ms, 250.0)

    def test_safety_calculation_failure_falls_back_to_zero_budget(self):
        with patch("app.learning.time_management.runtime_policy.iterative_search", return_value=fake_result()), patch(
            "app.learning.time_management.runtime_policy.safety_governor",
            side_effect=RuntimeError("simulated governor failure"),
        ):
            decision = decide_search_budget(
                chess.Board(), remaining_time_ms=5_000, max_depth=5, model=ConstantModel(90.0),
            )
        self.assertTrue(decision.fallback_used)
        self.assertIn("safety_governor_failed", decision.fallback_reason)
        self.assertEqual(decision.final_search_budget_ms, 0.0)


class ProductionAPIIntegrationTests(unittest.TestCase):
    def test_http_endpoint_accepts_legacy_request_and_keeps_response_shape(self):
        with patch.dict("os.environ", {"TIME_MANAGEMENT_ENABLED": "false"}), patch(
            "app.engine.search.alpha_beta", return_value=fake_result()
        ):
            response = get_ai_move(AIRequest.model_validate({
                "fen": chess.STARTING_FEN,
                "algorithm": "alpha-beta",
                "depth": 2,
            }))
        self.assertEqual(
            set(response.model_dump()),
            {"move", "algorithm", "depth", "score", "nodes", "time_ms"},
        )

    def test_existing_request_and_response_schema_remain_compatible(self):
        request = AIRequest.model_validate({
            "fen": chess.STARTING_FEN,
            "algorithm": "alpha-beta",
            "depth": 2,
        })
        self.assertIsNone(request.remaining_time_ms)
        response = AIResponse.model_validate({
            "move": "e2e4", "algorithm": "alpha-beta", "depth": 2,
            "score": 0, "nodes": 0, "time_ms": 1.0,
        })
        self.assertEqual(set(response.model_dump()), {"move", "algorithm", "depth", "score", "nodes", "time_ms"})

    def test_time_management_disabled_preserves_fixed_depth_path(self):
        request = AIRequest(fen=chess.STARTING_FEN, algorithm="alpha-beta", depth=3, remaining_time_ms=5_000)
        with patch.dict("os.environ", {"TIME_MANAGEMENT_ENABLED": "false"}), patch(
            "app.engine.search.alpha_beta", return_value=fake_result()
        ) as alpha_beta, patch("app.engine.search.decide_search_budget") as policy:
            response = get_ai_move(request)
        alpha_beta.assert_called_once_with(unittest.mock.ANY, 3)
        policy.assert_not_called()
        self.assertEqual(response.depth, 3)

    def test_enabled_policy_budget_is_passed_to_iterative_search_with_requested_depth(self):
        request = AIRequest(fen=chess.STARTING_FEN, algorithm="alpha_beta", depth=4, remaining_time_ms=8_000)
        decision = SimpleNamespace(
            predicted_time_ms=180.0, final_search_budget_ms=150.0, safety_cap_applied=True,
            fallback_used=False, fallback_reason=None, probe_time_ms=25.0,
            inference_latency_ms=0.5, policy_latency_ms=26.0, model_version="phase2c-test",
        )
        with patch.dict("os.environ", {"TIME_MANAGEMENT_ENABLED": "true"}), patch(
            "app.engine.search.decide_search_budget", return_value=decision,
        ) as policy, patch("app.engine.search.iterative_search", return_value=fake_result()) as search_mock:
            response = get_ai_move(request)
        policy.assert_called_once()
        self.assertEqual(policy.call_args.kwargs["max_depth"], 4)
        search_mock.assert_called_once()
        self.assertEqual(search_mock.call_args.kwargs["time_budget_ms"], 150.0)
        self.assertEqual(search_mock.call_args.kwargs["max_depth"], 4)
        self.assertNotEqual(search_mock.call_args.kwargs["time_budget_ms"], decision.predicted_time_ms)
        self.assertEqual(response.algorithm, "alpha-beta")
        self.assertEqual(response.depth, 4)
        self.assertEqual(response.time_ms, 10.0)

    def test_enabled_policy_is_used_by_the_chess_pages_timed_endpoint(self):
        request = TimedAIRequest(
            fen=chess.STARTING_FEN, algorithm="alpha-beta", time_budget_ms=2_000,
            remaining_time_ms=8_000, max_depth=5,
        )
        decision = SimpleNamespace(
            predicted_time_ms=170.0, final_search_budget_ms=140.0, safety_cap_applied=True,
            fallback_used=False, fallback_reason=None, probe_time_ms=25.0,
            inference_latency_ms=1.0, policy_latency_ms=27.0, model_version="phase2c-test",
        )
        with patch.dict("os.environ", {"TIME_MANAGEMENT_ENABLED": "true"}), patch(
            "app.engine.search.decide_search_budget", return_value=decision,
        ) as policy, patch("app.engine.search.iterative_search", return_value=fake_result()) as search_mock, patch(
            "app.engine.search.evaluate_and_log_shadow"
        ) as shadow:
            response = search_with_time_budget(request)
        policy.assert_called_once()
        self.assertEqual(policy.call_args.kwargs["max_depth"], 5)
        self.assertEqual(search_mock.call_args.kwargs["time_budget_ms"], 140.0)
        self.assertEqual(search_mock.call_args.kwargs["max_depth"], 5)
        shadow.assert_not_called()
        self.assertEqual(response.completed_depth, 2)
        self.assertEqual(response.algorithm, "alpha-beta")

    def test_enabled_low_clock_returns_legal_move_without_unsafe_search(self):
        request = AIRequest(fen=chess.STARTING_FEN, algorithm="alpha-beta", depth=4, remaining_time_ms=250)
        decision = SimpleNamespace(
            predicted_time_ms=None, final_search_budget_ms=0.0, safety_cap_applied=True,
            fallback_used=True, fallback_reason="insufficient_remaining_clock", probe_time_ms=0.0,
            inference_latency_ms=0.0, policy_latency_ms=0.1, model_version=None,
        )
        with patch.dict("os.environ", {"TIME_MANAGEMENT_ENABLED": "true"}), patch(
            "app.engine.search.decide_search_budget", return_value=decision,
        ), patch("app.engine.search.iterative_search") as search_mock:
            response = search(request)
        search_mock.assert_not_called()
        self.assertIn(chess.Move.from_uci(response.move), chess.Board().legal_moves)
        self.assertEqual(response.nodes, 0)

    def test_http_model_failure_uses_250ms_baseline_fallback(self):
        request = AIRequest(fen=chess.STARTING_FEN, algorithm="alpha-beta", depth=3, remaining_time_ms=2_000)
        with patch.dict("os.environ", {"TIME_MANAGEMENT_ENABLED": "true"}), patch(
            "app.learning.time_management.runtime_policy.iterative_search", return_value=fake_result()
        ) as probe, patch(
            "app.learning.time_management.runtime_policy.load_shadow_model_with_timing",
            side_effect=OSError("simulated model load failure"),
        ), patch(
            "app.engine.search.iterative_search", return_value=fake_result(42.0)
        ) as main_search:
            response = get_ai_move(request)
        self.assertEqual(response.algorithm, "alpha-beta")
        self.assertEqual(response.depth, 3)
        self.assertEqual(response.time_ms, 42.0)
        self.assertEqual(probe.call_count, 1)
        self.assertEqual(main_search.call_args.kwargs["time_budget_ms"], 250.0)
        self.assertEqual(main_search.call_args.kwargs["max_depth"], 3)

    def test_minimax_and_legacy_requests_do_not_enter_time_policy(self):
        with patch.dict("os.environ", {"TIME_MANAGEMENT_ENABLED": "true"}), patch(
            "app.engine.search.minimax", return_value=fake_result()
        ), patch("app.engine.search.decide_search_budget") as policy:
            response = search(AIRequest(fen=chess.STARTING_FEN, algorithm="minimax", depth=2, remaining_time_ms=5_000))
        policy.assert_not_called()
        self.assertEqual(response.algorithm, "minimax")

        with patch.dict("os.environ", {"TIME_MANAGEMENT_ENABLED": "true"}), patch(
            "app.engine.search.alpha_beta", return_value=fake_result()
        ), patch("app.engine.search.decide_search_budget") as policy:
            search(AIRequest(fen=chess.STARTING_FEN, algorithm="alpha-beta", depth=2))
        policy.assert_not_called()


if __name__ == "__main__":
    unittest.main()
