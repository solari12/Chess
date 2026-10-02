"""Production time-budget policy using the frozen Phase 2C model.

The canonical model schema contains search telemetry, so this policy performs
the same bounded pre-search probe used in the Phase 2G experiment. Every
budget is capped after probe/inference latency and the mandatory clock reserve.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from time import perf_counter
from typing import Any, Callable

import chess
import numpy as np

from app.engine.iterative_search import IterativeSearchResult, iterative_search
from app.learning.time_management.features import extract_position_features
from app.learning.time_management.model_features import FEATURE_NAMES, extract_model_features
from app.learning.time_management.phase2f2 import safety_governor
from app.learning.time_management.runtime_shadow import SAFETY_MARGIN_MS, load_shadow_model_with_timing


BASELINE_BUDGET_MS = 250.0
DEFAULT_PROBE_BUDGET_MS = 25.0
PROBE_MAX_DEPTH = 64


@dataclass(frozen=True)
class TimeManagementDecision:
    predicted_time_ms: float | None
    final_search_budget_ms: float
    safety_cap_applied: bool
    fallback_used: bool
    fallback_reason: str | None
    probe_time_ms: float
    inference_latency_ms: float
    policy_latency_ms: float
    model_version: str | None
    model_load_time_ms: float = 0.0
    feature_extraction_time_ms: float = 0.0


def decide_search_budget(
    board: chess.Board,
    *,
    remaining_time_ms: int,
    max_depth: int,
    model: Any | None = None,
    model_loader: Callable[[], tuple[Any, str]] | None = None,
    search_fn: Callable[..., IterativeSearchResult] | None = None,
    safety_margin_ms: float = SAFETY_MARGIN_MS,
    baseline_budget_ms: float = BASELINE_BUDGET_MS,
    probe_budget_ms: float = DEFAULT_PROBE_BUDGET_MS,
) -> TimeManagementDecision:
    """Predict and safely cap the main search budget; failures use baseline."""
    policy_started = perf_counter()
    probe_search = search_fn or iterative_search
    if remaining_time_ms < 0:
        return _fallback("invalid_remaining_time", remaining_time_ms, 0.0, 0.0, None, policy_started, safety_margin_ms, baseline_budget_ms)
    if not _finite_nonnegative(safety_margin_ms) or not _finite_nonnegative(baseline_budget_ms):
        return _fallback("invalid_safety_configuration", remaining_time_ms, 0.0, 0.0, None, policy_started, 0.0, 0.0)
    if not _finite_nonnegative(probe_budget_ms) or probe_budget_ms <= 0:
        return _fallback("invalid_probe_configuration", remaining_time_ms, 0.0, 0.0, None, policy_started, safety_margin_ms, baseline_budget_ms)
    if remaining_time_ms <= safety_margin_ms:
        return _fallback("insufficient_remaining_clock", remaining_time_ms, 0.0, 0.0, None, policy_started, safety_margin_ms, baseline_budget_ms)

    # Keep the 300 ms reserve even during the probe itself.
    safe_probe_budget = min(float(probe_budget_ms), float(remaining_time_ms) - safety_margin_ms)
    probe_started = perf_counter()
    try:
        probe_result = probe_search(board.copy(stack=False), safe_probe_budget, max_depth=PROBE_MAX_DEPTH)
        probe_elapsed_ms = max((perf_counter() - probe_started) * 1_000, float(probe_result.time_ms))
    except Exception as error:
        probe_elapsed_ms = max((perf_counter() - probe_started) * 1_000, 0.0)
        reason = f"probe_failed:{type(error).__name__}"
        return _fallback(reason, remaining_time_ms, probe_elapsed_ms, 0.0, None, policy_started, safety_margin_ms, baseline_budget_ms)

    remaining_after_probe = max(float(remaining_time_ms) - probe_elapsed_ms, 0.0)
    feature_extraction_ms = 0.0
    model_load_ms = 0.0
    inference_elapsed_ms = 0.0
    feature_started: float | None = None
    inference_started: float | None = None
    try:
        feature_started = perf_counter()
        feature_record = {
            "fen": board.fen(),
            "clock": {"remaining_time_ms": int(remaining_after_probe)},
            "position_features": extract_position_features(board, int(remaining_after_probe)),
            "search": {
                "completed_depth": probe_result.completed_depth,
                "total_nodes": probe_result.nodes,
                "total_time_ms": probe_result.time_ms,
                "timed_out": probe_result.timed_out,
                "analysis_budget_ms": safe_probe_budget,
                "max_analysis_depth": PROBE_MAX_DEPTH,
            },
        }
        features = extract_model_features(feature_record)
        feature_extraction_ms = max((perf_counter() - feature_started) * 1_000, 0.0)
        if features.shape != (len(FEATURE_NAMES),):
            raise ValueError("feature vector does not match the canonical Phase 2C schema")
        model_version: str | None = None
        if model is None:
            load_started = perf_counter()
            try:
                if model_loader is None:
                    model, model_version, model_load_ms = load_shadow_model_with_timing()
                else:
                    model, model_version = model_loader()
            except Exception:
                # Preserve failed cold-load cost in fallback telemetry too.
                model_load_ms = max((perf_counter() - load_started) * 1_000, 0.0)
                raise
        if getattr(model, "n_features_in_", len(FEATURE_NAMES)) != len(FEATURE_NAMES):
            raise ValueError("model feature count does not match the canonical Phase 2C schema")
        inference_started = perf_counter()
        try:
            raw_prediction = model.predict(features.reshape(1, -1))
        finally:
            inference_elapsed_ms = max((perf_counter() - inference_started) * 1_000, 0.0)
        prediction = np.asarray(raw_prediction, dtype=np.float64)
        if prediction.shape != (1,) or not np.isfinite(prediction).all():
            raise ValueError("model prediction must be one finite number")
        predicted_ms = float(prediction[0])
        if predicted_ms <= 0:
            return _fallback(
                "non_positive_prediction", remaining_time_ms, probe_elapsed_ms, inference_elapsed_ms,
                predicted_ms, policy_started, safety_margin_ms, baseline_budget_ms, model_version,
                model_load_time_ms=model_load_ms, feature_extraction_time_ms=feature_extraction_ms,
            )
        policy_elapsed_ms = _policy_elapsed_ms(
            policy_started, probe_elapsed_ms, feature_extraction_ms, model_load_ms, inference_elapsed_ms,
        )
        clock_after_policy_ms = max(int(remaining_time_ms - policy_elapsed_ms), 0)
        final_budget, _, safety_cap_applied = safety_governor(
            predicted_ms, clock_after_policy_ms, safety_margin_ms,
        )
        if final_budget <= 0:
            return _fallback(
                "insufficient_clock_after_probe_and_inference", remaining_time_ms, probe_elapsed_ms, inference_elapsed_ms,
                predicted_ms, policy_started, safety_margin_ms, baseline_budget_ms, model_version,
                model_load_time_ms=model_load_ms, feature_extraction_time_ms=feature_extraction_ms,
            )
        return TimeManagementDecision(
            predicted_time_ms=predicted_ms,
            final_search_budget_ms=final_budget,
            safety_cap_applied=safety_cap_applied,
            fallback_used=False,
            fallback_reason=None,
            probe_time_ms=probe_elapsed_ms,
            inference_latency_ms=inference_elapsed_ms,
            policy_latency_ms=_policy_elapsed_ms(
                policy_started, probe_elapsed_ms, feature_extraction_ms, model_load_ms, inference_elapsed_ms,
            ),
            model_version=model_version,
            model_load_time_ms=model_load_ms,
            feature_extraction_time_ms=feature_extraction_ms,
        )
    except Exception as error:
        if feature_started is not None and feature_extraction_ms == 0.0:
            feature_extraction_ms = max((perf_counter() - feature_started) * 1_000, 0.0)
        if inference_started is not None and inference_elapsed_ms == 0.0:
            inference_elapsed_ms = max((perf_counter() - inference_started) * 1_000, 0.0)
        reason = f"model_or_feature_failure:{type(error).__name__}"
        return _fallback(
            reason, remaining_time_ms, probe_elapsed_ms, inference_elapsed_ms, None,
            policy_started, safety_margin_ms, baseline_budget_ms,
            model_load_time_ms=model_load_ms,
            feature_extraction_time_ms=feature_extraction_ms,
        )


def _fallback(
    reason: str,
    remaining_time_ms: int,
    probe_elapsed_ms: float,
    inference_elapsed_ms: float,
    predicted_time_ms: float | None,
    policy_started: float,
    safety_margin_ms: float,
    baseline_budget_ms: float,
    model_version: str | None = None,
    *,
    model_load_time_ms: float = 0.0,
    feature_extraction_time_ms: float = 0.0,
) -> TimeManagementDecision:
    """Return the 250 ms baseline capped by clock left after all policy work."""
    if not all(_finite_nonnegative(value) for value in (
        probe_elapsed_ms, inference_elapsed_ms, safety_margin_ms, baseline_budget_ms,
    )):
        return TimeManagementDecision(
            predicted_time_ms=predicted_time_ms,
            final_search_budget_ms=0.0,
            safety_cap_applied=True,
            fallback_used=True,
            fallback_reason=f"{reason};unsafe_budget_inputs",
            probe_time_ms=0.0,
            inference_latency_ms=0.0,
            policy_latency_ms=max((perf_counter() - policy_started) * 1_000, 0.0),
            model_version=model_version,
            model_load_time_ms=model_load_time_ms,
            feature_extraction_time_ms=feature_extraction_time_ms,
        )
    # Invalid/missing clocks must never result in a guessed positive budget.
    policy_elapsed_ms = _policy_elapsed_ms(
        policy_started, probe_elapsed_ms, feature_extraction_time_ms, model_load_time_ms, inference_elapsed_ms,
    )
    clock_after_policy_ms = max(int(remaining_time_ms - policy_elapsed_ms), 0)
    try:
        budget, _, safety_cap_applied = safety_governor(
            baseline_budget_ms, clock_after_policy_ms, safety_margin_ms,
        )
    except Exception:
        budget, safety_cap_applied = 0.0, True
        reason = f"{reason};safety_governor_failed"
    return TimeManagementDecision(
        predicted_time_ms=predicted_time_ms,
        final_search_budget_ms=budget,
        safety_cap_applied=safety_cap_applied,
        fallback_used=True,
        fallback_reason=reason,
        probe_time_ms=probe_elapsed_ms,
        inference_latency_ms=inference_elapsed_ms,
        policy_latency_ms=policy_elapsed_ms,
        model_version=model_version,
        model_load_time_ms=model_load_time_ms,
        feature_extraction_time_ms=feature_extraction_time_ms,
    )


def _finite_nonnegative(value: float) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value) and value >= 0


def _policy_elapsed_ms(
    policy_started: float,
    probe_elapsed_ms: float,
    feature_extraction_ms: float,
    model_load_ms: float,
    inference_ms: float,
) -> float:
    wall_elapsed = max((perf_counter() - policy_started) * 1_000, 0.0)
    measured_components = probe_elapsed_ms + feature_extraction_ms + model_load_ms + inference_ms
    return max(wall_elapsed, measured_components)
