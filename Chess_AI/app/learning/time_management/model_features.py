"""Leakage-audited numeric features for the Phase 2C baseline experiment."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


# Ordering is fixed and is also written into model_features.json.
FEATURE_NAMES = (
    "remaining_time_ms",
    "legal_move_count",
    "capture_count",
    "checking_move_count",
    "side_in_check",
    "fullmove_number",
    "halfmove_clock",
    "material_white",
    "material_black",
    "material_imbalance",
    "pawn_count_white",
    "pawn_count_black",
    "knight_count_white",
    "knight_count_black",
    "bishop_count_white",
    "bishop_count_black",
    "rook_count_white",
    "rook_count_black",
    "queen_count_white",
    "queen_count_black",
    "king_safety_basic_indicator",
    "side_to_move_is_white",
    "completed_depth",
    "total_nodes",
    "total_search_time_ms",
    "timed_out",
    "analysis_budget_ms",
    "max_analysis_depth",
)

POSITION_FEATURE_FIELDS = FEATURE_NAMES[:21]
SEARCH_FEATURE_FIELDS = FEATURE_NAMES[22:]

FEATURE_AUDIT = {
    "remaining_time_ms": "Clock state known at the decision point; sourced from clock and checked against position features.",
    "legal_move_count": "Current board legal-move count; available from the current position.",
    "capture_count": "Legal capture options in the current position.",
    "checking_move_count": "Legal checking options in the current position.",
    "side_in_check": "Current check state from the board.",
    "fullmove_number": "Current FEN move counter.",
    "halfmove_clock": "Current FEN draw-rule counter.",
    "material_white": "Current material balance feature from the board.",
    "material_black": "Current material balance feature from the board.",
    "material_imbalance": "Difference between the two current material features.",
    "pawn_count_white": "Current white pawn count.",
    "pawn_count_black": "Current black pawn count.",
    "knight_count_white": "Current white knight count.",
    "knight_count_black": "Current black knight count.",
    "bishop_count_white": "Current white bishop count.",
    "bishop_count_black": "Current black bishop count.",
    "rook_count_white": "Current white rook count.",
    "rook_count_black": "Current black rook count.",
    "queen_count_white": "Current white queen count.",
    "queen_count_black": "Current black queen count.",
    "king_safety_basic_indicator": "Current deterministic king-safety feature from the position.",
    "side_to_move_is_white": "One means White to move; zero means Black to move.",
    "completed_depth": "Teacher-independent depth reached by the bounded Phase 2A probe; requires such a probe before a future decision.",
    "total_nodes": "Teacher-independent nodes counted in the bounded Phase 2A probe; requires such a probe before a future decision.",
    "total_search_time_ms": "Elapsed time of the bounded Phase 2A probe, not the selected teacher target; available only after the probe.",
    "timed_out": "Whether the bounded Phase 2A probe hit its deadline.",
    "analysis_budget_ms": "Configured cap for the bounded probe; known before running it.",
    "max_analysis_depth": "Configured depth cap for the bounded probe; known before running it.",
}

FORBIDDEN_FEATURE_FIELDS = frozenset(
    {
        "teacher_depth",
        "teacher_time_ms",
        "teacher_confidence",
        "information_gain_score",
        "selected_transition_gain",
        "label_status",
        "label_reason",
        "best_move_change_count",
        "max_score_delta",
        "average_score_delta",
        "last_significant_depth",
        "search_stability",
        "depth_transitions",
    }
)


def extract_model_features(record: dict[str, Any]) -> np.ndarray:
    """Return one finite float vector in ``FEATURE_NAMES`` order.

    Teacher fields are never read. Search-profile transitions are deliberately
    excluded because the teacher uses them to create the target and the entire
    profile may not exist when a live time decision is made.
    """
    position = record.get("position_features")
    search = record.get("search")
    clock = record.get("clock")
    if not isinstance(position, dict) or not isinstance(search, dict) or not isinstance(clock, dict):
        raise ValueError("Record must contain clock, position_features, and search objects")

    remaining_time = _number(clock, "remaining_time_ms")
    if remaining_time != _number(position, "remaining_time_ms"):
        raise ValueError("clock and position_features remaining_time_ms values differ")
    side_to_move = position.get("side_to_move")
    if side_to_move not in {"white", "black"}:
        raise ValueError("side_to_move must be 'white' or 'black'")

    values: list[float] = [remaining_time]
    for name in POSITION_FEATURE_FIELDS[1:21]:
        if name == "side_in_check":
            raw = position.get(name)
            if not isinstance(raw, bool):
                raise ValueError(f"{name} must be a boolean")
            values.append(float(raw))
        else:
            values.append(_number(position, name))
    values.append(float(side_to_move == "white"))

    for name in SEARCH_FEATURE_FIELDS:
        if name == "total_search_time_ms":
            source_name = "total_time_ms"
        else:
            source_name = name
        if name == "timed_out":
            raw = search.get(source_name)
            if not isinstance(raw, bool):
                raise ValueError("search.timed_out must be a boolean")
            values.append(float(raw))
        else:
            values.append(_number(search, source_name))

    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (len(FEATURE_NAMES),) or not np.isfinite(vector).all():
        raise ValueError("Model feature vector has an invalid shape or non-finite values")
    return vector


def _number(source: dict[str, Any], key: str) -> float:
    value = source.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Expected finite numeric model input {key!r}")
    return float(value)
