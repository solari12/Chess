"""Post-search, log-only inference using the Phase 2C Random Forest artifact."""

from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
import json
import logging
import math
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any

import chess
import joblib
import numpy as np

from app.engine.iterative_search import IterativeSearchResult
from app.learning.time_management.features import extract_position_features
from app.learning.time_management.model_features import FEATURE_NAMES, extract_model_features


logger = logging.getLogger(__name__)
TIME_MANAGEMENT_ROOT = Path(__file__).resolve().parents[3] / "data" / "time_management"
MODEL_DIRECTORY = TIME_MANAGEMENT_ROOT / "models" / "phase2c_20261001T122840Z_seed42"
DEFAULT_MODEL_PATH = MODEL_DIRECTORY / "random_forest.joblib"
DEFAULT_METADATA_PATH = MODEL_DIRECTORY / "metadata.json"
DEFAULT_LOG_PATH = TIME_MANAGEMENT_ROOT / "runtime_shadow.jsonl"
MODEL_NAME = "random_forest"
SAFETY_MARGIN_MS = 300.0
_log_lock = Lock()


class ShadowModelError(RuntimeError):
    """Raised when the configured Phase 2C shadow artifact cannot be used."""


@lru_cache(maxsize=4)
def _load_cached_model(model_path: str, metadata_path: str) -> tuple[Any, str]:
    artifact = Path(model_path)
    metadata_file = Path(metadata_path)
    if not artifact.is_file():
        raise ShadowModelError(f"Phase 2C Random Forest artifact is missing: {artifact}")
    if not metadata_file.is_file():
        raise ShadowModelError(f"Phase 2C model metadata is missing: {metadata_file}")
    try:
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ShadowModelError(f"Could not read Phase 2C model metadata {metadata_file}: {error}") from error
    if metadata.get("feature_names") != list(FEATURE_NAMES):
        raise ShadowModelError("Phase 2C model feature metadata does not match model_features.FEATURE_NAMES")
    try:
        model = joblib.load(artifact)
    except Exception as error:
        raise ShadowModelError(f"Could not load Phase 2C model artifact {artifact}: {error}") from error
    if not callable(getattr(model, "predict", None)):
        raise ShadowModelError(f"Phase 2C model artifact has no predict() method: {artifact}")
    if getattr(model, "n_features_in_", len(FEATURE_NAMES)) != len(FEATURE_NAMES):
        raise ShadowModelError(
            f"Phase 2C model expects {getattr(model, 'n_features_in_', 'unknown')} features; "
            f"runtime schema provides {len(FEATURE_NAMES)}"
        )
    return model, artifact.parent.name


def load_shadow_model(
    model_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
) -> tuple[Any, str]:
    """Load and validate the existing Phase 2C model, cached by artifact path."""
    artifact = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
    metadata = Path(metadata_path) if metadata_path is not None else artifact.parent / "metadata.json"
    return _load_cached_model(str(artifact.resolve()), str(metadata.resolve()))


def build_runtime_record(
    board: chess.Board,
    *,
    remaining_time_ms: int,
    result: IterativeSearchResult,
    actual_search_budget_ms: int,
    max_depth: int,
) -> dict[str, Any]:
    """Build the exact Phase 2C input structure from an already completed search."""
    if remaining_time_ms < 0:
        raise ValueError("remaining_time_ms must be non-negative")
    return {
        "fen": board.fen(),
        "clock": {"remaining_time_ms": remaining_time_ms},
        "position_features": extract_position_features(board, remaining_time_ms),
        "search": {
            "completed_depth": result.completed_depth,
            "total_nodes": result.nodes,
            "total_time_ms": result.time_ms,
            "timed_out": result.timed_out,
            "analysis_budget_ms": actual_search_budget_ms,
            "max_analysis_depth": max_depth,
        },
    }


def infer_shadow_record(
    board: chess.Board,
    *,
    remaining_time_ms: int,
    result: IterativeSearchResult,
    actual_search_budget_ms: int,
    max_depth: int,
    model: Any | None = None,
    model_version: str | None = None,
    safety_margin_ms: float = SAFETY_MARGIN_MS,
) -> dict[str, Any]:
    """Predict a capped shadow budget after the real search has completed."""
    if safety_margin_ms < 0 or not math.isfinite(safety_margin_ms):
        raise ValueError("safety_margin_ms must be a finite non-negative number")
    fen = board.fen()
    record = build_runtime_record(
        board,
        remaining_time_ms=remaining_time_ms,
        result=result,
        actual_search_budget_ms=actual_search_budget_ms,
        max_depth=max_depth,
    )
    features = extract_model_features(record)
    if model is None:
        model, loaded_version = load_shadow_model()
        model_version = model_version or loaded_version
    started = perf_counter()
    prediction = np.asarray(model.predict(features.reshape(1, -1)), dtype=np.float64)
    inference_time_ms = (perf_counter() - started) * 1_000
    if prediction.shape != (1,) or not np.isfinite(prediction).all():
        raise ShadowModelError("Phase 2C Random Forest returned a non-finite or incorrectly shaped prediction")
    raw_prediction_ms = float(prediction[0])
    usable_time_ms = max(float(remaining_time_ms) - safety_margin_ms, 0.0)
    shadow_budget_ms = min(max(raw_prediction_ms, 0.0), usable_time_ms)
    return {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "fen": fen,
        "remaining_time_ms": remaining_time_ms,
        "actual_time_ms": float(result.time_ms),
        "actual_completed_depth": int(result.completed_depth),
        "actual_nodes": int(result.nodes),
        "actual_timeout": bool(result.timed_out),
        "actual_move": result.move.uci() if result.move is not None else None,
        "actual_search_budget_ms": actual_search_budget_ms,
        "actual_max_depth": max_depth,
        "model": MODEL_NAME,
        "model_version": model_version or "injected-test-model",
        "raw_prediction_ms": raw_prediction_ms,
        "shadow_budget_ms": shadow_budget_ms,
        "usable_time_ms": usable_time_ms,
        "clock_cap_activated": raw_prediction_ms > usable_time_ms,
        "inference_time_ms": inference_time_ms,
    }


def append_shadow_record(record: dict[str, Any], log_path: str | Path = DEFAULT_LOG_PATH) -> Path:
    """Append one structured JSONL record without writing to stdout."""
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    with _log_lock, path.open("a", encoding="utf-8", newline="\n") as destination:
        destination.write(encoded)
        destination.write("\n")
    return path


def evaluate_and_log_shadow(
    board: chess.Board,
    *,
    remaining_time_ms: int,
    result: IterativeSearchResult,
    actual_search_budget_ms: int,
    max_depth: int,
    log_path: str | Path = DEFAULT_LOG_PATH,
) -> dict[str, Any]:
    """Infer and append exactly one shadow record for one completed search."""
    record = infer_shadow_record(
        board,
        remaining_time_ms=remaining_time_ms,
        result=result,
        actual_search_budget_ms=actual_search_budget_ms,
        max_depth=max_depth,
    )
    append_shadow_record(record, log_path)
    return record
