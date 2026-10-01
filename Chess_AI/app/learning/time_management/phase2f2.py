"""Held-out Phase 2F.2 Teacher-vs-model validation (no model fitting)."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
from typing import Any, Sequence

import chess
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

from app.engine.iterative_search import iterative_search
from app.learning.time_management.model_features import (
    FEATURE_NAMES,
    extract_model_features,
)
from app.learning.time_management.runtime_shadow import (
    DEFAULT_MODEL_PATH,
    SAFETY_MARGIN_MS,
    build_runtime_record,
    load_shadow_model,
)


ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = ROOT / "data" / "time_management"
DEFAULT_LABELED_DATASET = DATA_ROOT / "validation_training_dataset.jsonl"
DEFAULT_MODEL_EVALUATION = DATA_ROOT / "model_evaluation.json"
DEFAULT_HELDOUT_PATH = DATA_ROOT / "phase2f2_heldout_dataset.jsonl"
DEFAULT_PREDICTIONS_PATH = DATA_ROOT / "phase2f2_heldout_predictions.jsonl"
DEFAULT_PROBES_PATH = DATA_ROOT / "phase2f2_probe_predictions.jsonl"
DEFAULT_REPORT_PATH = DATA_ROOT / "phase2f2_report.json"
PROBE_BUDGETS_MS = (25, 50, 100, 150)
LOW_CLOCKS_MS = (1_000, 750, 500, 300)
CONFIDENCE_BINS = (("low", 0.0, 0.4), ("medium", 0.4, 0.7), ("high", 0.7, math.inf))
TEACHER_INPUT_FIELDS = frozenset({
    "teacher_time_ms", "teacher_depth", "teacher_confidence", "confidence",
    "information_gain", "information_gain_score", "label_status",
    "selected_transition_gain", "depth_transitions", "best_move_change_count",
    "max_score_delta", "average_score_delta", "last_significant_depth",
    "search_stability", "label_reason", "usable_clock_ms",
})


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as source:
        for line_no, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path} line {line_no}: {error}") from error
            if not isinstance(item, dict):
                raise ValueError(f"Expected JSON object in {path} line {line_no}")
            records.append(item)
    return records


def write_jsonl(path: str | Path, records: Sequence[dict[str, Any]]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )
    return output


def grouped_heldout_split(
    records: Sequence[dict[str, Any]], *, test_size: float, random_state: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Recreate the Phase 2C GroupShuffleSplit exactly without fitting a model."""
    labeled = [record for record in records if record.get("teacher", {}).get("label_status") == "labeled"]
    if len(labeled) < 2:
        raise ValueError("At least two genuinely labeled records are required")
    fens = np.asarray([_required_str(record, "fen") for record in labeled], dtype=object)
    targets = np.asarray([_teacher_target(record) for record in labeled], dtype=np.float64)
    # Calling the shared extractor also validates that the held-out rows have
    # the same 28 telemetry inputs expected by the saved model.
    features = np.vstack([extract_model_features(record) for record in labeled])
    if features.shape[1] != len(FEATURE_NAMES):
        raise ValueError("Held-out feature matrix does not match the model schema")
    if len(set(fens.tolist())) < 2:
        raise ValueError("Grouped holdout needs at least two distinct FENs")
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(features, targets, groups=fens))
    train = [labeled[int(index)] for index in train_idx]
    heldout = [labeled[int(index)] for index in test_idx]
    train_fens = {record["fen"] for record in train}
    heldout_fens = {record["fen"] for record in heldout}
    overlap = train_fens & heldout_fens
    if overlap:
        raise AssertionError(f"FEN leakage in grouped split: {len(overlap)} overlapping groups")
    return train, heldout, {
        "method": "sklearn.model_selection.GroupShuffleSplit",
        "group_key": "fen",
        "test_size": test_size,
        "random_state": random_state,
        "training_record_count": len(train),
        "training_fen_group_count": len(train_fens),
        "heldout_record_count": len(heldout),
        "heldout_fen_group_count": len(heldout_fens),
        "fen_overlap_count": len(overlap),
    }


def create_heldout_dataset(
    source_path: str | Path = DEFAULT_LABELED_DATASET,
    evaluation_path: str | Path = DEFAULT_MODEL_EVALUATION,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Use the exact Phase 2C model train/test FEN partition and verify metadata."""
    evaluation = json.loads(Path(evaluation_path).read_text(encoding="utf-8"))
    model, metadata_version = load_shadow_model()
    del model
    metadata_path = DEFAULT_MODEL_PATH.parent / "metadata.json"
    model_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if model_metadata.get("feature_names") != list(FEATURE_NAMES):
        raise ValueError("Saved Phase 2C artifact feature schema does not match model_features.py")
    if model_metadata.get("target") != "teacher_time_ms" or model_metadata.get("random_state") != evaluation.get("random_state"):
        raise ValueError("Saved Phase 2C artifact and evaluation metadata disagree on target or grouped-split seed")
    source_resolved = Path(source_path).resolve()
    try:
        source_identity = source_resolved.relative_to(ROOT).as_posix()
    except ValueError:
        source_identity = source_resolved.as_posix()
    if evaluation.get("dataset", "").replace("\\", "/") != source_identity:
        raise ValueError("Phase 2C evaluation was produced from a different labeled dataset")
    if evaluation.get("split_method") != "grouped_by_fen":
        raise ValueError("Phase 2C training metadata does not document a grouped FEN split")
    train, heldout, split = grouped_heldout_split(
        load_jsonl(source_path),
        test_size=float(evaluation["test_size"]),
        random_state=int(evaluation["random_state"]),
    )
    split["model_version"] = metadata_version
    split["training_source"] = str(source_path).replace("\\", "/")
    if (
        split["training_fen_group_count"] != evaluation["train_fen_groups"]
        or split["heldout_fen_group_count"] != evaluation["test_fen_groups"]
        or split["training_record_count"] != evaluation["train_records"]
        or split["heldout_record_count"] != evaluation["test_records"]
    ):
        raise ValueError("Recreated grouped split does not match saved Phase 2C training evaluation")
    return train, heldout, split


def predict_heldout(
    heldout: Sequence[dict[str, Any]],
    training: Sequence[dict[str, Any]],
    *,
    model: Any | None = None,
    model_version: str | None = None,
) -> list[dict[str, Any]]:
    """Infer with the saved RF using the canonical feature extractor only."""
    if model is None:
        model, loaded_version = load_shadow_model()
        model_version = model_version or loaded_version
    baseline = float(statistics.median(_teacher_target(record) for record in training))
    rows = []
    for record in heldout:
        target = _teacher_target(record)
        features = extract_model_features(record)
        if features.shape != (len(FEATURE_NAMES),):
            raise ValueError("Feature vector does not match trained model metadata")
        values = np.asarray(model.predict(features.reshape(1, -1)), dtype=np.float64)
        if values.shape != (1,) or not np.isfinite(values).all() or values[0] < 0:
            raise ValueError("Saved model returned an invalid or negative prediction")
        raw = float(values[0])
        remaining = _remaining_clock(record)
        final, usable, activated = safety_governor(raw, remaining)
        teacher = record["teacher"]
        search = record["search"]
        rows.append({
            "sample_id": _required_str(record, "sample_id"),
            "fen": _required_str(record, "fen"),
            "remaining_time_ms": remaining,
            "teacher_time_ms": target,
            "teacher_depth": float(teacher["teacher_depth"]),
            "teacher_confidence": float(teacher["teacher_confidence"]),
            "information_gain_score": float(teacher["information_gain_score"]),
            "label_status": teacher["label_status"],
            "completed_depth": int(search["completed_depth"]),
            "total_nodes": int(search["total_nodes"]),
            "timed_out": bool(search["timed_out"]),
            "raw_prediction_ms": raw,
            "prediction_ms": raw,
            "training_median_baseline_ms": baseline,
            "usable_time_ms": usable,
            "final_shadow_budget_ms": final,
            "clock_cap_activated": activated,
            "model": "random_forest",
            "model_version": model_version or "injected-test-model",
            "evaluation_mode": "oracle_telemetry_from_existing_bounded_search",
            "feature_names": list(FEATURE_NAMES),
        })
    return rows


def safety_governor(raw_prediction_ms: float, remaining_time_ms: int, safety_margin_ms: float = SAFETY_MARGIN_MS) -> tuple[float, float, bool]:
    """Simulate the existing offline recommendation cap without touching runtime."""
    if not math.isfinite(raw_prediction_ms) or raw_prediction_ms < 0:
        raise ValueError("Prediction must be finite and non-negative")
    if remaining_time_ms < 0 or safety_margin_ms < 0 or not math.isfinite(safety_margin_ms):
        raise ValueError("Clock and safety margin must be non-negative")
    usable = max(float(remaining_time_ms) - safety_margin_ms, 0.0)
    return min(raw_prediction_ms, usable), usable, raw_prediction_ms > usable


def run_probe_experiment(
    heldout: Sequence[dict[str, Any]],
    *,
    model: Any | None = None,
    model_version: str | None = None,
    budgets_ms: Sequence[int] = PROBE_BUDGETS_MS,
    low_clocks_ms: Sequence[int] = LOW_CLOCKS_MS,
    fen_group_limit: int = 3,
    max_depth: int = 6,
) -> list[dict[str, Any]]:
    """Run offline probes on three held-out FENs and attach exact Teacher matches."""
    if model is None:
        model, loaded_version = load_shadow_model()
        model_version = model_version or loaded_version
    if not heldout:
        return []
    teacher_by_pair = {(row["fen"], _remaining_clock(row)): row for row in heldout}
    selected_fens = sorted({row["fen"] for row in heldout})[:fen_group_limit]
    pairs: dict[tuple[str, int], dict[str, Any] | None] = {
        (fen, _remaining_clock(row)): row
        for fen in selected_fens
        for row in heldout
        if row["fen"] == fen
    }
    for fen in selected_fens:
        for clock in low_clocks_ms:
            pairs.setdefault((fen, clock), teacher_by_pair.get((fen, clock)))

    results: list[dict[str, Any]] = []
    for (fen, clock), teacher_record in sorted(pairs.items(), key=lambda item: (item[0][0], item[0][1])):
        usable_before_probe = max(float(clock) - SAFETY_MARGIN_MS, 0.0)
        board = chess.Board(fen)
        if not board.is_valid():
            raise ValueError(f"Invalid held-out FEN: {fen}")
        for probe_budget in budgets_ms:
            teacher_value = _teacher_target(teacher_record) if teacher_record is not None else None
            common = {
                "fen": fen,
                "remaining_time_ms": clock,
                "teacher_sample_id": teacher_record.get("sample_id") if teacher_record else None,
                "teacher_time_ms": teacher_value,
                "probe_budget_ms": probe_budget,
                "usable_clock_before_probe_ms": usable_before_probe,
                "model": "random_forest",
                "model_version": model_version or "injected-test-model",
                "probe_mode": "offline_pre_search_feasibility",
            }
            if probe_budget > usable_before_probe or usable_before_probe <= 0:
                results.append({
                    **common,
                    "probe_skipped": True,
                    "probe_skip_reason": "No safe clock remains after the 300 ms reserve" if usable_before_probe <= 0 else "Probe budget exceeds usable clock after reserve",
                    "probe_elapsed_ms": 0.0,
                    "completed_depth": None,
                    "total_nodes": None,
                    "features_available": False,
                    "raw_prediction_ms": None,
                    "absolute_error_ms": None,
                    "signed_error_ms": None,
                    "probe_cost_percent_of_clock": 0.0,
                    "probe_cost_percent_of_usable_clock": None,
                    "usable_clock_after_probe_ms": usable_before_probe,
                })
                continue
            result = iterative_search(board, time_budget_ms=probe_budget, max_depth=max_depth)
            runtime_record = build_runtime_record(
                board,
                remaining_time_ms=clock,
                result=result,
                actual_search_budget_ms=probe_budget,
                max_depth=max_depth,
            )
            features = extract_model_features(runtime_record)
            prediction_array = np.asarray(model.predict(features.reshape(1, -1)), dtype=np.float64)
            if prediction_array.shape != (1,) or not np.isfinite(prediction_array).all() or prediction_array[0] < 0:
                raise ValueError("Saved model returned an invalid or negative probe prediction")
            prediction = float(prediction_array[0])
            elapsed = max(float(result.time_ms), 0.0)
            signed_error = prediction - teacher_value if teacher_value is not None else None
            results.append({
                **common,
                "probe_skipped": False,
                "probe_skip_reason": None,
                "probe_elapsed_ms": elapsed,
                "completed_depth": int(result.completed_depth),
                "total_nodes": int(result.nodes),
                "timed_out": bool(result.timed_out),
                "features_available": True,
                "feature_names": list(FEATURE_NAMES),
                "raw_prediction_ms": prediction,
                "absolute_error_ms": abs(signed_error) if signed_error is not None else None,
                "signed_error_ms": signed_error,
                "probe_cost_percent_of_clock": elapsed / clock * 100 if clock else 0.0,
                "probe_cost_percent_of_usable_clock": elapsed / usable_before_probe * 100 if usable_before_probe else None,
                "usable_clock_after_probe_ms": max(float(clock) - elapsed - SAFETY_MARGIN_MS, 0.0),
            })
    return results


def build_phase2f2_report(
    heldout: Sequence[dict[str, Any]],
    predictions: Sequence[dict[str, Any]],
    probes: Sequence[dict[str, Any]],
    training: Sequence[dict[str, Any]],
    split_audit: dict[str, Any],
) -> dict[str, Any]:
    """Aggregate held-out accuracy, slices, leakage, probe costs, and evidence."""
    by_id = {row["sample_id"]: row for row in predictions}
    if len(by_id) != len(predictions) or set(by_id) != {row["sample_id"] for row in heldout}:
        raise ValueError("Predictions must contain exactly one row for every held-out sample_id")
    pairs = [(row, by_id[row["sample_id"]]) for row in heldout]
    for raw, pred in pairs:
        if (
            pred.get("fen") != raw.get("fen")
            or pred.get("remaining_time_ms") != _remaining_clock(raw)
            or float(pred.get("teacher_time_ms", -1)) != _teacher_target(raw)
        ):
            raise ValueError(f"Prediction row {pred.get('sample_id')} does not match its held-out FEN/clock/Teacher target")
    target = [float(pred["teacher_time_ms"]) for _, pred in pairs]
    estimate = [float(pred["prediction_ms"]) for _, pred in pairs]
    baseline_value = float(statistics.median(_teacher_target(row) for row in training))
    baseline = [baseline_value] * len(target)
    model_metrics = regression_metrics(target, estimate)
    baseline_metrics = regression_metrics(target, baseline)
    absolute = [abs(pred - truth) for truth, pred in zip(target, estimate)]
    signed = [pred - truth for truth, pred in zip(target, estimate)]
    bucket_stats = _group_metrics(pairs, lambda raw, pred: str(_remaining_clock(raw)), "clock")
    depth_stats = _group_metrics(pairs, lambda raw, pred: str(int(float(raw["teacher"]["teacher_depth"]))), "teacher_depth")
    confidence_stats: dict[str, Any] = {}
    for name, lower, upper in CONFIDENCE_BINS:
        group = [pair for pair in pairs if lower <= float(pair[0]["teacher"]["teacher_confidence"]) < upper]
        confidence_stats[name] = {
            "definition": f"{lower:.2f} <= confidence < {upper:.2f}" if math.isfinite(upper) else f"confidence >= {lower:.2f}",
            **_metric_slice(group),
        }
    cap_count = sum(bool(pred["clock_cap_activated"]) for _, pred in pairs)
    observed_confidence_maes = [
        values["mae_ms"] for values in confidence_stats.values() if values["mae_ms"] is not None
    ]
    confidence_mae_range = max(observed_confidence_maes) - min(observed_confidence_maes) if len(observed_confidence_maes) > 1 else None
    feature_names = tuple(FEATURE_NAMES)
    leakage_fields_in_features = sorted(set(feature_names) & TEACHER_INPUT_FIELDS)
    feature_contract = {
        "schema_matches_saved_model_metadata": True,
        "feature_names": list(feature_names),
        "teacher_target_and_derived_fields_in_features": leakage_fields_in_features,
        "leakage_check_passed": not leakage_fields_in_features,
        "excluded_teacher_fields": sorted(TEACHER_INPUT_FIELDS),
        "teacher_fields_used_for_metrics_only": ["teacher_time_ms", "teacher_depth", "teacher_confidence", "information_gain_score", "label_status"],
        "oracle_telemetry_features": ["completed_depth", "total_nodes", "total_search_time_ms", "timed_out", "analysis_budget_ms", "max_analysis_depth"],
        "future_depth_profiles_or_teacher_summaries_in_features": False,
        "oracle_telemetry_warning": "Main metrics use telemetry from an already completed bounded search. They do not prove that pre-search inference is ready.",
    }
    probe_report = _probe_report(probes, baseline_ms=baseline_value)
    probe_25 = probe_report["by_probe_budget_ms"]["25"]
    probe_25_metrics = probe_25["prediction_error_vs_teacher"]
    probe_25_baseline = probe_25["training_median_baseline_error_vs_teacher"]
    low_clock_300 = [row for row in probes if row.get("remaining_time_ms") == 300]
    decision_evidence = {
        "model_beats_training_median_baseline_on_mae": model_metrics["mae_ms"] < baseline_metrics["mae_ms"],
        "model_beats_training_median_baseline_on_rmse": model_metrics["rmse_ms"] < baseline_metrics["rmse_ms"],
        "heldout_fen_overlap_is_zero": split_audit.get("fen_overlap_count") == 0,
        "leakage_check_passed": feature_contract["leakage_check_passed"],
        "small_probe_has_teacher_matched_predictions": any(row.get("probe_budget_ms") == 25 and row.get("features_available") and row.get("teacher_time_ms") is not None for row in probes),
        "pre_search_25ms_probe_beats_matched_training_median": bool(
            probe_25_metrics and probe_25_baseline and probe_25_metrics["mae_ms"] < probe_25_baseline["mae_ms"]
        ),
        "probe_skipped_at_300ms_after_reserve": bool(low_clock_300) and all(row.get("probe_skipped") for row in low_clock_300),
    }
    go = all(decision_evidence.values())
    return {
        "phase": "2F.2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "evaluation_mode": "held_out_teacher_vs_model_oracle_telemetry",
        "target": "teacher_time_ms",
        "split_audit": split_audit,
        "training_set_median_baseline_ms": baseline_value,
        "heldout_metrics": {
            "model": model_metrics,
            "training_median_baseline": baseline_metrics,
            "relative_mae_improvement_percent": (baseline_metrics["mae_ms"] - model_metrics["mae_ms"]) / baseline_metrics["mae_ms"] * 100 if baseline_metrics["mae_ms"] else None,
            "relative_rmse_improvement_percent": (baseline_metrics["rmse_ms"] - model_metrics["rmse_ms"]) / baseline_metrics["rmse_ms"] * 100 if baseline_metrics["rmse_ms"] else None,
            "target_distribution_ms": distribution(target),
            "prediction_distribution_ms": distribution(estimate),
            "signed_error_definition": "prediction_ms - teacher_time_ms; positive is overprediction",
            "absolute_error_tolerance_hit_rate_percent": {
                str(tolerance): sum(error <= tolerance for error in absolute) / len(absolute) * 100 if absolute else None
                for tolerance in (25, 50, 100)
            },
            "overprediction_percent": sum(value > 0 for value in signed) / len(signed) * 100 if signed else None,
            "underprediction_percent": sum(value < 0 for value in signed) / len(signed) * 100 if signed else None,
            "exact_prediction_percent": sum(value == 0 for value in signed) / len(signed) * 100 if signed else None,
        },
        "error_by_clock_bucket_ms": bucket_stats,
        "error_by_teacher_depth": depth_stats,
        "error_by_teacher_confidence": {
            "bin_edges": "low: [0.00, 0.40); medium: [0.40, 0.70); high: [0.70, 1.00].",
            "groups": confidence_stats,
            "error_changes_across_nonempty_bins": confidence_mae_range is not None and confidence_mae_range > 0,
            "mae_range_across_nonempty_bins_ms": confidence_mae_range,
            "interpretation": "Compare the reported MAE/bias across non-empty bins; empty bins mean confidence coverage is insufficient for that comparison.",
        },
        "top_20_absolute_errors": _top_errors(pairs),
        "leakage_audit": feature_contract,
        "probe_cost_analysis": probe_report,
        "runtime_safety_simulation": _safety_report([pred["raw_prediction_ms"] for _, pred in pairs]),
        "evidence_table": {
            "heldout_model_mae_ms": model_metrics["mae_ms"],
            "training_median_baseline_mae_ms": baseline_metrics["mae_ms"],
            "relative_mae_improvement_percent": (baseline_metrics["mae_ms"] - model_metrics["mae_ms"]) / baseline_metrics["mae_ms"] * 100 if baseline_metrics["mae_ms"] else None,
            "model_rmse_ms": model_metrics["rmse_ms"],
            "baseline_rmse_ms": baseline_metrics["rmse_ms"],
            "model_r2": model_metrics["r2"],
            "clock_bucket_mae_ms": {key: value["mae_ms"] for key, value in bucket_stats.items()},
            "teacher_depth_mae_ms": {key: value["mae_ms"] for key, value in depth_stats.items()},
            "probe_cost_by_budget_ms": {key: value["probe_elapsed_ms"]["median"] for key, value in probe_report["by_probe_budget_ms"].items()},
            "heldout_fen_groups": split_audit.get("heldout_fen_group_count"),
            "fen_overlap_count": split_audit.get("fen_overlap_count"),
            "leakage_check_passed": feature_contract["leakage_check_passed"],
        },
        "phase2g_recommendation": {
            "decision": "GO" if go else "NOT YET",
            "decision_basis": "Factual evidence checks only: held-out improvement over the training median on MAE and RMSE, zero FEN overlap, no feature leakage, a teacher-matched 25 ms probe that beats the matched training-median baseline, and no probe at 300 ms where the reserve leaves no usable clock. No arbitrary accuracy cutoff is applied.",
            "checks": decision_evidence,
            "reason": (
                "The frozen model improves over the training-median baseline on held-out teacher labels, with zero FEN overlap and no teacher leakage. Short-probe predictions also beat the matched training-median baseline, and the 300 ms case is skipped because the reserve leaves no usable budget. Evidence supports only a controlled Phase 2G experiment with shadow observation and the existing safety cap."
                if go else _recommendation_reason(decision_evidence, model_metrics, baseline_metrics, probe_25_metrics, probe_25_baseline)
            ),
            "limitations": [
                "This is a single saved grouped holdout from a relatively small corpus; Phase 2G should continue to log shadow results and retain an immediate rollback path.",
                "The corpus has only the clock values present in the existing teacher dataset; low-clock probe predictions have no exact-clock Teacher target unless a matching row exists.",
                "The model's main evaluation is oracle telemetry after a bounded search. Only the separate matched probe rows measure pre-search feasibility.",
                "A GO authorizes evaluation in a controlled experiment, not unrestricted model-controlled gameplay.",
            ],
            "runtime_code_changed": False,
            "model_retrained": False,
        },
    }


def regression_metrics(target: Sequence[float], prediction: Sequence[float]) -> dict[str, float | None]:
    if len(target) != len(prediction) or not target:
        raise ValueError("Metrics require equally sized, non-empty target and prediction arrays")
    truth = np.asarray(target, dtype=np.float64)
    estimate = np.asarray(prediction, dtype=np.float64)
    error = estimate - truth
    return {
        "sample_count": len(truth),
        "mae_ms": float(mean_absolute_error(truth, estimate)),
        "rmse_ms": float(math.sqrt(mean_squared_error(truth, estimate))),
        "median_absolute_error_ms": float(np.median(np.abs(error))),
        "r2": float(r2_score(truth, estimate)) if len(truth) >= 2 else None,
        "mean_signed_error_ms": float(np.mean(error)),
        "median_signed_error_ms": float(np.median(error)),
    }


def distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "median": None, "mean": None, "p90": None, "max": None}
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * 0.9
    lower, upper = math.floor(position), math.ceil(position)
    return {
        "count": len(ordered), "min": ordered[0], "median": float(statistics.median(ordered)),
        "mean": float(statistics.fmean(ordered)),
        "p90": ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower),
        "max": ordered[-1],
    }


def _group_metrics(pairs: Sequence[tuple[dict[str, Any], dict[str, Any]]], key_fn, section: str) -> dict[str, Any]:
    groups: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for raw, pred in pairs:
        groups.setdefault(key_fn(raw, pred), []).append((raw, pred))
    report = {}
    for key, group in sorted(groups.items(), key=lambda item: float(item[0])):
        report[key] = _metric_slice(group)
        if section == "clock":
            report[key].update({
                "teacher_target_median_ms": float(statistics.median(float(p["teacher_time_ms"]) for _, p in group)),
                "teacher_target_mean_ms": float(statistics.fmean(float(p["teacher_time_ms"]) for _, p in group)),
                "prediction_median_ms": float(statistics.median(float(p["prediction_ms"]) for _, p in group)),
                "prediction_mean_ms": float(statistics.fmean(float(p["prediction_ms"]) for _, p in group)),
                "cap_activation_rate": sum(bool(p["clock_cap_activated"]) for _, p in group) / len(group),
            })
        else:
            report[key].update({
                "teacher_target_mean_ms": float(statistics.fmean(float(p["teacher_time_ms"]) for _, p in group)),
                "teacher_target_median_ms": float(statistics.median(float(p["teacher_time_ms"]) for _, p in group)),
                "prediction_mean_ms": float(statistics.fmean(float(p["prediction_ms"]) for _, p in group)),
                "prediction_median_ms": float(statistics.median(float(p["prediction_ms"]) for _, p in group)),
            })
    return report


def _metric_slice(pairs: Sequence[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
    targets = [float(pred["teacher_time_ms"]) for _, pred in pairs]
    outputs = [float(pred["prediction_ms"]) for _, pred in pairs]
    metrics = regression_metrics(targets, outputs) if pairs else {
        "sample_count": 0, "mae_ms": None, "rmse_ms": None,
        "median_absolute_error_ms": None, "r2": None,
        "mean_signed_error_ms": None, "median_signed_error_ms": None,
    }
    return metrics


def _top_errors(pairs: Sequence[tuple[dict[str, Any], dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    for raw, pred in pairs:
        rows.append({
            "sample_id": pred["sample_id"],
            "fen": pred["fen"],
            "remaining_time_ms": pred["remaining_time_ms"],
            "teacher_time_ms": pred["teacher_time_ms"],
            "prediction_ms": pred["prediction_ms"],
            "absolute_error_ms": abs(float(pred["prediction_ms"]) - float(pred["teacher_time_ms"])),
            "teacher_depth": pred["teacher_depth"],
            "confidence": pred["teacher_confidence"],
            "completed_depth": pred["completed_depth"],
            "total_nodes": pred["total_nodes"],
            "timed_out": pred["timed_out"],
        })
    return sorted(rows, key=lambda row: (-row["absolute_error_ms"], row["sample_id"]))[:20]


def _probe_report(probes: Sequence[dict[str, Any]], *, baseline_ms: float) -> dict[str, Any]:
    by_budget = {}
    for budget in PROBE_BUDGETS_MS:
        rows = [row for row in probes if row.get("probe_budget_ms") == budget]
        attempted = [row for row in rows if not row.get("probe_skipped")]
        matched = [row for row in attempted if row.get("teacher_time_ms") is not None]
        target = [float(row["teacher_time_ms"]) for row in matched]
        prediction = [float(row["raw_prediction_ms"]) for row in matched]
        cost = [float(row["probe_cost_percent_of_clock"]) for row in attempted]
        by_budget[str(budget)] = {
            "sample_count": len(rows),
            "attempted_count": len(attempted),
            "skipped_count": len(rows) - len(attempted),
            "feature_availability_rate": len(attempted) / len(rows) if rows else None,
            "probe_elapsed_ms": distribution([float(row["probe_elapsed_ms"]) for row in attempted]),
            "completed_depth": distribution([float(row["completed_depth"]) for row in attempted]),
            "nodes": distribution([float(row["total_nodes"]) for row in attempted]),
            "probe_cost_percent_of_clock": distribution(cost),
            "probe_cost_percent_of_usable_clock": distribution([
                float(row["probe_cost_percent_of_usable_clock"])
                for row in attempted if row.get("probe_cost_percent_of_usable_clock") is not None
            ]),
            "prediction_error_vs_teacher": regression_metrics(target, prediction) if target else None,
            "training_median_baseline_error_vs_teacher": regression_metrics(
                target, [baseline_ms] * len(target)
            ) if target else None,
            "teacher_matched_probe_count": len(matched),
            "clock_groups": {
                str(clock): {
                    "sample_count": sum(row["remaining_time_ms"] == clock for row in rows),
                    "attempted_count": sum(row["remaining_time_ms"] == clock and not row.get("probe_skipped") for row in rows),
                    "median_probe_elapsed_ms": _median([float(row["probe_elapsed_ms"]) for row in rows if row["remaining_time_ms"] == clock and not row.get("probe_skipped")]),
                    "median_clock_cost_percent": _median([float(row["probe_cost_percent_of_clock"]) for row in rows if row["remaining_time_ms"] == clock and not row.get("probe_skipped")]),
                    "median_usable_clock_cost_percent": _median([float(row["probe_cost_percent_of_usable_clock"]) for row in rows if row["remaining_time_ms"] == clock and not row.get("probe_skipped") and row.get("probe_cost_percent_of_usable_clock") is not None]),
                    "teacher_matched_count": sum(row["remaining_time_ms"] == clock and row.get("teacher_time_ms") is not None and not row.get("probe_skipped") for row in rows),
                }
                for clock in sorted({int(row["remaining_time_ms"]) for row in rows}, reverse=True)
            },
        }
    return {
        "budgets_ms": list(PROBE_BUDGETS_MS),
        "source": "New offline probes on three held-out FEN groups; Phase 2F.1 probes were not reused because their FEN/clock pairs do not provide sufficient exact held-out Teacher matches.",
        "low_clock_values_evaluated_ms": list(LOW_CLOCKS_MS),
        "at_300ms_safety_result": {
            "usable_time_ms": max(300 - SAFETY_MARGIN_MS, 0),
            "probes_skipped": bool([row for row in probes if row.get("remaining_time_ms") == 300]) and all(
                row.get("probe_skipped") for row in probes if row.get("remaining_time_ms") == 300
            ),
            "operationally_meaningful": False,
            "reason": "The 300 ms safety reserve leaves zero usable time, so no pre-search probe is run at this clock.",
        },
        "by_probe_budget_ms": by_budget,
    }


def _safety_report(predictions: Sequence[float]) -> dict[str, Any]:
    examples = []
    for clock in (300, 299, 0):
        final, usable, capped = safety_governor(max(predictions, default=0.0), clock)
        examples.append({
            "remaining_time_ms": clock,
            "usable_time_ms": usable,
            "final_shadow_budget_ms": final,
            "cap_activated": capped,
            "nonnegative": final >= 0,
            "not_above_usable_clock": final <= usable,
        })
    total = len(predictions)
    governed = [safety_governor(float(raw), 300_000)[:2] for raw in predictions]
    return {
        "formula": "usable=max(remaining_time_ms-300,0); final=min(max(raw_prediction_ms,0),usable)",
        "safety_margin_ms": SAFETY_MARGIN_MS,
        "all_evaluated_predictions_nonnegative": all(value >= 0 for value in predictions),
        "all_final_budgets_nonnegative": all(final >= 0 for final, _ in governed),
        "all_final_budgets_at_or_below_usable": all(final <= usable for final, usable in governed),
        "boundary_examples_using_largest_heldout_prediction": examples,
    }


def _median(values: Sequence[float]) -> float | None:
    return float(statistics.median(values)) if values else None


def _recommendation_reason(
    checks: dict[str, bool],
    model_metrics: dict[str, float | int | None],
    baseline_metrics: dict[str, float | int | None],
    probe_metrics: dict[str, float | int | None] | None,
    probe_baseline: dict[str, float | int | None] | None,
) -> str:
    details = []
    if not checks["model_beats_training_median_baseline_on_mae"]:
        details.append(f"held-out MAE {model_metrics['mae_ms']:.2f} ms does not beat baseline {baseline_metrics['mae_ms']:.2f} ms")
    if not checks["model_beats_training_median_baseline_on_rmse"]:
        details.append(f"held-out RMSE {model_metrics['rmse_ms']:.2f} ms does not beat baseline {baseline_metrics['rmse_ms']:.2f} ms")
    if not checks["pre_search_25ms_probe_beats_matched_training_median"]:
        if probe_metrics and probe_baseline:
            details.append(
                f"25 ms pre-search probe MAE {probe_metrics['mae_ms']:.2f} ms exceeds its matched training-median baseline MAE {probe_baseline['mae_ms']:.2f} ms"
            )
        else:
            details.append("25 ms pre-search probe lacks enough matched Teacher labels for a baseline comparison")
    if not checks["heldout_fen_overlap_is_zero"]:
        details.append("training and held-out FEN groups overlap")
    if not checks["leakage_check_passed"]:
        details.append("the feature leakage audit failed")
    if not checks["small_probe_has_teacher_matched_predictions"]:
        details.append("no teacher-matched 25 ms probe prediction is available")
    if not checks["probe_skipped_at_300ms_after_reserve"]:
        details.append("the 300 ms reserve boundary was not verified as a skipped probe")
    return "Phase 2G is not yet justified: " + "; ".join(details or ["required validation evidence is missing"]) + "."


def _teacher_target(record: dict[str, Any]) -> float:
    value = record.get("teacher", {}).get("teacher_time_ms")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("A held-out labeled record must contain non-negative finite teacher_time_ms")
    return float(value)


def _remaining_clock(record: dict[str, Any]) -> int:
    value = record.get("clock", {}).get("remaining_time_ms")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("Record requires a non-negative integer remaining_time_ms")
    return value


def _required_str(record: dict[str, Any], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Record requires non-empty string {field!r}")
    return value


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run frozen-model Phase 2F.2 held-out validation without retraining.")
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate", help="Write the existing Phase 2C grouped holdout to a separate JSONL.")
    generate.add_argument("--input", default=str(DEFAULT_LABELED_DATASET))
    generate.add_argument("--evaluation", default=str(DEFAULT_MODEL_EVALUATION))
    generate.add_argument("--output", default=str(DEFAULT_HELDOUT_PATH))
    predict = commands.add_parser("predict", help="Run frozen Phase 2C RF inference on the holdout.")
    predict.add_argument("--heldout", default=str(DEFAULT_HELDOUT_PATH))
    predict.add_argument("--training", default=str(DEFAULT_LABELED_DATASET))
    predict.add_argument("--output", default=str(DEFAULT_PREDICTIONS_PATH))
    probe = commands.add_parser("probe", help="Run bounded offline probes and matched-label predictions.")
    probe.add_argument("--heldout", default=str(DEFAULT_HELDOUT_PATH))
    probe.add_argument("--output", default=str(DEFAULT_PROBES_PATH))
    report_parser = commands.add_parser("report", help="Write validation, slice, leakage, safety, and recommendation report.")
    report_parser.add_argument("--heldout", default=str(DEFAULT_HELDOUT_PATH))
    report_parser.add_argument("--training", default=str(DEFAULT_LABELED_DATASET))
    report_parser.add_argument("--predictions", default=str(DEFAULT_PREDICTIONS_PATH))
    report_parser.add_argument("--probes", default=str(DEFAULT_PROBES_PATH))
    report_parser.add_argument("--evaluation", default=str(DEFAULT_MODEL_EVALUATION))
    report_parser.add_argument("--output", default=str(DEFAULT_REPORT_PATH))
    args = parser.parse_args(argv)

    if args.command == "generate":
        _, heldout, audit = create_heldout_dataset(args.input, args.evaluation)
        write_jsonl(args.output, heldout)
        print(json.dumps(audit, indent=2))
        print(f"Held-out dataset: {args.output}")
    elif args.command == "predict":
        training, _, _ = create_heldout_dataset(args.training)
        rows = predict_heldout(load_jsonl(args.heldout), training)
        write_jsonl(args.output, rows)
        print(f"Frozen-model predictions: {len(rows)} rows -> {args.output}")
    elif args.command == "probe":
        rows = run_probe_experiment(load_jsonl(args.heldout))
        write_jsonl(args.output, rows)
        print(f"Offline pre-search probe rows: {len(rows)} -> {args.output}")
    else:
        training, heldout_from_split, audit = create_heldout_dataset(args.training, args.evaluation)
        heldout = load_jsonl(args.heldout)
        predictions = load_jsonl(args.predictions)
        probes = load_jsonl(args.probes) if Path(args.probes).is_file() else []
        if {row["sample_id"] for row in heldout} != {row["sample_id"] for row in heldout_from_split}:
            raise ValueError("Held-out dataset file does not match the saved Phase 2C grouped split")
        result = build_phase2f2_report(heldout, predictions, probes, training, audit)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps(result["evidence_table"], indent=2, allow_nan=False))
        print(json.dumps(result["phase2g_recommendation"], indent=2, allow_nan=False))
        print(f"Phase 2F.2 report: {output}")


if __name__ == "__main__":
    main()
