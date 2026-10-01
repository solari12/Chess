"""Train and evaluate Phase 2C grouped baseline regressors."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
from typing import Any, Sequence

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import GroupShuffleSplit

from app.learning.time_management.model_features import (
    FEATURE_AUDIT,
    FEATURE_NAMES,
    FORBIDDEN_FEATURE_FIELDS,
    extract_model_features,
)

DEFAULT_INPUT = "data/time_management/validation_training_dataset.jsonl"
DEFAULT_OUTPUT_ROOT = "data/time_management"
DEFAULT_RANDOM_STATE = 42
TEST_SIZE = 0.2

MODEL_CONFIG = {
    "gradient_boosting": {
        "class": "sklearn.ensemble.GradientBoostingRegressor",
        "n_estimators": 80,
        "learning_rate": 0.05,
        "max_depth": 2,
        "loss": "huber",
        "random_state": DEFAULT_RANDOM_STATE,
    },
    "random_forest": {
        "class": "sklearn.ensemble.RandomForestRegressor",
        "n_estimators": 160,
        "max_depth": 6,
        "min_samples_leaf": 2,
        "max_features": 0.8,
        "n_jobs": -1,
        "random_state": DEFAULT_RANDOM_STATE,
    },
}


def load_labeled_records(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number}: {error}") from error
            if record.get("teacher", {}).get("label_status") == "labeled":
                target = record["teacher"].get("teacher_time_ms")
                if isinstance(target, bool) or not isinstance(target, (int, float)) or not math.isfinite(target):
                    raise ValueError(f"Labeled record {record.get('sample_id')} has an invalid teacher_time_ms")
                if target < 0:
                    raise ValueError("teacher_time_ms cannot be negative")
                records.append(record)
    if not records:
        raise ValueError("No labeled records found")
    return records


def build_feature_matrix(records: Sequence[dict[str, Any]]) -> np.ndarray:
    if not records:
        raise ValueError("At least one record is required")
    matrix = np.vstack([extract_model_features(record) for record in records])
    if matrix.shape != (len(records), len(FEATURE_NAMES)):
        raise ValueError("Feature matrix shape does not match feature metadata")
    return matrix


def train_and_evaluate(
    records: Sequence[dict[str, Any]],
    *,
    random_state: int = DEFAULT_RANDOM_STATE,
    test_size: float = TEST_SIZE,
) -> dict[str, Any]:
    """Fit both tree regressors and a train-median baseline using FEN groups."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    fen_groups = np.asarray([record["fen"] for record in records], dtype=object)
    unique_group_count = len(set(fen_groups.tolist()))
    if unique_group_count < 2:
        raise ValueError("Grouped train/test split requires at least two unique FEN groups")

    features = build_feature_matrix(records)
    targets = np.asarray([float(record["teacher"]["teacher_time_ms"]) for record in records], dtype=np.float64)
    if not np.isfinite(targets).all():
        raise ValueError("Targets must be finite")

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_indices, test_indices = next(splitter.split(features, targets, groups=fen_groups))
    train_fens = set(fen_groups[train_indices].tolist())
    test_fens = set(fen_groups[test_indices].tolist())
    if train_fens.intersection(test_fens):
        raise AssertionError("Grouped split leaked FEN groups between train and test")
    if len(train_indices) == 0 or len(test_indices) < 2:
        raise ValueError("Grouped split did not produce enough train/test records for metrics")

    train_targets = targets[train_indices]
    test_targets = targets[test_indices]
    median_prediction = float(statistics.median(train_targets.tolist()))
    models = {
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=80,
            learning_rate=0.05,
            max_depth=2,
            loss="huber",
            random_state=random_state,
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=160,
            max_depth=6,
            min_samples_leaf=2,
            max_features=0.8,
            n_jobs=-1,
            random_state=random_state,
        ),
    }
    models["gradient_boosting"].fit(features[train_indices], train_targets)
    models["random_forest"].fit(features[train_indices], train_targets)

    predictions: dict[str, np.ndarray] = {
        "median_baseline": np.full(len(test_indices), median_prediction, dtype=np.float64),
        **{
            name: model.predict(features[test_indices]).astype(np.float64)
            for name, model in models.items()
        },
    }
    for name, predicted in predictions.items():
        if predicted.shape != test_targets.shape or not np.isfinite(predicted).all():
            raise ValueError(f"{name} returned an invalid prediction array")

    model_results: dict[str, Any] = {}
    test_records = [records[index] for index in test_indices]
    for name, predicted in predictions.items():
        model_results[name] = {
            "metrics": _regression_metrics(test_targets, predicted),
            "prediction_distribution": _distribution(predicted),
            "clock_group_errors": _clock_group_errors(test_records, test_targets, predicted),
        }

    importance = {
        name: {
            feature_name: float(value)
            for feature_name, value in zip(FEATURE_NAMES, model.feature_importances_)
        }
        for name, model in models.items()
    }
    target_stats = _distribution(targets)
    return {
        "models": models,
        "evaluation": {
            "phase": "2C",
            "dataset": DEFAULT_INPUT,
            "target": "teacher_time_ms",
            "split_method": "grouped_by_fen",
            "random_state": random_state,
            "test_size": test_size,
            "total_fen_groups": unique_group_count,
            "train_fen_groups": len(train_fens),
            "test_fen_groups": len(test_fens),
            "train_records": len(train_indices),
            "test_records": len(test_indices),
            "fen_group_overlap": len(train_fens.intersection(test_fens)),
            "target_statistics_ms": target_stats,
            "median_baseline_value_ms": median_prediction,
            "metrics": model_results,
            "feature_importance": importance,
            "clock_target_variation": _clock_target_variation(records),
            "clock_is_a_feature": "remaining_time_ms" in FEATURE_NAMES,
            "clock_awareness_demonstrated_by_targets": _clock_target_variation(records)["median_range_ms"] > 0,
            "feature_importance_note": "Exploratory model diagnostics, not causal explanations.",
            "model_config": _model_config(random_state),
            "feature_names": list(FEATURE_NAMES),
            "sklearn_version": sklearn.__version__,
        },
    }


def save_artifacts(
    training_result: dict[str, Any],
    *,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    dataset_path: str = DEFAULT_INPUT,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, str]:
    """Save versioned joblib models and reports without overwriting artifacts."""
    root = Path(output_root)
    models_root = root / "models"
    models_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    artifact_directory = models_root / f"phase2c_{timestamp}_seed{random_state}"
    artifact_directory.mkdir(exist_ok=False)

    evaluation_path = root / "model_evaluation.json"
    features_path = root / "model_features.json"
    if evaluation_path.exists() or features_path.exists():
        artifact_directory.rmdir()
        raise FileExistsError(
            "model_evaluation.json or model_features.json already exists; move/archive them before another run"
        )

    feature_metadata = {
        "phase": "2C",
        "dataset": dataset_path,
        "target": "teacher_time_ms",
        "feature_names": list(FEATURE_NAMES),
        "feature_audit": FEATURE_AUDIT,
        "excluded_teacher_fields": sorted(FORBIDDEN_FEATURE_FIELDS),
        "side_to_move_encoding": {"white": 1.0, "black": 0.0},
        "search_telemetry_availability": (
            "Search summary features require an existing bounded probe search. "
            "Depth profiles and teacher-derived transition summaries are excluded."
        ),
        "model_artifact_directory": str(artifact_directory),
        "sklearn_version": sklearn.__version__,
    }
    metadata = {
        "phase": "2C",
        "dataset": dataset_path,
        "target": "teacher_time_ms",
        "split_method": "grouped_by_fen",
        "random_state": random_state,
        "feature_names": list(FEATURE_NAMES),
        "model_config": _model_config(random_state),
        "sklearn_version": sklearn.__version__,
        "serialization": "joblib; load with the same scikit-learn version",
    }
    model_paths: dict[str, str] = {}
    for name, model in training_result["models"].items():
        model_path = artifact_directory / f"{name}.joblib"
        joblib.dump(model, model_path)
        model_paths[name] = str(model_path)
    metadata["model_paths"] = model_paths
    (artifact_directory / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_json_exclusive(evaluation_path, training_result["evaluation"])
    _write_json_exclusive(features_path, feature_metadata)
    return {
        "model_directory": str(artifact_directory),
        "gradient_boosting": model_paths["gradient_boosting"],
        "random_forest": model_paths["random_forest"],
        "metadata": str(artifact_directory / "metadata.json"),
        "evaluation": str(evaluation_path),
        "features": str(features_path),
    }


def _regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    values = {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(math.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
        "median_absolute_error": float(median_absolute_error(actual, predicted)),
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("Evaluation produced a non-finite metric")
    return values


def _clock_group_errors(
    test_records: Sequence[dict[str, Any]],
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, dict[str, float | int | None]]:
    groups: dict[str, list[float]] = {}
    for record, truth, estimate in zip(test_records, actual, predicted):
        clock = str(int(record["clock"]["remaining_time_ms"]))
        groups.setdefault(clock, []).append(abs(float(truth) - float(estimate)))
    return {
        clock: {
            "sample_count": len(errors),
            "mae": float(statistics.fmean(errors)) if errors else None,
            "median_absolute_error": float(statistics.median(errors)) if errors else None,
        }
        for clock, errors in sorted(groups.items(), key=lambda item: int(item[0]), reverse=True)
    }


def _clock_target_variation(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    values: dict[str, list[float]] = {}
    for record in records:
        clock = str(int(record["clock"]["remaining_time_ms"]))
        values.setdefault(clock, []).append(float(record["teacher"]["teacher_time_ms"]))
    medians = {clock: float(statistics.median(targets)) for clock, targets in sorted(values.items())}
    return {
        "median_target_by_clock_ms": medians,
        "median_range_ms": max(medians.values()) - min(medians.values()) if medians else 0.0,
    }


def _distribution(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
    }


def _model_config(random_state: int) -> dict[str, dict[str, Any]]:
    return {
        "gradient_boosting": {**MODEL_CONFIG["gradient_boosting"], "random_state": random_state},
        "random_forest": {**MODEL_CONFIG["random_forest"], "random_state": random_state},
    }


def _write_json_exclusive(path: Path, data: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as destination:
        destination.write(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False))
        destination.write("\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Phase 2C grouped baseline time-target regressors.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    records = load_labeled_records(args.input)
    training_result = train_and_evaluate(
        records,
        random_state=args.random_state,
        test_size=args.test_size,
    )
    paths = save_artifacts(
        training_result,
        output_root=args.output_root,
        dataset_path=args.input,
        random_state=args.random_state,
    )
    report = training_result["evaluation"]
    print("Phase 2C baseline experiment complete")
    print(f"FEN groups: {report['total_fen_groups']} total, {report['train_fen_groups']} train, {report['test_fen_groups']} test")
    print(f"Records: {report['train_records']} train, {report['test_records']} test")
    print(f"Target statistics (ms): {report['target_statistics_ms']}")
    for name, result in report["metrics"].items():
        print(f"{name}: {result['metrics']}")
    print(f"Artifacts: {paths}")
    print("This is a small baseline experiment, not a production model or evidence of clock awareness.")


if __name__ == "__main__":
    main()
