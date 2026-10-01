"""Phase 2D robustness, ablation, and error evaluation for baseline regressors."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
from typing import Any, Sequence

import numpy as np
from scipy.stats import rankdata
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

from app.learning.time_management.model_features import FEATURE_NAMES
from app.learning.time_management.train_models import DEFAULT_INPUT, TEST_SIZE, load_labeled_records

DEFAULT_SEEDS = (42, 43, 44, 45, 46)
CLOCKS_MS = (300_000, 120_000, 30_000, 10_000, 3_000)
FEATURE_GROUPS = {
    "full": tuple(FEATURE_NAMES),
    "no_probe": tuple(
        name
        for name in FEATURE_NAMES
        if name
        not in {
            "completed_depth",
            "total_nodes",
            "total_search_time_ms",
            "timed_out",
            "analysis_budget_ms",
            "max_analysis_depth",
        }
    ),
    "minimal": (
        "remaining_time_ms",
        "legal_move_count",
        "material_imbalance",
        "side_to_move_is_white",
        "side_in_check",
    ),
    "full_minus_completed_depth": tuple(name for name in FEATURE_NAMES if name != "completed_depth"),
}
REGRESSOR_GROUPS = ("full", "no_probe", "minimal", "full_minus_completed_depth")


def evaluate_dataset(
    records: Sequence[dict[str, Any]],
    *,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    test_size: float = TEST_SIZE,
    include_learning_curve: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run fixed-seed grouped evaluations and return report plus top error cases.

    Test predictions are evaluated per seed and never concatenated. Aggregate
    values below are summaries of per-seed metrics only.
    """
    _validate_feature_groups()
    if len(seeds) < 1 or len(set(seeds)) != len(seeds):
        raise ValueError("Provide one or more distinct random seeds")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    if not records:
        raise ValueError("No labeled records to evaluate")

    all_features = np.vstack([_feature_row(record) for record in records])
    targets = np.asarray([float(record["teacher"]["teacher_time_ms"]) for record in records], dtype=np.float64)
    fen_groups = np.asarray([record["fen"] for record in records], dtype=object)
    if not np.isfinite(all_features).all() or not np.isfinite(targets).all():
        raise ValueError("Features and targets must be finite")
    total_groups = len(set(fen_groups.tolist()))
    if total_groups < 2:
        raise ValueError("At least two FEN groups are required")

    feature_indices = {
        group: np.asarray([FEATURE_NAMES.index(name) for name in names], dtype=int)
        for group, names in FEATURE_GROUPS.items()
    }
    seed_results: list[dict[str, Any]] = []
    primary_errors: dict[str, Any] | None = None
    curve_seed_results: dict[str, list[dict[str, Any]]] = {"full": []}

    for seed in seeds:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_indices, test_indices = next(splitter.split(all_features, targets, groups=fen_groups))
        train_fens = set(fen_groups[train_indices].tolist())
        test_fens = set(fen_groups[test_indices].tolist())
        if train_fens.intersection(test_fens):
            raise AssertionError(f"FEN group leakage in seed {seed}")
        if len(test_indices) < 2:
            raise ValueError(f"Seed {seed} produced fewer than two test records")

        seed_result: dict[str, Any] = {
            "seed": int(seed),
            "split": {
                "total_fen_groups": total_groups,
                "train_fen_groups": len(train_fens),
                "test_fen_groups": len(test_fens),
                "train_records": len(train_indices),
                "test_records": len(test_indices),
                "fen_group_overlap": len(train_fens.intersection(test_fens)),
            },
            "feature_groups": {},
        }
        train_targets = targets[train_indices]
        test_targets = targets[test_indices]
        median_estimate = np.full(len(test_indices), float(statistics.median(train_targets.tolist())))
        full_predictions: dict[str, np.ndarray] = {"median_baseline": median_estimate}

        for feature_group in REGRESSOR_GROUPS:
            indices = feature_indices[feature_group]
            x_train = all_features[train_indices][:, indices]
            x_test = all_features[test_indices][:, indices]
            group_metrics: dict[str, Any] = {}
            for model_name in ("gradient_boosting", "random_forest"):
                model = _new_model(model_name, int(seed))
                model.fit(x_train, train_targets)
                prediction = np.asarray(model.predict(x_test), dtype=np.float64)
                if prediction.shape != test_targets.shape or not np.isfinite(prediction).all():
                    raise ValueError(f"{model_name}/{feature_group}/seed{seed} returned invalid predictions")
                group_metrics[model_name] = _model_report(test_targets, prediction)
                if feature_group == "full":
                    full_predictions[model_name] = prediction
                if feature_group == "full_minus_completed_depth":
                    full_model_report = seed_result["feature_groups"]["full"][model_name]["metrics"]
                    group_metrics[model_name]["metric_delta_vs_full"] = {
                        metric: group_metrics[model_name]["metrics"][metric] - full_model_report[metric]
                        for metric in ("mae", "rmse", "r2", "median_absolute_error")
                    }
            if feature_group == "full":
                group_metrics["median_baseline"] = _model_report(test_targets, median_estimate)
            seed_result["feature_groups"][feature_group] = group_metrics

        full_clock_errors = {
            model_name: _clock_errors(
                [records[index] for index in test_indices],
                test_targets,
                prediction,
            )
            for model_name, prediction in full_predictions.items()
        }
        seed_result["full_feature_clock_errors"] = full_clock_errors
        seed_result["full_feature_prediction_coverage"] = {
            model_name: _range_coverage(test_targets, prediction)
            for model_name, prediction in full_predictions.items()
        }
        seed_result["full_feature_residual_correlations"] = {
            model_name: _residual_correlations(
                [records[index] for index in test_indices], test_targets, prediction
            )
            for model_name, prediction in full_predictions.items()
        }

        if int(seed) == 42:
            primary_errors = {
                model_name: _largest_errors(
                    [records[index] for index in test_indices], test_targets, prediction, count=20
                )
                for model_name, prediction in full_predictions.items()
            }
        elif primary_errors is None:
            primary_errors = {
                model_name: _largest_errors(
                    [records[index] for index in test_indices], test_targets, prediction, count=20
                )
                for model_name, prediction in full_predictions.items()
            }
        if include_learning_curve:
            _evaluate_learning_curve(
                all_features,
                targets,
                fen_groups,
                train_indices,
                test_indices,
                int(seed),
                feature_indices,
                curve_seed_results,
            )
        seed_results.append(seed_result)

    report = {
        "phase": "2D",
        "dataset": DEFAULT_INPUT,
        "target": "teacher_time_ms",
        "seeds": [int(seed) for seed in seeds],
        "test_size": test_size,
        "total_records": len(records),
        "total_fen_groups": total_groups,
        "model_config": {
            "gradient_boosting": {
                "class": "sklearn.ensemble.GradientBoostingRegressor",
                "n_estimators": 80,
                "learning_rate": 0.05,
                "max_depth": 2,
                "loss": "huber",
            },
            "random_forest": {
                "class": "sklearn.ensemble.RandomForestRegressor",
                "n_estimators": 160,
                "max_depth": 6,
                "min_samples_leaf": 2,
                "max_features": 0.8,
                "n_jobs": 1,
            },
        },
        "feature_groups": {name: list(names) for name, names in FEATURE_GROUPS.items()},
        "seed_results": seed_results,
        "metric_aggregates": _aggregate_metrics(seed_results),
        "clock_error_aggregates": _aggregate_clock_errors(seed_results),
        "target_distribution_by_clock": _target_distribution_by_clock(records),
        "target_clock_variation": _target_clock_variation(records),
        "prediction_coverage_aggregates": _aggregate_coverage(seed_results),
        "learning_curve": _aggregate_learning_curve(curve_seed_results) if include_learning_curve else None,
        "interpretation_limits": [
            "Clock-group errors are prediction-error summaries; they do not show that the target depends on clock.",
            "The Phase 2B.1 teacher target had identical clock-group medians and its clock cap did not activate.",
            "Feature importance and residual correlations are descriptive and do not establish causality.",
            "Only 97 labeled FEN groups are available; clock variants are grouped and never cross a split.",
        ],
    }
    if primary_errors is None:
        raise AssertionError("No primary error analysis was produced")
    return report, primary_errors


def write_evaluation_artifacts(
    report: dict[str, Any],
    error_cases: dict[str, Any],
    *,
    output_root: str | Path = "data/time_management",
) -> dict[str, str]:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "phase2d_evaluation.json"
    errors_path = root / "phase2d_error_cases.json"
    learning_curve_path = root / "phase2d_learning_curve.json"
    paths = [report_path, errors_path]
    if report.get("learning_curve") is not None:
        paths.append(learning_curve_path)
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing Phase 2D output: {', '.join(existing)}")

    _write_json_exclusive(report_path, report)
    _write_json_exclusive(errors_path, error_cases)
    if report.get("learning_curve") is not None:
        _write_json_exclusive(learning_curve_path, report["learning_curve"])
    return {
        "evaluation": str(report_path),
        "error_cases": str(errors_path),
        **({"learning_curve": str(learning_curve_path)} if report.get("learning_curve") is not None else {}),
    }


def _feature_row(record: dict[str, Any]) -> np.ndarray:
    from app.learning.time_management.model_features import extract_model_features

    return extract_model_features(record)


def _new_model(name: str, seed: int):
    if name == "gradient_boosting":
        return GradientBoostingRegressor(
            n_estimators=80,
            learning_rate=0.05,
            max_depth=2,
            loss="huber",
            random_state=seed,
        )
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=160,
            max_depth=6,
            min_samples_leaf=2,
            max_features=0.8,
            # Serial fit avoids tiny parallel floating-point summation drift
            # between repeated fixed-seed validation runs.
            n_jobs=1,
            random_state=seed,
        )
    raise ValueError(f"Unknown model: {name}")


def _model_report(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    errors = np.abs(actual - predicted)
    metrics = {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(math.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
        "median_absolute_error": float(median_absolute_error(actual, predicted)),
    }
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError("Evaluation metrics must be finite")
    return {
        "metrics": metrics,
        "prediction_distribution": _distribution(predicted),
        "absolute_error_distribution": _distribution(errors),
    }


def _clock_errors(
    records: Sequence[dict[str, Any]], actual: np.ndarray, predicted: np.ndarray
) -> dict[str, dict[str, float | int]]:
    groups: dict[str, list[float]] = {}
    for record, truth, estimate in zip(records, actual, predicted):
        clock = str(int(record["clock"]["remaining_time_ms"]))
        groups.setdefault(clock, []).append(abs(float(truth) - float(estimate)))
    return {
        clock: {
            "sample_count": len(values),
            "mae": float(statistics.fmean(values)),
            "median_absolute_error": float(statistics.median(values)),
        }
        for clock, values in sorted(groups.items(), key=lambda item: int(item[0]), reverse=True)
    }


def _target_distribution_by_clock(records: Sequence[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    groups: dict[str, list[float]] = {}
    for record in records:
        clock = str(int(record["clock"]["remaining_time_ms"]))
        groups.setdefault(clock, []).append(float(record["teacher"]["teacher_time_ms"]))
    result = {}
    for clock in sorted(groups, key=int, reverse=True):
        values = np.asarray(groups[clock], dtype=np.float64)
        result[clock] = {"count": len(values), **_distribution(values), "p75": _quantile(values, 0.75), "p90": _quantile(values, 0.9)}
    return result


def _target_clock_variation(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    distribution = _target_distribution_by_clock(records)
    medians = {clock: metrics["median"] for clock, metrics in distribution.items()}
    return {
        "median_target_by_clock_ms": medians,
        "median_range_ms": max(medians.values()) - min(medians.values()) if medians else 0.0,
        "target_distributions_identical_by_median": len(set(medians.values())) <= 1,
    }


def _range_coverage(all_targets: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    minimum = float(np.min(all_targets))
    maximum = float(np.max(all_targets))
    return {
        "prediction_min": float(np.min(predictions)),
        "prediction_max": float(np.max(predictions)),
        "target_min": minimum,
        "target_max": maximum,
        "percent_below_target_min": float(np.mean(predictions < minimum) * 100),
        "percent_above_target_max": float(np.mean(predictions > maximum) * 100),
    }


def _largest_errors(
    records: Sequence[dict[str, Any]], actual: np.ndarray, predicted: np.ndarray, *, count: int
) -> list[dict[str, Any]]:
    errors = np.abs(actual - predicted)
    order = np.argsort(errors)[::-1][:count]
    legal_counts = np.asarray([record["position_features"]["legal_move_count"] for record in records])
    capture_counts = np.asarray([record["position_features"]["capture_count"] for record in records])
    checking_counts = np.asarray([record["position_features"]["checking_move_count"] for record in records])
    material_imbalances = np.asarray([abs(record["position_features"]["material_imbalance"]) for record in records])
    legal_low = float(np.quantile(legal_counts, 0.25))
    legal_high = float(np.quantile(legal_counts, 0.75))
    capture_high = float(np.quantile(capture_counts, 0.75))
    checking_high = float(np.quantile(checking_counts, 0.75))
    material_high = float(np.quantile(material_imbalances, 0.75))
    clocks = [int(record["clock"]["remaining_time_ms"]) for record in records]
    low_clock, high_clock = min(clocks), max(clocks)
    cases = []
    for index in order:
        record = records[int(index)]
        position = record["position_features"]
        search = record["search"]
        legal_count = int(position["legal_move_count"])
        material_imbalance = abs(float(position["material_imbalance"]))
        clock = int(record["clock"]["remaining_time_ms"])
        cases.append(
            {
                "sample_id": record["sample_id"],
                "fen": record["fen"],
                "remaining_time_ms": clock,
                "teacher_time_ms": float(actual[index]),
                "prediction": float(predicted[index]),
                "absolute_error": float(errors[index]),
                "completed_depth": int(search["completed_depth"]),
                "total_nodes": int(search["total_nodes"]),
                "legal_move_count": legal_count,
                "material_imbalance": float(position["material_imbalance"]),
                "capture_count": int(position["capture_count"]),
                "checking_move_count": int(position["checking_move_count"]),
                "timed_out": bool(search["timed_out"]),
                "pattern_indicators": {
                    "low_legal_move_count_q1": legal_count <= legal_low,
                    "high_legal_move_count_q3": legal_count >= legal_high,
                    "high_absolute_material_imbalance_q3": material_imbalance >= material_high,
                    "high_capture_count_q3": int(position["capture_count"]) >= capture_high,
                    "high_checking_move_count_q3": int(position["checking_move_count"]) >= checking_high,
                    "extreme_clock_in_test_set": clock in {low_clock, high_clock},
                    "timed_out": bool(search["timed_out"]),
                },
            }
        )
    return cases


def _residual_correlations(
    records: Sequence[dict[str, Any]], actual: np.ndarray, predicted: np.ndarray
) -> dict[str, dict[str, float | None]]:
    absolute_error = np.abs(actual - predicted)
    columns = {
        "teacher_time_ms": actual,
        "remaining_time_ms": np.asarray([r["clock"]["remaining_time_ms"] for r in records], dtype=float),
        "legal_move_count": np.asarray([r["position_features"]["legal_move_count"] for r in records], dtype=float),
        "completed_depth": np.asarray([r["search"]["completed_depth"] for r in records], dtype=float),
        "total_nodes": np.asarray([r["search"]["total_nodes"] for r in records], dtype=float),
    }
    return {
        name: {
            "pearson": _correlation(absolute_error, values),
            "spearman": _correlation(rankdata(absolute_error), rankdata(values)),
        }
        for name, values in columns.items()
    }


def _correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2 or np.ptp(left) == 0 or np.ptp(right) == 0:
        return None
    value = float(np.corrcoef(left, right)[0, 1])
    return value if math.isfinite(value) else None


def _evaluate_learning_curve(
    all_features: np.ndarray,
    targets: np.ndarray,
    fen_groups: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    seed: int,
    feature_indices: dict[str, np.ndarray],
    output: dict[str, list[dict[str, Any]]],
) -> None:
    train_fens = np.asarray(sorted(set(fen_groups[train_indices].tolist())), dtype=object)
    permutation = np.random.default_rng(seed).permutation(train_fens)
    fixed_test = test_indices
    for fraction in (0.25, 0.5, 0.75, 1.0):
        selected_count = max(1, int(math.ceil(len(permutation) * fraction)))
        selected_fens = set(permutation[:selected_count].tolist())
        subset_indices = np.asarray(
            [index for index in train_indices if fen_groups[index] in selected_fens], dtype=int
        )
        for model_group in ("full",):
            feature_subset = feature_indices[model_group]
            for model_name in ("gradient_boosting", "random_forest"):
                model = _new_model(model_name, seed)
                model.fit(all_features[subset_indices][:, feature_subset], targets[subset_indices])
                prediction = model.predict(all_features[fixed_test][:, feature_subset]).astype(float)
                metrics = _basic_metrics(targets[fixed_test], prediction)
                output[model_group].append(
                    {
                        "seed": seed,
                        "training_fraction": fraction,
                        "training_fen_groups": selected_count,
                        "training_records": len(subset_indices),
                        "model": model_name,
                        "metrics": metrics,
                    }
                )


def _basic_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(math.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
        "median_absolute_error": float(median_absolute_error(actual, predicted)),
    }


def _aggregate_metrics(seed_results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    values: dict[str, dict[str, list[float]]] = {}
    for result in seed_results:
        for group, models in result["feature_groups"].items():
            for model_name, model_result in models.items():
                key = f"{group}/{model_name}"
                values.setdefault(key, {metric: [] for metric in model_result["metrics"]})
                for metric, value in model_result["metrics"].items():
                    values[key][metric].append(float(value))
    return {
        key: {metric: _aggregate_distribution(metric_values) for metric, metric_values in metrics.items()}
        for key, metrics in values.items()
    }


def _aggregate_clock_errors(seed_results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    gathered: dict[str, dict[str, dict[str, list[float]]]] = {}
    for seed_result in seed_results:
        for model_name, clocks in seed_result["full_feature_clock_errors"].items():
            for clock, metrics in clocks.items():
                model_groups = gathered.setdefault(model_name, {}).setdefault(clock, {"sample_count": [], "mae": [], "median_absolute_error": []})
                for metric, value in metrics.items():
                    model_groups[metric].append(float(value))
    return {
        model: {
            clock: {metric: _aggregate_distribution(values) for metric, values in metrics.items()}
            for clock, metrics in sorted(clocks.items(), key=lambda item: int(item[0]), reverse=True)
        }
        for model, clocks in gathered.items()
    }


def _aggregate_coverage(seed_results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    gathered: dict[str, dict[str, list[float]]] = {}
    for seed_result in seed_results:
        for model_name, values in seed_result["full_feature_prediction_coverage"].items():
            model_metrics = gathered.setdefault(model_name, {key: [] for key in values})
            for key, value in values.items():
                model_metrics[key].append(float(value))
    return {
        model: {metric: _aggregate_distribution(values) for metric, values in metrics.items()}
        for model, metrics in gathered.items()
    }


def _aggregate_learning_curve(seed_results: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for feature_group, entries in seed_results.items():
        grouped: dict[str, dict[str, dict[str, list[float]]]] = {}
        for entry in entries:
            fraction = str(int(entry["training_fraction"] * 100))
            model = entry["model"]
            metrics = grouped.setdefault(fraction, {}).setdefault(model, {key: [] for key in entry["metrics"]})
            for key, value in entry["metrics"].items():
                metrics[key].append(float(value))
        result[feature_group] = {
            fraction: {
                model: {metric: _aggregate_distribution(values) for metric, values in metrics.items()}
                for model, metrics in models.items()
            }
            for fraction, models in sorted(grouped.items(), key=lambda item: int(item[0]))
        }
    return result


def _aggregate_distribution(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "standard_deviation": float(statistics.pstdev(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _distribution(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
    }


def _quantile(values: np.ndarray, quantile: float) -> float:
    return float(np.quantile(values, quantile))


def _write_json_exclusive(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as output:
        output.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
        output.write("\n")


def _validate_feature_groups() -> None:
    if FEATURE_GROUPS["full"] != tuple(FEATURE_NAMES):
        raise AssertionError("The full feature group must match the trained feature vector")
    prohibited_probe = {"completed_depth", "total_nodes", "total_search_time_ms", "timed_out", "analysis_budget_ms", "max_analysis_depth"}
    if prohibited_probe.intersection(FEATURE_GROUPS["no_probe"]):
        raise AssertionError("The no-probe feature group contains probe-result/configuration fields")
    if set(FEATURE_GROUPS["minimal"]) != {
        "remaining_time_ms",
        "legal_move_count",
        "material_imbalance",
        "side_to_move_is_white",
        "side_in_check",
    }:
        raise AssertionError("Minimal feature group differs from the requested position+clock set")
    if "completed_depth" in FEATURE_GROUPS["full_minus_completed_depth"]:
        raise AssertionError("Completed-depth ablation still includes completed_depth")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 2D grouped robustness and ablation evaluation.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output-root", default="data/time_management")
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    parser.add_argument("--skip-learning-curve", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    records = load_labeled_records(args.input)
    report, errors = evaluate_dataset(
        records,
        seeds=seeds,
        test_size=args.test_size,
        include_learning_curve=not args.skip_learning_curve,
    )
    report["dataset"] = args.input
    paths = write_evaluation_artifacts(report, errors, output_root=args.output_root)
    print("Phase 2D evaluation complete")
    print(f"FEN groups: {report['total_fen_groups']}; labeled records: {report['total_records']}")
    print(json.dumps(report["metric_aggregates"], ensure_ascii=False, indent=2))
    print(f"Artifacts: {paths}")


if __name__ == "__main__":
    main()
