"""Deterministic heuristic teacher and target generator for Phase 2B.

This module labels existing Phase 2A observations. It does not search positions,
train a model, or claim that its time targets are mathematically optimal.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import statistics
from typing import Any, Sequence


@dataclass(frozen=True)
class TeacherConfig:
    """Visible and configurable parameters for the offline heuristic.

    Move changes receive a larger weight because they directly show that the
    selected decision changed. Score changes use a saturating normalization,
    so raw centipawn values cannot dominate. Node growth modestly discounts
    information gain as a hardware-independent proxy for added search cost.
    """

    score_scale: float = 100.0
    move_change_weight: float = 0.65
    score_change_weight: float = 0.35
    cost_weight: float = 0.25
    min_gain_for_continue: float = 0.12
    clock_safety_margin_ms: float = 300.0
    confidence_transition_scale: int = 3
    timeout_confidence_multiplier: float = 0.8

    def validate(self) -> None:
        numeric_values = (
            self.score_scale,
            self.move_change_weight,
            self.score_change_weight,
            self.cost_weight,
            self.min_gain_for_continue,
            self.clock_safety_margin_ms,
            self.timeout_confidence_multiplier,
        )
        if not all(math.isfinite(value) for value in numeric_values):
            raise ValueError("Teacher configuration values must be finite")
        if self.score_scale <= 0:
            raise ValueError("score_scale must be positive")
        if self.move_change_weight < 0 or self.score_change_weight < 0:
            raise ValueError("Signal weights cannot be negative")
        if self.move_change_weight + self.score_change_weight <= 0:
            raise ValueError("At least one information signal weight must be positive")
        if not 0 <= self.cost_weight <= 1:
            raise ValueError("cost_weight must be between 0 and 1")
        if not 0 <= self.min_gain_for_continue <= 1:
            raise ValueError("min_gain_for_continue must be between 0 and 1")
        if self.clock_safety_margin_ms < 0:
            raise ValueError("clock_safety_margin_ms cannot be negative")
        if self.confidence_transition_scale < 1:
            raise ValueError("confidence_transition_scale must be at least 1")
        if not 0 <= self.timeout_confidence_multiplier <= 1:
            raise ValueError("timeout_confidence_multiplier must be between 0 and 1")


def calculate_depth_transition(
    previous: dict[str, Any],
    current: dict[str, Any],
    config: TeacherConfig = TeacherConfig(),
) -> dict[str, Any]:
    """Compare consecutive depth observations and compute normalized signals."""
    config.validate()
    previous_depth = _required_int(previous, "depth")
    current_depth = _required_int(current, "depth")
    if current_depth != previous_depth + 1:
        raise ValueError("Depth profile transitions must be consecutive")

    previous_move = _required_str(previous, "move")
    current_move = _required_str(current, "move")
    previous_score = _required_number(previous, "score")
    current_score = _required_number(current, "score")
    previous_time = _required_number(previous, "time_ms")
    current_time = _required_number(current, "time_ms")
    previous_nodes = _required_int(previous, "nodes")
    current_nodes = _required_int(current, "nodes")
    if min(previous_time, current_time, previous_nodes, current_nodes) < 0:
        raise ValueError("Depth profile times and node counts must be non-negative")

    move_changed = previous_move != current_move
    score_delta = current_score - previous_score
    absolute_score_delta = abs(score_delta)
    evaluation_signal = absolute_score_delta / (config.score_scale + absolute_score_delta)
    decision_signal = 1.0 if move_changed else 0.0

    additional_time_ms = max(0.0, current_time - previous_time)
    additional_nodes = max(0, current_nodes - previous_nodes)
    time_growth_ratio = _safe_ratio(current_time, previous_time)
    node_growth_ratio = _safe_ratio(current_nodes, previous_nodes)
    # Added nodes are a machine-independent cost proxy. It is bounded [0, 1),
    # and safely handles a zero-node previous iteration.
    cost_signal = additional_nodes / (previous_nodes + additional_nodes) if additional_nodes else 0.0

    weight_total = config.move_change_weight + config.score_change_weight
    raw_gain = (
        config.move_change_weight * decision_signal
        + config.score_change_weight * evaluation_signal
    ) / weight_total
    information_gain = raw_gain * (1.0 - config.cost_weight * cost_signal)

    return {
        "from_depth": previous_depth,
        "to_depth": current_depth,
        "best_move_changed": move_changed,
        "score_delta": score_delta,
        "absolute_score_delta": absolute_score_delta,
        "additional_time_ms": additional_time_ms,
        "additional_nodes": additional_nodes,
        "time_growth_ratio": time_growth_ratio,
        "node_growth_ratio": node_growth_ratio,
        "decision_change_signal": decision_signal,
        "evaluation_change_signal": evaluation_signal,
        "cost_signal": cost_signal,
        "information_gain_score": information_gain,
    }


def calculate_information_gain(transition: dict[str, Any], config: TeacherConfig = TeacherConfig()) -> float:
    """Return a bounded, interpretable 0..1 information-gain score."""
    config.validate()
    decision = _required_number(transition, "decision_change_signal")
    evaluation = _required_number(transition, "evaluation_change_signal")
    cost = _required_number(transition, "cost_signal")
    if not (0 <= decision <= 1 and 0 <= evaluation <= 1 and 0 <= cost <= 1):
        raise ValueError("Transition signals must be normalized to the range 0..1")
    weight_total = config.move_change_weight + config.score_change_weight
    raw_gain = (
        config.move_change_weight * decision
        + config.score_change_weight * evaluation
    ) / weight_total
    return raw_gain * (1.0 - config.cost_weight * cost)


def generate_target(record: dict[str, Any], config: TeacherConfig = TeacherConfig()) -> dict[str, Any]:
    """Create derived teacher features and a clock-capped target for one record."""
    config.validate()
    profiles = record.get("depth_profiles")
    if not isinstance(profiles, list):
        raise ValueError("record.depth_profiles must be a list")
    remaining_time = _required_number(record.get("clock", {}), "remaining_time_ms")
    if remaining_time < 0:
        raise ValueError("remaining_time_ms must be non-negative")

    transitions = [
        calculate_depth_transition(previous, current, config)
        for previous, current in zip(profiles, profiles[1:])
    ]
    gains = [transition["information_gain_score"] for transition in transitions]
    score_deltas = [transition["absolute_score_delta"] for transition in transitions]
    move_change_count = sum(transition["best_move_changed"] for transition in transitions)
    average_gain = statistics.fmean(gains) if gains else 0.0
    stability = max(0.0, min(1.0, 1.0 - average_gain))
    significant_depths = [
        transition["to_depth"]
        for transition in transitions
        if transition["information_gain_score"] >= config.min_gain_for_continue
    ]

    teacher_depth: int | None = None
    label_reason: str | None = None
    if len(profiles) < 2:
        label_reason = "fewer_than_two_completed_depths"
    else:
        # Require at least one observed transition. If it is already stable,
        # use depth 2 as a confirmation point. Otherwise continue through
        # meaningful gains and stop before the first low-gain transition.
        teacher_depth = 2
        saw_significant_gain = False
        for transition in transitions:
            gain = transition["information_gain_score"]
            if gain >= config.min_gain_for_continue:
                teacher_depth = transition["to_depth"]
                saw_significant_gain = True
            elif not saw_significant_gain:
                teacher_depth = transition["to_depth"]
                break
            else:
                break

    usable_clock_ms = max(0.0, remaining_time - config.clock_safety_margin_ms)
    if teacher_depth is not None and usable_clock_ms <= 0:
        teacher_depth = None
        label_reason = "no_usable_clock_after_safety_reserve"
    if teacher_depth is not None and _required_number(profiles[teacher_depth - 1], "time_ms") <= 0:
        # A zero duration means the source timer could not resolve this search;
        # treating it as a zero-ms target would fabricate a training label.
        teacher_depth = None
        label_reason = "selected_depth_time_unmeasurable"

    timeout = bool(record.get("search", {}).get("timed_out", False))
    if teacher_depth is None:
        teacher_time_ms: float | None = None
        confidence = 0.0
        label_status = "insufficient_data"
    else:
        observed_time = _required_number(profiles[teacher_depth - 1], "time_ms")
        teacher_time_ms = min(max(0.0, observed_time), usable_clock_ms)
        evidence = min(1.0, len(transitions) / config.confidence_transition_scale)
        confidence = evidence * (0.5 + 0.5 * stability)
        if timeout:
            confidence *= config.timeout_confidence_multiplier
        confidence = max(0.0, min(1.0, confidence))
        label_status = "labeled"

    return {
        "teacher_depth": teacher_depth,
        "teacher_time_ms": teacher_time_ms,
        "teacher_confidence": confidence,
        "information_gain_score": average_gain,
        "selected_transition_gain": (
            transitions[teacher_depth - 2]["information_gain_score"]
            if teacher_depth is not None and teacher_depth >= 2
            else None
        ),
        "depth_transitions": transitions,
        "best_move_change_count": move_change_count,
        "max_score_delta": max(score_deltas, default=0.0),
        "average_score_delta": statistics.fmean(score_deltas) if score_deltas else 0.0,
        "last_significant_depth": max(significant_depths, default=None),
        "search_stability": stability,
        "usable_clock_ms": usable_clock_ms,
        "label_status": label_status,
        "label_reason": label_reason,
    }


def label_dataset(records: Sequence[dict[str, Any]], config: TeacherConfig = TeacherConfig()) -> list[dict[str, Any]]:
    """Return labeled copies; the Phase 2A input objects remain untouched."""
    config.validate()
    labeled_records: list[dict[str, Any]] = []
    for record in records:
        copied = json.loads(json.dumps(record, allow_nan=False))
        copied["teacher"] = generate_target(copied, config)
        labeled_records.append(copied)
    return labeled_records


def label_jsonl(input_path: str | Path, output_path: str | Path, config: TeacherConfig = TeacherConfig()) -> tuple[dict[str, Any], Path]:
    """Read raw JSONL, write a separate labeled JSONL, and summarize labels."""
    raw_records: list[dict[str, Any]] = []
    with Path(input_path).open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                raw_records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on input line {line_number}: {error}") from error

    labeled_records = label_dataset(raw_records, config)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as destination:
        for record in labeled_records:
            destination.write(json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":")))
            destination.write("\n")

    summary = summarize_labels(labeled_records, config)
    return summary, output


def summarize_labels(records: Sequence[dict[str, Any]], config: TeacherConfig = TeacherConfig()) -> dict[str, Any]:
    """Produce deterministic target, confidence, stability, and clock summaries."""
    labeled = [record for record in records if record["teacher"]["label_status"] == "labeled"]
    insufficient = [record for record in records if record["teacher"]["label_status"] != "labeled"]
    times = [record["teacher"]["teacher_time_ms"] for record in labeled]
    depths = [record["teacher"]["teacher_depth"] for record in labeled]
    confidences = [record["teacher"]["teacher_confidence"] for record in labeled]
    clock_counts: dict[str, int] = {}
    for record in records:
        clock = str(record["clock"]["remaining_time_ms"])
        clock_counts[clock] = clock_counts.get(clock, 0) + 1

    return {
        "raw_records": len(records),
        "labeled_samples": len(labeled),
        "insufficient_samples": len(insufficient),
        "label_rate": len(labeled) / len(records) if records else 0.0,
        "teacher_time_ms": _distribution(times),
        "teacher_depth": _distribution(depths),
        "teacher_confidence": _distribution(confidences),
        "clock_distribution": dict(sorted(clock_counts.items(), key=lambda item: int(item[0]))),
        "samples_where_best_move_changed": sum(
            record["teacher"]["best_move_change_count"] > 0 for record in records
        ),
        "samples_with_significant_score_change": sum(
            any(
                transition["evaluation_change_signal"] >= config.min_gain_for_continue
                for transition in record["teacher"]["depth_transitions"]
            )
            for record in records
        ),
        "samples_with_stable_search": sum(
            bool(record["teacher"]["depth_transitions"])
            and record["teacher"]["search_stability"] >= 1.0 - config.min_gain_for_continue
            for record in records
        ),
        "samples_with_insufficient_depth": sum(
            record["teacher"]["label_reason"] == "fewer_than_two_completed_depths"
            for record in records
        ),
        "insufficient_reasons": {
            reason: sum(record["teacher"]["label_reason"] == reason for record in records)
            for reason in sorted(
                {
                    record["teacher"]["label_reason"]
                    for record in records
                    if record["teacher"]["label_reason"] is not None
                }
            )
        },
        "teacher_config": asdict(config),
        "teacher_note": "Heuristic offline target; not a mathematically optimal time or a statistical probability.",
    }


def _distribution(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "median": None, "mean": None, "max": None}
    return {
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "max": max(values),
    }


def _safe_ratio(current: float, previous: float) -> float | None:
    if previous == 0:
        return None
    ratio = current / previous
    return ratio if math.isfinite(ratio) else None


def _required_number(record: dict[str, Any], field: str) -> float:
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Expected finite numeric field {field!r}")
    return float(value)


def _required_int(record: dict[str, Any], field: str) -> int:
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Expected integer field {field!r}")
    return value


def _required_str(record: dict[str, Any], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str):
        raise ValueError(f"Expected string field {field!r}")
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Label Phase 2A search telemetry with heuristic teacher targets.")
    parser.add_argument("--input", default="data/time_management/dataset.jsonl")
    parser.add_argument("--output", default="data/time_management/training_dataset.jsonl")
    parser.add_argument("--summary", default="data/time_management/teacher_summary.json")
    parser.add_argument("--move-change-weight", type=float, default=0.65)
    parser.add_argument("--score-change-weight", type=float, default=0.35)
    parser.add_argument("--score-scale", type=float, default=100.0)
    parser.add_argument("--cost-weight", type=float, default=0.25)
    parser.add_argument("--min-gain", type=float, default=0.12)
    parser.add_argument("--clock-safety-margin-ms", type=float, default=300.0)
    parser.add_argument("--confidence-transition-scale", type=int, default=3)
    parser.add_argument("--timeout-confidence-multiplier", type=float, default=0.8)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = TeacherConfig(
        score_scale=args.score_scale,
        move_change_weight=args.move_change_weight,
        score_change_weight=args.score_change_weight,
        cost_weight=args.cost_weight,
        min_gain_for_continue=args.min_gain,
        clock_safety_margin_ms=args.clock_safety_margin_ms,
        confidence_transition_scale=args.confidence_transition_scale,
        timeout_confidence_multiplier=args.timeout_confidence_multiplier,
    )
    config.validate()
    summary, output_path = label_jsonl(args.input, args.output, config)
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    print("Teacher labeling complete")
    print(f"Raw records: {summary['raw_records']}")
    print(f"Labeled: {summary['labeled_samples']}")
    print(f"Insufficient data: {summary['insufficient_samples']}")
    print(f"Label rate: {summary['label_rate']:.1%}")
    _print_distribution("Teacher time (ms)", summary["teacher_time_ms"])
    _print_distribution("Teacher depth", summary["teacher_depth"])
    _print_distribution("Teacher confidence", summary["teacher_confidence"])
    print(f"Clock distribution: {summary['clock_distribution']}")
    print(f"Samples with move changes: {summary['samples_where_best_move_changed']}")
    print(f"Samples with significant score changes: {summary['samples_with_significant_score_change']}")
    print(f"Samples with stable search: {summary['samples_with_stable_search']}")
    print(f"Samples with insufficient depth: {summary['samples_with_insufficient_depth']}")
    print(f"Training dataset: {output_path}")
    print(f"Summary: {summary_path}")


def _print_distribution(label: str, values: dict[str, float | None]) -> None:
    print(
        f"{label}: min={values['min']}, median={values['median']}, "
        f"mean={values['mean']}, max={values['max']}"
    )


if __name__ == "__main__":
    main()
