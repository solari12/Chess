"""Build, validate, and write Phase 2A JSONL datasets."""

from collections.abc import Callable, Sequence
import json
import math
from pathlib import Path

import chess

from app.engine.evaluation import evaluate
from app.learning.time_management import DATASET_VERSION, GENERATOR_VERSION
from app.learning.time_management.features import extract_position_features
from app.learning.time_management.models import DatasetSummary, PositionSample
from app.learning.time_management.position_generator import generate_positions
from app.learning.time_management.telemetry import collect_search_telemetry

Evaluator = Callable[[chess.Board], int]
FORBIDDEN_TARGET_FIELDS = {
    "recommended_time_ms",
    "target_time_ms",
    "optimal_time_ms",
    "target_depth",
    "good_time",
    "bad_time",
}


def build_dataset(
    *,
    games: int = 10,
    sampling_interval: int = 4,
    game_generation_depth: int = 2,
    max_plies: int = 80,
    analysis_budget_ms: int = 1_000,
    max_analysis_depth: int = 6,
    remaining_times_ms: Sequence[int] = (300_000, 120_000, 30_000, 10_000, 3_000),
    seed: int = 42,
    pgn_path: str | None = None,
    candidate_count: int = 3,
    evaluation_profile: str = "default",
    evaluator: Evaluator = evaluate,
) -> tuple[list[dict], dict, DatasetSummary]:
    """Generate sampled positions and analyze every configured synthetic clock.

    Search telemetry is repeated for each clock variant as requested; clock
    variants remain explicit observations and never become time targets.
    """
    _validate_config(analysis_budget_ms, max_analysis_depth, remaining_times_ms, evaluation_profile)
    clock_values = list(dict.fromkeys(remaining_times_ms))
    samples = generate_positions(
        games=games,
        sampling_interval=sampling_interval,
        game_generation_depth=game_generation_depth,
        max_plies=max_plies,
        seed=seed,
        pgn_path=pgn_path,
        candidate_count=candidate_count,
        evaluator=evaluator,
    )

    records: list[dict] = []
    seen: set[tuple[str, int]] = set()
    for position_index, sample in enumerate(samples, start=1):
        board = chess.Board(sample.fen)
        if not board.is_valid():
            raise ValueError(f"Invalid sampled FEN: {sample.fen}")
        for clock_index, remaining_ms in enumerate(clock_values, start=1):
            identity = (sample.fen, remaining_ms)
            if identity in seen:
                continue
            seen.add(identity)
            features = extract_position_features(board, remaining_ms)
            telemetry = collect_search_telemetry(
                board,
                analysis_budget_ms=analysis_budget_ms,
                max_analysis_depth=max_analysis_depth,
                evaluator=evaluator,
            )
            record = {
                "sample_id": f"P{position_index:06d}_T{clock_index:03d}",
                "fen": sample.fen,
                "source": {
                    "type": sample.source,
                    "game_index": sample.game_index,
                    "ply": sample.ply,
                },
                "clock": {"remaining_time_ms": remaining_ms, "clock_source": "synthetic"},
                "position_features": features,
                "search": {key: value for key, value in telemetry.items() if key != "depth_profiles"},
                "depth_profiles": telemetry["depth_profiles"],
            }
            validate_record(record)
            records.append(record)

    metadata = {
        "dataset_version": DATASET_VERSION,
        "generator_version": GENERATOR_VERSION,
        "seed": seed,
        "games": games,
        "sampling_interval": sampling_interval,
        "game_generation_depth": game_generation_depth,
        "game_candidate_count": candidate_count,
        "max_plies": max_plies,
        "analysis_budget_ms": analysis_budget_ms,
        "max_analysis_depth": max_analysis_depth,
        "remaining_times_ms": clock_values,
        "evaluation_profile": evaluation_profile,
        "pgn_source": pgn_path,
        "position_source": "engine_games+starting_position" + ("+pgn" if pgn_path else ""),
    }
    summary = summarize_dataset(records, games, len(samples), len(clock_values))
    return records, metadata, summary


def validate_record(record: dict) -> None:
    """Reject malformed records before they can be serialized."""
    if not all(key in record for key in ("sample_id", "fen", "clock", "position_features", "search", "depth_profiles")):
        raise ValueError("Dataset record is missing a required top-level field")
    if FORBIDDEN_TARGET_FIELDS.intersection(_walk_keys(record)):
        raise ValueError("Phase 2A records must not contain target or recommended-time fields")
    board = chess.Board(record["fen"])
    if not board.is_valid():
        raise ValueError(f"Dataset record has an invalid FEN: {record['fen']}")
    legal_moves = list(board.legal_moves)
    features = record["position_features"]
    if features.get("legal_move_count") != len(legal_moves):
        raise ValueError("legal_move_count does not match the stored FEN")
    if not legal_moves and not board.is_game_over(claim_draw=True):
        raise ValueError("A non-terminal position must have at least one legal move")
    if record["clock"].get("clock_source") != "synthetic":
        raise ValueError("Offline clock records must be marked synthetic")
    remaining_ms = record["clock"].get("remaining_time_ms")
    if not isinstance(remaining_ms, int) or remaining_ms < 0:
        raise ValueError("remaining_time_ms must be a non-negative integer")
    profiles = record["depth_profiles"]
    completed_depth = record["search"].get("completed_depth")
    if completed_depth != len(profiles):
        raise ValueError("completed_depth must match the number of completed depth profiles")
    for expected_depth, profile in enumerate(profiles, start=1):
        if profile.get("depth") != expected_depth:
            raise ValueError("Depth profiles must be ordered and contiguous")
        try:
            move = chess.Move.from_uci(profile["move"])
        except (KeyError, ValueError) as error:
            raise ValueError("Depth profile contains an invalid move") from error
        if move not in legal_moves:
            raise ValueError(f"Recorded search move {move.uci()} is illegal in {record['fen']}")
        if profile.get("nodes", -1) < 0 or profile.get("time_ms", -1) < 0:
            raise ValueError("Search nodes and times must be non-negative")
    if record["search"].get("total_nodes", -1) < 0 or record["search"].get("total_time_ms", -1) < 0:
        raise ValueError("Total search nodes and time must be non-negative")
    if not _all_finite(record):
        raise ValueError("Dataset record contains a non-finite feature or telemetry value")


def write_dataset(records: Sequence[dict], metadata: dict, output_path: str | Path) -> tuple[Path, Path]:
    """Write JSONL and a sibling metadata JSON file, creating parent folders."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = path.with_name(f"{path.stem}.metadata.json")
    with path.open("w", encoding="utf-8", newline="\n") as dataset_file:
        for record in records:
            validate_record(record)
            dataset_file.write(json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":")))
            dataset_file.write("\n")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    return path, metadata_path


def summarize_dataset(records: Sequence[dict], games: int, sampled_positions: int, clock_variants: int) -> DatasetSummary:
    count = len(records)
    divisor = count or 1
    return DatasetSummary(
        games=games,
        sampled_positions=sampled_positions,
        clock_variants=clock_variants,
        dataset_records=count,
        average_legal_moves=sum(item["position_features"]["legal_move_count"] for item in records) / divisor,
        average_completed_depth=sum(item["search"]["completed_depth"] for item in records) / divisor,
        average_search_time_ms=sum(item["search"]["total_time_ms"] for item in records) / divisor,
        timeout_rate=sum(bool(item["search"]["timed_out"]) for item in records) / divisor,
    )


def _validate_config(
    analysis_budget_ms: int,
    max_analysis_depth: int,
    remaining_times_ms: Sequence[int],
    evaluation_profile: str,
) -> None:
    if analysis_budget_ms < 1:
        raise ValueError("analysis_budget_ms must be at least 1")
    if not 1 <= max_analysis_depth <= 64:
        raise ValueError("max_analysis_depth must be between 1 and 64")
    if not remaining_times_ms:
        raise ValueError("At least one remaining clock value is required")
    if any(not isinstance(value, int) or value < 0 for value in remaining_times_ms):
        raise ValueError("remaining clock values must be non-negative integers")
    if not evaluation_profile:
        raise ValueError("evaluation_profile cannot be empty")


def _walk_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from _walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_keys(child)


def _all_finite(value) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_all_finite(child) for child in value.values())
    if isinstance(value, list):
        return all(_all_finite(child) for child in value)
    return True
