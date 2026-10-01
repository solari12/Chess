"""Tests for offline Phase 2A time-management dataset generation."""

import json
from pathlib import Path
import tempfile
import unittest

import chess

from app.learning.time_management.dataset import (
    FORBIDDEN_TARGET_FIELDS,
    build_dataset,
    validate_record,
    write_dataset,
)
from app.learning.time_management.features import extract_position_features
from app.learning.time_management.position_generator import generate_positions


class TimeManagementFeatureTests(unittest.TestCase):
    def test_start_position_features_are_interpretable_and_correct(self):
        features = extract_position_features(chess.Board(), remaining_time_ms=30_000)

        self.assertEqual(features["remaining_time_ms"], 30_000)
        self.assertEqual(features["side_to_move"], "white")
        self.assertEqual(features["legal_move_count"], 20)
        self.assertEqual(features["capture_count"], 0)
        self.assertFalse(features["side_in_check"])
        self.assertEqual(features["material_white"], 4_000)
        self.assertEqual(features["material_black"], 4_000)
        self.assertEqual(features["material_imbalance"], 0)
        self.assertEqual(features["pawn_count_white"], 8)
        self.assertEqual(features["knight_count_black"], 2)

    def test_check_position_reports_side_in_check(self):
        board = chess.Board("4k3/8/8/8/8/8/4r3/4K3 w - - 0 1")
        features = extract_position_features(board, remaining_time_ms=5_000)
        self.assertTrue(features["side_in_check"])


class TimeManagementDatasetTests(unittest.TestCase):
    def test_seed_reproduces_sample_positions(self):
        config = dict(
            games=2,
            sampling_interval=2,
            game_generation_depth=1,
            max_plies=6,
            seed=13,
        )
        first = generate_positions(**config)
        second = generate_positions(**config)
        self.assertEqual(
            [(item.fen, item.source, item.game_index, item.ply) for item in first],
            [(item.fen, item.source, item.game_index, item.ply) for item in second],
        )

    def test_one_position_produces_each_synthetic_clock_variant(self):
        records, _, _ = build_dataset(
            games=0,
            remaining_times_ms=[300_000, 120_000, 30_000, 3_000],
            analysis_budget_ms=25,
            max_analysis_depth=2,
        )
        self.assertEqual(len(records), 4)
        self.assertEqual(len({record["fen"] for record in records}), 1)
        self.assertEqual(
            [record["clock"]["remaining_time_ms"] for record in records],
            [300_000, 120_000, 30_000, 3_000],
        )
        self.assertTrue(all(record["clock"]["clock_source"] == "synthetic" for record in records))

    def test_dataset_schema_has_observations_but_no_target_fields(self):
        records, metadata, _ = build_dataset(
            games=0,
            remaining_times_ms=[10_000],
            analysis_budget_ms=50,
            max_analysis_depth=2,
        )
        record = records[0]
        for key in ("sample_id", "fen", "clock", "position_features", "search", "depth_profiles"):
            self.assertIn(key, record)
        self.assertEqual(metadata["evaluation_profile"], "default")
        self.assertFalse(FORBIDDEN_TARGET_FIELDS.intersection(_all_keys(record)))

    def test_completed_profile_moves_are_legal_and_ordered(self):
        records, _, _ = build_dataset(
            games=0,
            remaining_times_ms=[5_000],
            analysis_budget_ms=100,
            max_analysis_depth=3,
        )
        record = records[0]
        validate_record(record)
        board = chess.Board(record["fen"])
        depths = [item["depth"] for item in record["depth_profiles"]]
        self.assertEqual(depths, list(range(1, record["search"]["completed_depth"] + 1)))
        for profile in record["depth_profiles"]:
            self.assertIn(chess.Move.from_uci(profile["move"]), board.legal_moves)
            for field in ("nodes", "time_ms", "best_move_changed", "score_delta", "time_growth_ratio", "node_growth_ratio"):
                self.assertIn(field, profile)

    def test_duplicate_fen_clock_pairs_are_not_written_twice(self):
        records, _, _ = build_dataset(
            games=3,
            sampling_interval=1,
            game_generation_depth=1,
            max_plies=0,
            remaining_times_ms=[2_000, 2_000],
            analysis_budget_ms=10,
            max_analysis_depth=1,
        )
        identities = [(item["fen"], item["clock"]["remaining_time_ms"]) for item in records]
        self.assertEqual(len(identities), len(set(identities)))

    def test_pgn_terminal_position_is_supported(self):
        pgn_text = "1. f3 e5 2. g4 Qh4#\n"
        with tempfile.TemporaryDirectory() as temp_dir:
            pgn_path = Path(temp_dir) / "terminal.pgn"
            pgn_path.write_text(pgn_text, encoding="utf-8")
            records, _, _ = build_dataset(
                games=0,
                pgn_path=str(pgn_path),
                sampling_interval=4,
                remaining_times_ms=[1_000],
                analysis_budget_ms=20,
                max_analysis_depth=2,
            )
        terminal_records = [item for item in records if chess.Board(item["fen"]).is_game_over()]
        self.assertEqual(len(terminal_records), 1)
        self.assertEqual(terminal_records[0]["search"]["completed_depth"], 0)
        self.assertEqual(terminal_records[0]["depth_profiles"], [])

    def test_jsonl_and_metadata_are_written(self):
        records, metadata, _ = build_dataset(
            games=0,
            remaining_times_ms=[1_000],
            analysis_budget_ms=10,
            max_analysis_depth=1,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "nested" / "sample.jsonl"
            data_path, metadata_path = write_dataset(records, metadata, output)
            loaded = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines()]
            loaded_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(len(loaded), len(records))
        self.assertEqual(loaded_metadata["seed"], 42)


def _all_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from _all_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _all_keys(child)


if __name__ == "__main__":
    unittest.main()
