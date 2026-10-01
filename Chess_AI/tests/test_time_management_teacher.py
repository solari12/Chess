"""Tests for the deterministic Phase 2B heuristic teacher."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from app.learning.time_management.teacher import (
    TeacherConfig,
    calculate_depth_transition,
    calculate_information_gain,
    generate_target,
    label_dataset,
    label_jsonl,
)


def profile(depth, move, score, time_ms, nodes):
    return {"depth": depth, "move": move, "score": score, "time_ms": time_ms, "nodes": nodes}


def raw_record(profiles, remaining_ms=30_000, sample_id="P000001_T001"):
    return {
        "sample_id": sample_id,
        "fen": "test-fen",
        "clock": {"remaining_time_ms": remaining_ms, "clock_source": "synthetic"},
        "position_features": {"side_to_move": "white"},
        "search": {"timed_out": False},
        "depth_profiles": profiles,
    }


class TeacherSignalTests(unittest.TestCase):
    def test_stable_search_has_low_gain_and_selects_a_later_confirmation_depth(self):
        record = raw_record([
            profile(1, "e2e4", 20, 3, 10),
            profile(2, "e2e4", 21, 12, 30),
            profile(3, "e2e4", 22, 60, 100),
            profile(4, "e2e4", 21, 310, 400),
        ])

        teacher = generate_target(record)

        self.assertLess(teacher["information_gain_score"], 0.12)
        self.assertEqual(teacher["teacher_depth"], 2)
        self.assertGreater(teacher["search_stability"], 0.9)

    def test_move_change_increases_decision_signal_and_information_gain(self):
        previous = profile(3, "e2e4", 15, 50, 100)
        stable = profile(4, "e2e4", 15, 220, 300)
        changed = profile(4, "d5d4", 15, 220, 300)
        config = TeacherConfig(cost_weight=0)

        stable_transition = calculate_depth_transition(previous, stable, config)
        changed_transition = calculate_depth_transition(previous, changed, config)

        self.assertFalse(stable_transition["best_move_changed"])
        self.assertTrue(changed_transition["best_move_changed"])
        self.assertEqual(changed_transition["decision_change_signal"], 1)
        self.assertGreater(
            calculate_information_gain(changed_transition, config),
            calculate_information_gain(stable_transition, config),
        )

    def test_large_score_change_contributes_more_than_small_change(self):
        previous = profile(1, "e2e4", 0, 3, 10)
        small = calculate_depth_transition(previous, profile(2, "e2e4", 2, 12, 20), TeacherConfig(cost_weight=0))
        large = calculate_depth_transition(previous, profile(2, "e2e4", 100, 12, 20), TeacherConfig(cost_weight=0))

        self.assertGreater(large["evaluation_change_signal"], small["evaluation_change_signal"])
        self.assertGreater(large["information_gain_score"], small["information_gain_score"])
        self.assertLess(small["information_gain_score"], 0.02)

    def test_target_uses_cumulative_time_at_first_low_gain_depth(self):
        record = raw_record([
            profile(1, "e2e4", 0, 3, 10),
            profile(2, "e2e4", 100, 12, 30),
            profile(3, "e2e4", 200, 60, 100),
            profile(4, "e2e4", 300, 310, 400),
            profile(5, "e2e4", 301, 1_900, 2_000),
        ])
        config = TeacherConfig(cost_weight=0, min_gain_for_continue=0.15)

        teacher = generate_target(record, config)

        self.assertEqual(teacher["teacher_depth"], 4)
        self.assertEqual(teacher["teacher_time_ms"], 310)

    def test_remaining_clock_caps_target_after_safety_reserve(self):
        record = raw_record([
            profile(1, "e2e4", 0, 3, 10),
            profile(2, "d2d4", 100, 5_000, 30),
        ], remaining_ms=3_000)

        teacher = generate_target(record, TeacherConfig(cost_weight=0))

        self.assertEqual(teacher["teacher_time_ms"], 2_700)
        self.assertLessEqual(teacher["teacher_time_ms"], teacher["usable_clock_ms"])

    def test_one_completed_depth_is_insufficient_without_fabricated_target(self):
        teacher = generate_target(raw_record([profile(1, "e2e4", 0, 3, 10)]))

        self.assertEqual(teacher["label_status"], "insufficient_data")
        self.assertIsNone(teacher["teacher_time_ms"])
        self.assertIsNone(teacher["teacher_depth"])
        self.assertEqual(teacher["teacher_confidence"], 0)

    def test_unmeasurable_zero_search_time_is_not_emitted_as_a_target(self):
        record = raw_record([
            profile(1, "e2e4", 0, 0, 10),
            profile(2, "d2d4", 100, 0, 30),
        ])

        teacher = generate_target(record, TeacherConfig(cost_weight=0))

        self.assertEqual(teacher["label_status"], "insufficient_data")
        self.assertEqual(teacher["label_reason"], "selected_depth_time_unmeasurable")
        self.assertIsNone(teacher["teacher_time_ms"])

    def test_clock_variants_of_same_fen_receive_separate_capped_labels(self):
        profiles = [profile(1, "e2e4", 0, 3, 10), profile(2, "d2d4", 100, 5_000, 30)]
        records = [
            raw_record(profiles, remaining_ms=30_000, sample_id="P1_T1"),
            raw_record(profiles, remaining_ms=3_000, sample_id="P1_T2"),
        ]

        labeled = label_dataset(records, TeacherConfig(cost_weight=0))

        self.assertEqual(labeled[0]["fen"], labeled[1]["fen"])
        self.assertEqual(labeled[0]["teacher"]["teacher_time_ms"], 5_000)
        self.assertEqual(labeled[1]["teacher"]["teacher_time_ms"], 2_700)

    def test_same_input_and_config_produce_identical_labels(self):
        records = [raw_record([
            profile(1, "e2e4", 0, 3, 10),
            profile(2, "e2e4", 1, 12, 30),
            profile(3, "e2e4", 2, 60, 100),
        ])]
        config = TeacherConfig()

        first = label_dataset(records, config)
        second = label_dataset(records, config)

        self.assertEqual(first, second)

    def test_jsonl_labeling_keeps_raw_file_byte_for_byte_unchanged(self):
        record = raw_record([
            profile(1, "e2e4", 0, 3, 10),
            profile(2, "e2e4", 1, 12, 30),
        ])
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "raw.jsonl"
            output_path = Path(directory) / "training.jsonl"
            original = json.dumps(record, separators=(",", ":")) + "\n"
            input_path.write_text(original, encoding="utf-8")
            before = hashlib.sha256(input_path.read_bytes()).hexdigest()

            summary, output = label_jsonl(input_path, output_path)

            after = hashlib.sha256(input_path.read_bytes()).hexdigest()
            labeled_record = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(before, after)
        self.assertEqual(summary["raw_records"], 1)
        self.assertEqual(summary["labeled_samples"], 1)
        self.assertIn("teacher", labeled_record)
        self.assertNotIn("teacher", record)


if __name__ == "__main__":
    unittest.main()
