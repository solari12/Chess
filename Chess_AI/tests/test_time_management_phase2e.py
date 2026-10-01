"""Focused Phase 2E low-clock and safety-cap validation tests."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from app.learning.time_management.phase2e import (
    TIME_PRESSURE_CLOCKS_MS,
    build_clock_report,
    enrich_with_clock_diagnostics,
    generate_phase2e_artifacts,
)
from app.learning.time_management.position_generator import generate_positions
from app.learning.time_management.teacher import TeacherConfig, generate_target


def record_for_clock(remaining_ms: int, natural_ms: float = 1_200.0) -> dict:
    return {
        "sample_id": f"clock-{remaining_ms}",
        "fen": "same-fen-for-focused-test",
        "clock": {"remaining_time_ms": remaining_ms, "clock_source": "synthetic"},
        "position_features": {},
        "search": {"timed_out": False, "completed_depth": 2, "total_nodes": 20},
        "depth_profiles": [
            {"depth": 1, "move": "e2e4", "score": 0, "time_ms": 10, "nodes": 10},
            {"depth": 2, "move": "e2e4", "score": 0, "time_ms": natural_ms, "nodes": 20},
        ],
    }


class Phase2EClockCapTests(unittest.TestCase):
    def test_cap_activates_when_natural_target_exceeds_usable_clock(self):
        labeled = enrich_with_clock_diagnostics([record_for_clock(500)])[0]
        diagnostic = labeled["teacher_diagnostics"]

        self.assertEqual(diagnostic["natural_teacher_time_ms"], 1_200)
        self.assertEqual(diagnostic["usable_time_ms"], 200)
        self.assertEqual(diagnostic["teacher_time_ms"], 200)
        self.assertTrue(diagnostic["clock_cap_activated"])
        self.assertEqual(diagnostic["cap_amount_ms"], 1_000)

    def test_cap_does_not_activate_when_natural_target_fits(self):
        labeled = enrich_with_clock_diagnostics([record_for_clock(2_000)])[0]
        diagnostic = labeled["teacher_diagnostics"]

        self.assertEqual(diagnostic["usable_time_ms"], 1_700)
        self.assertEqual(diagnostic["teacher_time_ms"], 1_200)
        self.assertFalse(diagnostic["clock_cap_activated"])
        self.assertEqual(diagnostic["cap_amount_ms"], 0)

    def test_exact_safety_margin_produces_no_negative_target(self):
        labeled = enrich_with_clock_diagnostics([record_for_clock(300)])[0]
        diagnostic = labeled["teacher_diagnostics"]

        self.assertEqual(diagnostic["usable_time_ms"], 0)
        self.assertIsNone(diagnostic["teacher_time_ms"])
        self.assertEqual(diagnostic["label_status"], "insufficient_data")
        self.assertTrue(diagnostic["clock_cap_activated"])

    def test_below_safety_margin_produces_no_negative_target(self):
        labeled = enrich_with_clock_diagnostics([record_for_clock(299)])[0]
        diagnostic = labeled["teacher_diagnostics"]

        self.assertEqual(diagnostic["usable_time_ms"], 0)
        self.assertIsNone(diagnostic["teacher_time_ms"])
        self.assertEqual(diagnostic["label_status"], "insufficient_data")

    def test_zero_remaining_time_produces_no_negative_target(self):
        labeled = enrich_with_clock_diagnostics([record_for_clock(0)])[0]
        diagnostic = labeled["teacher_diagnostics"]

        self.assertEqual(diagnostic["usable_time_ms"], 0)
        self.assertIsNone(diagnostic["teacher_time_ms"])
        self.assertGreaterEqual(diagnostic["cap_amount_ms"], 0)

    def test_teacher_never_emits_negative_final_target_across_edge_clocks(self):
        for remaining_ms in (30_000, 500, 300, 299, 0):
            with self.subTest(remaining_ms=remaining_ms):
                target = generate_target(record_for_clock(remaining_ms))
                final = target["teacher_time_ms"]
                self.assertTrue(final is None or final >= 0)

    def test_fixed_seed_produces_same_position_sequence(self):
        kwargs = {
            "games": 1,
            "sampling_interval": 2,
            "game_generation_depth": 1,
            "max_plies": 4,
            "seed": 42,
            "candidate_count": 1,
        }
        first = generate_positions(**kwargs)
        second = generate_positions(**kwargs)

        self.assertEqual([sample.fen for sample in first], [sample.fen for sample in second])

    def test_phase2a_raw_dataset_is_not_modified_by_phase2e_generation(self):
        raw_phase2a = Path("data/time_management/dataset.jsonl")
        before = hashlib.sha256(raw_phase2a.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            paths = generate_phase2e_artifacts(
                output_dir=directory,
                games=0,
                max_plies=0,
                analysis_budget_ms=25,
                max_analysis_depth=2,
                seed=42,
            )
            raw = [json.loads(line) for line in Path(paths["raw_dataset"]).read_text(encoding="utf-8").splitlines()]
            after = hashlib.sha256(raw_phase2a.read_bytes()).hexdigest()

        self.assertEqual(before, after)
        self.assertEqual({item["clock"]["remaining_time_ms"] for item in raw}, set(TIME_PRESSURE_CLOCKS_MS))

    def test_same_fen_and_clock_gives_deterministic_teacher_output(self):
        raw = record_for_clock(500)
        first = enrich_with_clock_diagnostics([raw], TeacherConfig())
        second = enrich_with_clock_diagnostics([raw], TeacherConfig())

        self.assertEqual(first, second)

    def test_report_includes_cap_groups_monotonicity_and_edge_cases(self):
        raw = [record_for_clock(clock) for clock in (2_000, 500, 300)]
        labeled = enrich_with_clock_diagnostics(raw)
        report, examples = build_clock_report(raw, labeled)

        self.assertEqual(set(report["clock_cap_by_group"]), {str(clock) for clock in TIME_PRESSURE_CLOCKS_MS})
        self.assertEqual(report["clock_monotonicity"]["comparable_fen_groups"], 1)
        self.assertEqual(report["clock_monotonicity"]["violations"], 0)
        self.assertIn("exact_safety_margin", report["safety_margin_edge_cases"])
        self.assertIn("monotonic_violations", examples)


if __name__ == "__main__":
    unittest.main()
