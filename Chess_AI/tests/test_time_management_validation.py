"""Tests for Phase 2B.1 target distribution reporting."""

import unittest

from app.learning.time_management.validation import build_validation_report


def paired_records(sample_id, clock_ms, teacher_time, depth, move_changes, info_gain, stability):
    raw = {
        "sample_id": sample_id,
        "fen": f"fen-{sample_id}",
        "clock": {"remaining_time_ms": clock_ms, "clock_source": "synthetic"},
        "position_features": {"legal_move_count": 20},
        "search": {"timed_out": False},
        "depth_profiles": [],
    }
    labeled = {
        **raw,
        "teacher": {
            "label_status": "labeled",
            "teacher_time_ms": teacher_time,
            "teacher_depth": depth,
            "teacher_confidence": 0.8,
            "information_gain_score": info_gain,
            "search_stability": stability,
            "best_move_change_count": move_changes,
            "depth_transitions": [
                {"evaluation_change_signal": info_gain}
            ] if info_gain > 0 else [],
        },
    }
    return raw, labeled


class TimeManagementValidationTests(unittest.TestCase):
    def test_summary_percentiles_depth_counts_and_clock_groups(self):
        pairs = [
            paired_records("A", 300_000, 10, 2, 1, 0.2, 0.8),
            paired_records("B", 3_000, 30, 3, 2, 0.4, 0.6),
            paired_records("C", 300_000, 20, 2, 0, 0.0, 1.0),
        ]
        raw = [pair[0] for pair in pairs]
        labeled = [pair[1] for pair in pairs]

        report = build_validation_report(raw, labeled)

        self.assertEqual(report["raw_records"], 3)
        self.assertEqual(report["labeled_records"], 3)
        self.assertEqual(report["insufficient_records"], 0)
        self.assertEqual(report["teacher_depth_distribution"]["counts"], {"2": 2, "3": 1})
        self.assertEqual(report["teacher_time_ms_distribution"]["p25"], 15)
        self.assertEqual(report["teacher_time_ms_distribution"]["p90"], 28)
        self.assertAlmostEqual(report["teacher_confidence_distribution"]["mean"], 0.8)
        self.assertAlmostEqual(report["information_gain_distribution"]["p75"], 0.3)
        self.assertEqual(report["clock_distribution"]["300000"]["samples"], 2)
        self.assertEqual(report["clock_distribution"]["300000"]["labeled_count"], 2)
        self.assertEqual(report["clock_distribution"]["300000"]["median_teacher_time_ms"], 15)
        self.assertEqual(report["clock_distribution"]["3000"]["median_teacher_depth"], 3)
        self.assertEqual(report["search_stability"]["positions_with_repeated_best_move_changes"], 1)

    def test_raw_and_labeled_records_must_match_without_mutating_raw(self):
        raw, labeled = paired_records("same", 30_000, 12, 2, 0, 0.0, 1.0)
        original = dict(raw)
        report = build_validation_report([raw], [labeled])
        self.assertEqual(raw, original)
        self.assertEqual(report["raw_records"], 1)

    def test_mismatched_raw_and_labeled_samples_are_rejected(self):
        raw, labeled = paired_records("raw", 30_000, 12, 2, 0, 0.0, 1.0)
        labeled["sample_id"] = "different"
        with self.assertRaisesRegex(ValueError, "matching sample_id"):
            build_validation_report([raw], [labeled])


if __name__ == "__main__":
    unittest.main()
