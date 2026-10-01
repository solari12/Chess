"""Command-line entry point for offline Phase 2A dataset generation."""

import argparse

from app.learning.time_management.dataset import build_dataset, write_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an offline Alpha-Beta search telemetry dataset.")
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--sampling-interval", type=int, default=4)
    parser.add_argument("--game-depth", type=int, default=2)
    parser.add_argument("--game-candidate-count", type=int, default=3)
    parser.add_argument("--max-plies", type=int, default=80)
    parser.add_argument("--analysis-budget-ms", type=int, default=1_000)
    parser.add_argument("--max-analysis-depth", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--remaining-times-ms", default="300000,120000,30000,10000,3000")
    parser.add_argument("--pgn", help="Optional local PGN file to sample (never downloaded).")
    parser.add_argument("--evaluation-profile", default="default")
    parser.add_argument("--output", default="data/time_management/dataset.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    remaining_times = [int(value.strip()) for value in args.remaining_times_ms.split(",") if value.strip()]
    records, metadata, summary = build_dataset(
        games=args.games,
        sampling_interval=args.sampling_interval,
        game_generation_depth=args.game_depth,
        candidate_count=args.game_candidate_count,
        max_plies=args.max_plies,
        analysis_budget_ms=args.analysis_budget_ms,
        max_analysis_depth=args.max_analysis_depth,
        remaining_times_ms=remaining_times,
        seed=args.seed,
        pgn_path=args.pgn,
        evaluation_profile=args.evaluation_profile,
    )
    dataset_path, metadata_path = write_dataset(records, metadata, args.output)
    print("Dataset generation complete.")
    print(f"Games: {summary.games}")
    print(f"Sampled positions: {summary.sampled_positions}")
    print(f"Clock variants: {summary.clock_variants}")
    print(f"Dataset records: {summary.dataset_records}")
    print(f"Average legal moves: {summary.average_legal_moves:.2f}")
    print(f"Average completed depth: {summary.average_completed_depth:.2f}")
    print(f"Average search time: {summary.average_search_time_ms:.2f} ms")
    print(f"Timeout rate: {summary.timeout_rate:.1%}")
    print(f"JSONL: {dataset_path}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
