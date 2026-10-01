"""Run a small set of real timed engine searches with shadow logging enabled."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Sequence

import chess

from app.engine.search import search_with_time_budget
from app.learning.time_management.runtime_shadow import DEFAULT_LOG_PATH
from app.schemas.chess import TimedAIRequest


def run_sample(
    *,
    searches: int,
    time_budget_ms: int,
    remaining_time_ms: int,
    max_depth: int,
    log_path: str | Path = DEFAULT_LOG_PATH,
) -> int:
    """Execute real searches and let the timed endpoint append each shadow row."""
    if searches < 1 or time_budget_ms < 1 or remaining_time_ms < 1:
        raise ValueError("searches, time_budget_ms, and remaining_time_ms must be positive")
    board = chess.Board()
    clock_left = {chess.WHITE: remaining_time_ms, chess.BLACK: remaining_time_ms}
    completed = 0
    for _ in range(searches):
        side_to_move = board.turn
        remaining = clock_left[side_to_move]
        if board.is_game_over(claim_draw=True) or remaining <= 0:
            break
        actual_budget = max(1, min(time_budget_ms, remaining))
        request = TimedAIRequest(
            fen=board.fen(),
            algorithm="alpha-beta",
            time_budget_ms=actual_budget,
            remaining_time_ms=remaining,
            max_depth=max_depth,
        )
        # The regular search orchestration is used; only the shadow log path is
        # redirected for this sample command.
        response = search_with_time_budget(request, shadow_log_path=str(log_path))
        completed += 1
        clock_left[side_to_move] = max(0, remaining - math.ceil(response.time_ms))
        if not response.move:
            break
        board.push_uci(response.move)
    return completed


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run real searches and log Phase 2F shadow predictions.")
    parser.add_argument("--searches", type=int, default=5)
    parser.add_argument("--time-budget-ms", type=int, default=1_000)
    parser.add_argument("--remaining-time-ms", type=int, default=30_000)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--log", default=str(DEFAULT_LOG_PATH))
    args = parser.parse_args(argv)
    completed = run_sample(
        searches=args.searches,
        time_budget_ms=args.time_budget_ms,
        remaining_time_ms=args.remaining_time_ms,
        max_depth=args.max_depth,
        log_path=args.log,
    )
    print(f"Completed real timed searches with shadow telemetry: {completed}")
    print(f"JSONL: {args.log}")


if __name__ == "__main__":
    main()
