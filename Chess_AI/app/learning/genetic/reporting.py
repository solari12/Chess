"""Summaries for completed Genetic Laboratory self-play games."""

from collections import Counter
from collections.abc import Iterable

from app.learning.genetic.models import CandidateGameTrace


def summarize_game_traces(traces: Iterable[CandidateGameTrace]) -> dict[str, object]:
    completed = [trace for trace in traces if trace.result is not None]
    outcomes = Counter(trace.result for trace in completed)
    draws_by_reason = Counter(
        trace.termination or "other" for trace in completed if trace.result == "draw"
    )
    terminations = Counter(trace.termination or "other" for trace in completed)
    plies = [trace.plies for trace in completed]
    return {
        "total_games": len(completed),
        "wins": outcomes["win"],
        "draws": outcomes["draw"],
        "losses": outcomes["loss"],
        "draws_by_termination_reason": dict(sorted(draws_by_reason.items())),
        "termination_counts": dict(sorted(terminations.items())),
        "average_plies": (sum(plies) / len(plies)) if plies else 0.0,
        "max_plies": max(plies, default=0),
    }
