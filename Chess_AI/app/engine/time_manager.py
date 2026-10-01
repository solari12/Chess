"""Simple deadline construction for a single engine move search."""

TIME_SAFETY_MARGIN_MS = 50.0


def deadline_after(
    started_at: float,
    time_budget_ms: float,
    safety_margin_ms: float = TIME_SAFETY_MARGIN_MS,
) -> float:
    """Return a monotonic deadline with a small fixed/proportional reserve.

    The proportional cap keeps tiny test budgets positive; normal budgets use
    the explicit 50 ms reserve to leave room for Python/API overhead.
    """
    margin = min(safety_margin_ms, time_budget_ms * 0.1)
    return started_at + max(0.0, time_budget_ms - margin) / 1_000
