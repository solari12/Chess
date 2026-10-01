"""Initial population generation and valid chromosome bounds."""

import random

from app.learning.genetic.models import BASELINE_WEIGHTS, Individual, PieceWeights

WEIGHT_BOUNDS: dict[str, tuple[int, int]] = {
    "pawn": (50, 200),
    "knight": (150, 500),
    "bishop": (150, 500),
    "rook": (300, 800),
    "queen": (700, 1200),
}


def random_weights(rng: random.Random) -> PieceWeights:
    """Draw integer genes inside the experiment's configured bounds."""
    baseline = BASELINE_WEIGHTS.as_dict()
    return PieceWeights(**{
        gene: min(high, max(low, round(rng.gauss(baseline[gene], (high - low) / 6))))
        for gene, (low, high) in WEIGHT_BOUNDS.items()
    })


def create_initial_population(
    population_size: int,
    rng: random.Random,
    generation: int = 1,
    first_id: int = 1,
) -> list[Individual]:
    if population_size < 2:
        raise ValueError("population_size must be at least 2")
    return [
        Individual(
            id=f"I{index:04d}",
            weights=random_weights(rng),
            fitness=None,
            generation=generation,
        )
        for index in range(first_id, first_id + population_size)
    ]
