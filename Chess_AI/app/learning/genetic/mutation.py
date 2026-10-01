"""Bounded per-gene mutation that records every effective gene change."""

import random

from app.learning.genetic.models import GENES, MutationEvent, PieceWeights
from app.learning.genetic.population import WEIGHT_BOUNDS


def mutate(
    weights: PieceWeights,
    mutation_rate: float,
    mutation_strength: int,
    rng: random.Random,
) -> tuple[PieceWeights, list[MutationEvent]]:
    if not 0 <= mutation_rate <= 1:
        raise ValueError("mutation_rate must be between 0 and 1")
    if mutation_strength < 1:
        raise ValueError("mutation_strength must be positive")

    values = weights.as_dict()
    events: list[MutationEvent] = []
    for gene in GENES:
        if rng.random() >= mutation_rate:
            continue

        old_value = values[gene]
        low, high = WEIGHT_BOUNDS[gene]
        delta = rng.randint(-mutation_strength, mutation_strength)
        if delta == 0:
            delta = rng.choice((-1, 1))
        new_value = min(high, max(low, old_value + delta))

        if new_value == old_value:
            if old_value == low:
                new_value = min(high, old_value + rng.randint(1, mutation_strength))
            else:
                new_value = max(low, old_value - rng.randint(1, mutation_strength))

        values[gene] = new_value
        events.append(MutationEvent(gene, old_value, new_value, new_value - old_value))

    return PieceWeights(**values), events
