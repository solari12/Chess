"""Tournament selection for choosing genetic parents."""

import random
from collections.abc import Sequence

from app.learning.genetic.models import Individual


def tournament_select(
    population: Sequence[Individual],
    tournament_size: int,
    rng: random.Random,
) -> Individual:
    """Select the fittest member of a randomly sampled mini-tournament.

    Tournament selection randomly chooses a small subset of the population,
    compares their fitness, and selects the best one as a parent.
    """
    if not population:
        raise ValueError("population cannot be empty")
    if not 1 <= tournament_size <= len(population):
        raise ValueError("tournament_size must be within the population size")

    contestants = rng.sample(list(population), tournament_size)
    return max(contestants, key=lambda individual: individual.fitness if individual.fitness is not None else float("-inf"))
