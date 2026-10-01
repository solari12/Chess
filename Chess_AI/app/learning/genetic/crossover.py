"""Uniform gene-by-gene crossover with a complete parent-source record."""

import random
from dataclasses import dataclass

from app.learning.genetic.models import GENES, GeneOrigin, Individual, PieceWeights


@dataclass(frozen=True)
class CrossoverResult:
    weights: PieceWeights
    gene_origins: list[GeneOrigin]


def uniform_crossover(
    parent_a: Individual,
    parent_b: Individual,
    rng: random.Random,
) -> CrossoverResult:
    values: dict[str, int] = {}
    origins: list[GeneOrigin] = []
    weights_a = parent_a.weights.as_dict()
    weights_b = parent_b.weights.as_dict()

    for gene in GENES:
        source_parent = "parent_a" if rng.random() < 0.5 else "parent_b"
        source = parent_a if source_parent == "parent_a" else parent_b
        value = weights_a[gene] if source_parent == "parent_a" else weights_b[gene]
        values[gene] = value
        origins.append(GeneOrigin(gene, value, source_parent, source.id))

    return CrossoverResult(PieceWeights(**values), origins)
