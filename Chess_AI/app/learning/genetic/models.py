"""Data models for genetic material-weight experiments."""

from dataclasses import dataclass, field
from typing import Any, Literal


GENES = ("pawn", "knight", "bishop", "rook", "queen")


@dataclass(frozen=True)
class PieceWeights:
    pawn: int
    knight: int
    bishop: int
    rook: int
    queen: int

    def __post_init__(self) -> None:
        if any(type(value) is not int for value in self.as_tuple()):
            raise TypeError("Piece weights must be integers")

    def as_tuple(self) -> tuple[int, int, int, int, int]:
        return (self.pawn, self.knight, self.bishop, self.rook, self.queen)

    def as_dict(self) -> dict[str, int]:
        return dict(zip(GENES, self.as_tuple(), strict=True))


BASELINE_WEIGHTS = PieceWeights(100, 320, 330, 500, 900)


@dataclass
class Individual:
    id: str
    weights: PieceWeights
    fitness: int | None
    generation: int
    wins: int = 0
    draws: int = 0
    losses: int = 0
    games_played: int = 0
    elite_from: str | None = None
    bank_source_experiment_id: str | None = None
    bank_source_generation: int | None = None
    bank_source_individual_id: str | None = None
    bank_source_lineage: dict[str, Any] | None = None


@dataclass(frozen=True)
class MutationEvent:
    gene: str
    old_value: int
    new_value: int
    delta: int


@dataclass(frozen=True)
class GeneOrigin:
    gene: str
    value: int
    source_parent: Literal["parent_a", "parent_b"]
    source_parent_id: str


@dataclass(frozen=True)
class ParentPair:
    parent_a_id: str
    parent_b_id: str


@dataclass
class ChildEvent:
    child_id: str
    generation: int
    individual: Individual
    parent_a_id: str
    parent_b_id: str
    crossover_weights: PieceWeights
    pre_mutation_weights: PieceWeights
    post_mutation_weights: PieceWeights
    gene_origins: list[GeneOrigin]
    mutations: list[MutationEvent] = field(default_factory=list)


@dataclass
class EvolutionStep:
    generation: int
    population: list[Individual]
    selected_parents: list[str]
    parent_pairs: list[ParentPair]
    children: list[ChildEvent]
    elite_individuals: list[Individual]
    best_individual: Individual
    average_fitness: float


@dataclass(frozen=True)
class GameMoveTrace:
    ply: int
    move_number: int
    color: Literal["white", "black"]
    san: str
    uci: str
    fen: str
    evaluation: float
    nodes: int
    depth: int
    search_budget_ms: float = 0.0
    search_time_ms: float = 0.0
    policy_time_ms: float = 0.0
    remaining_time_ms: int = 0


@dataclass
class CandidateGameTrace:
    generation: int
    candidate_id: str
    candidate_weights: PieceWeights
    candidate_color: Literal["white", "black"]
    game_index: int
    games_total: int
    baseline_weights: PieceWeights
    opponent_id: str = "BASELINE"
    opponent_type: Literal["baseline", "population", "elite"] = "baseline"
    opponent_weights: PieceWeights | None = None
    result: Literal["win", "draw", "loss"] | None = None
    plies: int = 0
    fitness_delta: int = 0
    termination: str | None = None
    moves: list[GameMoveTrace] = field(default_factory=list)


@dataclass(frozen=True)
class ExperimentConfig:
    population_size: int = 10
    games_per_individual: int = 8
    time_control_ms: int = 5_000
    mutation_rate: float = 0.15
    mutation_strength: int = 25
    elite_count: int = 2
    tournament_size: int = 3
    max_plies: int = 200
    generations: int = 20
    seed: int | None = None


@dataclass
class EvolutionState:
    status: Literal["ready", "running", "complete"]
    config: ExperimentConfig
    generation: int = 0
    population: list[Individual] = field(default_factory=list)
    history: list[EvolutionStep] = field(default_factory=list)
    game_traces: list[CandidateGameTrace] = field(default_factory=list)
