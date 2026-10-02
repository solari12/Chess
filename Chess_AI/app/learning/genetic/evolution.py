"""One-generation-at-a-time genetic evolution coordinator."""

import random
from dataclasses import asdict
from typing import Callable

from app.learning.genetic.crossover import uniform_crossover
from app.learning.genetic.fitness import FitnessOpponent, evaluate_fitness
from app.learning.genetic.models import (
    ChildEvent,
    EvolutionState,
    EvolutionStep,
    ExperimentConfig,
    Individual,
    ParentPair,
    BASELINE_WEIGHTS,
)
from app.learning.genetic.mutation import mutate
from app.learning.genetic.population import create_initial_population
from app.learning.genetic.selection import tournament_select

EventCallback = Callable[[dict[str, object]], None]


class GeneticEvolution:
    """Stateful experiment that exposes its complete generation history."""

    def __init__(self, config: ExperimentConfig):
        self._validate_config(config)
        self.config = config
        self.rng = random.Random(config.seed)
        opponent_seed = None if config.seed is None else config.seed + 1_000_003
        self.opponent_rng = random.Random(opponent_seed)
        self.next_individual_id = 1
        self.state = EvolutionState(status="ready", config=config)

    @staticmethod
    def _validate_config(config: ExperimentConfig) -> None:
        if config.population_size < 2:
            raise ValueError("population_size must be at least 2")
        if config.games_per_individual < 4 or config.games_per_individual % 2:
            raise ValueError("games_per_individual must be even and at least 4")
        if config.population_size < config.games_per_individual // 2:
            raise ValueError("population_size must provide enough unique opponents for games_per_individual")
        if not 1_000 <= config.time_control_ms <= 600_000:
            raise ValueError("time_control_ms must be between 1000 and 600000")
        if not 0 <= config.mutation_rate <= 1:
            raise ValueError("mutation_rate must be between 0 and 1")
        if config.mutation_strength < 1:
            raise ValueError("mutation_strength must be positive")
        if not 0 <= config.elite_count < config.population_size:
            raise ValueError("elite_count must be between 0 and population_size - 1")
        if not 1 <= config.tournament_size <= config.population_size:
            raise ValueError("tournament_size must be within the population size")
        if config.max_plies < 1 or config.generations < 1:
            raise ValueError("max_plies and generations must be positive")

    def _new_id(self) -> str:
        individual_id = f"I{self.next_individual_id:04d}"
        self.next_individual_id += 1
        return individual_id

    def _evaluate(
        self,
        individual: Individual,
        *,
        candidate_index: int,
        candidates_total: int,
        completed_candidates: int,
        population: list[Individual],
        previous_top: list[Individual],
        on_event: EventCallback | None,
    ) -> None:
        _emit(
            on_event,
            type="candidate_started",
            generation=individual.generation,
            candidate_id=individual.id,
            candidate_weights=individual.weights.as_dict(),
            candidate_index=candidate_index,
            candidates_total=candidates_total,
            completed_candidates=completed_candidates,
        )

        def forward(event: dict[str, object]) -> None:
            if on_event:
                on_event({
                    **event,
                    "candidate_index": candidate_index,
                    "candidates_total": candidates_total,
                    "completed_candidates": completed_candidates,
                })

        result = evaluate_fitness(
            individual.weights,
            games_per_individual=self.config.games_per_individual,
            time_control_ms=self.config.time_control_ms,
            max_plies=self.config.max_plies,
            generation=individual.generation,
            candidate_id=individual.id,
            opponents=self._opponent_pool(individual, population, previous_top),
            on_event=forward,
        )
        individual.fitness = result.fitness
        individual.wins = result.wins
        individual.draws = result.draws
        individual.losses = result.losses
        individual.games_played = len(result.games)
        self.state.game_traces.extend(game.trace for game in result.games)
        _emit(
            on_event,
            type="candidate_finished",
            generation=individual.generation,
            candidate_id=individual.id,
            record={"wins": result.wins, "draws": result.draws, "losses": result.losses, "games_played": len(result.games)},
            fitness=result.fitness,
            candidate_index=candidate_index,
            candidates_total=candidates_total,
            completed_candidates=candidate_index,
        )

    @staticmethod
    def _rank(population: list[Individual]) -> list[Individual]:
        return sorted(population, key=lambda item: item.fitness or 0, reverse=True)

    def _opponent_pool(
        self,
        candidate: Individual,
        population: list[Individual],
        previous_top: list[Individual],
    ) -> list[FitnessOpponent]:
        """Sample unique opponents, always retaining BASELINE in the pool."""
        opponent_slots = self.config.games_per_individual // 2
        opponents = [FitnessOpponent("BASELINE", "baseline", BASELINE_WEIGHTS)]
        opponent_count = opponent_slots - 1
        if previous_top:
            # Fitness is known after generation one: draw from the five strongest
            # members of the previous generation, without replacement.
            candidates = previous_top[:5]
            opponent_type = "elite"
            id_prefix = "ELITE-"
        else:
            # Generation one has no ranking yet, so sample from this population.
            candidates = [item for item in population if item.id != candidate.id]
            opponent_type = "population"
            id_prefix = ""
        if len(candidates) < opponent_count:
            raise ValueError("not enough unique candidates for the configured opponent pool")
        selected = self.opponent_rng.sample(candidates, opponent_count)
        opponents.extend(
            FitnessOpponent(f"{id_prefix}{item.id}", opponent_type, item.weights)
            for item in selected
        )
        return opponents

    @staticmethod
    def _make_step(
        generation: int,
        population: list[Individual],
        selected_parents: list[str],
        parent_pairs: list[ParentPair],
        children: list[ChildEvent],
        elites: list[Individual],
    ) -> EvolutionStep:
        ranked = GeneticEvolution._rank(population)
        average_fitness = sum(ind.fitness or 0 for ind in population) / len(population)
        return EvolutionStep(
            generation=generation,
            population=list(population),
            selected_parents=selected_parents,
            parent_pairs=parent_pairs,
            children=children,
            elite_individuals=elites,
            best_individual=ranked[0],
            average_fitness=average_fitness,
        )

    def start(self, on_event: EventCallback | None = None) -> EvolutionStep:
        if self.state.generation != 0:
            raise ValueError("experiment has already started")

        population = create_initial_population(
            self.config.population_size,
            self.rng,
            generation=1,
            first_id=self.next_individual_id,
        )
        self.next_individual_id += len(population)
        self.state.status = "running"
        self.state.game_traces.clear()
        _emit(
            on_event,
            type="evaluation_started",
            generation=1,
            candidates_total=len(population),
            completed_candidates=0,
            games_per_individual=self.config.games_per_individual,
        )
        for index, individual in enumerate(population, start=1):
            self._evaluate(
                individual,
                candidate_index=index,
                candidates_total=len(population),
                completed_candidates=index - 1,
                population=population,
                previous_top=[],
                on_event=on_event,
            )

        self.state.population = population
        self.state.generation = 1
        if self.config.generations == 1:
            self.state.status = "complete"
        else:
            self.state.status = "ready"
        step = self._make_step(1, population, [], [], [], [])
        self.state.history.append(step)
        _emit(
            on_event,
            type="generation_complete",
            generation=1,
            completed_candidates=len(population),
            candidates_total=len(population),
        )
        return step

    def step_generation(self, on_event: EventCallback | None = None) -> EvolutionStep:
        if self.state.generation == 0:
            return self.start(on_event)
        if self.state.status == "complete":
            raise ValueError("experiment is complete")

        next_generation = self.state.generation + 1
        ranked = self._rank(self.state.population)
        elite_sources = ranked[: self.config.elite_count]
        previous_top = ranked[:5]
        elites: list[Individual] = []
        next_population: list[Individual] = []
        for source in elite_sources:
            elite = Individual(
                id=self._new_id(),
                weights=source.weights,
                fitness=source.fitness,
                generation=next_generation,
                wins=source.wins,
                draws=source.draws,
                losses=source.losses,
                games_played=source.games_played,
                elite_from=source.id,
            )
            elites.append(elite)
            next_population.append(elite)

        selected_parents: list[str] = []
        parent_pairs: list[ParentPair] = []
        children: list[ChildEvent] = []

        while len(next_population) < self.config.population_size:
            parent_a = tournament_select(
                self.state.population, self.config.tournament_size, self.rng
            )
            parent_b_candidates = [
                individual
                for individual in self.state.population
                if individual.id != parent_a.id
            ]
            parent_b = tournament_select(
                parent_b_candidates,
                min(self.config.tournament_size, len(parent_b_candidates)),
                self.rng,
            )
            selected_parents.extend((parent_a.id, parent_b.id))
            parent_pairs.append(ParentPair(parent_a.id, parent_b.id))

            crossover = uniform_crossover(parent_a, parent_b, self.rng)
            child_weights, mutations = mutate(
                crossover.weights,
                self.config.mutation_rate,
                self.config.mutation_strength,
                self.rng,
            )
            child = Individual(
                id=self._new_id(),
                weights=child_weights,
                fitness=None,
                generation=next_generation,
            )
            next_population.append(child)
            children.append(
                ChildEvent(
                    child_id=child.id,
                    generation=next_generation,
                    individual=child,
                    parent_a_id=parent_a.id,
                    parent_b_id=parent_b.id,
                    crossover_weights=crossover.weights,
                    pre_mutation_weights=crossover.weights,
                    post_mutation_weights=child_weights,
                    gene_origins=crossover.gene_origins,
                    mutations=mutations,
                )
            )

        _emit(
            on_event,
            type="generation_lineage",
            lineage={
                "generation": next_generation,
                "population": [asdict(individual) for individual in next_population],
                "children": [asdict(child) for child in children],
                "elite_individuals": [asdict(elite) for elite in elites],
            },
        )

        self.state.game_traces.clear()
        self.state.status = "running"
        _emit(
            on_event,
            type="evaluation_started",
            generation=next_generation,
            candidates_total=len(next_population),
            completed_candidates=len(elites),
            games_per_individual=self.config.games_per_individual,
        )
        for child_index, individual in enumerate(next_population, start=1):
            self._evaluate(
                individual,
                candidate_index=child_index,
                candidates_total=len(next_population),
                completed_candidates=child_index - 1,
                population=next_population,
                previous_top=previous_top,
                on_event=on_event,
            )

        self.state.population = next_population
        self.state.generation = next_generation
        if next_generation >= self.config.generations:
            self.state.status = "complete"
        else:
            self.state.status = "ready"
        step = self._make_step(
            next_generation,
            next_population,
            list(dict.fromkeys(selected_parents)),
            parent_pairs,
            children,
            elites,
        )
        self.state.history.append(step)
        _emit(
            on_event,
            type="generation_complete",
            generation=next_generation,
            completed_candidates=len(next_population),
            candidates_total=len(next_population),
        )
        return step

    def run_generations(self, count: int) -> list[EvolutionStep]:
        if count < 1:
            raise ValueError("count must be positive")
        steps: list[EvolutionStep] = []
        for _ in range(count):
            if self.state.generation == 0:
                steps.append(self.start())
            elif self.state.status != "complete":
                steps.append(self.step_generation())
            else:
                break
        return steps


def _emit(callback: EventCallback | None, **event: object) -> None:
    if callback is not None:
        callback(event)
