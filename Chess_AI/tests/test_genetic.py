import asyncio
import json
import random
import unittest

import chess

from app.api.genetic_lab import (
    GeneticMoveRequest,
    GeneticWeightsRequest,
    StartExperimentRequest,
    get_experiment_state,
    play_genetic_move,
    reset_experiment,
    start_experiment,
    start_experiment_stream,
    step_experiment,
)
from app.learning.genetic.crossover import uniform_crossover
from app.learning.genetic.evolution import GeneticEvolution
from app.learning.genetic.fitness import FitnessOpponent, evaluate_fitness, evaluate_with_weights
from app.learning.genetic.models import ExperimentConfig, Individual, PieceWeights
from app.learning.genetic.mutation import mutate
from app.learning.genetic.population import WEIGHT_BOUNDS, create_initial_population
from app.learning.genetic.selection import tournament_select


class GeneticModelTests(unittest.TestCase):
    def test_piece_weights_reject_non_integer_values(self) -> None:
        with self.assertRaises(TypeError):
            PieceWeights(100, 320, 330, 500, 900.5)  # type: ignore[arg-type]

    def test_population_is_seeded_integer_and_within_gene_bounds(self) -> None:
        first = create_initial_population(10, random.Random(42))
        second = create_initial_population(10, random.Random(42))

        self.assertEqual([item.weights for item in first], [item.weights for item in second])
        self.assertEqual(len({item.id for item in first}), 10)
        for individual in first:
            for gene, value in individual.weights.as_dict().items():
                low, high = WEIGHT_BOUNDS[gene]
                self.assertIs(type(value), int)
                self.assertGreaterEqual(value, low)
                self.assertLessEqual(value, high)

    def test_tournament_selection_picks_fittest_contestant(self) -> None:
        population = [
            Individual(f"I{i}", PieceWeights(100, 320, 330, 500, 900), fitness, 1)
            for i, fitness in enumerate((-1, 0, 3))
        ]
        self.assertEqual(tournament_select(population, 3, random.Random(5)).id, "I2")

    def test_uniform_crossover_records_parent_source_for_every_gene(self) -> None:
        parent_a = Individual("A", PieceWeights(100, 200, 300, 400, 800), 1, 1)
        parent_b = Individual("B", PieceWeights(150, 250, 350, 450, 950), 0, 1)

        result = uniform_crossover(parent_a, parent_b, random.Random(7))
        values = result.weights.as_dict()

        self.assertEqual(len(result.gene_origins), 5)
        for origin in result.gene_origins:
            source = parent_a if origin.source_parent == "parent_a" else parent_b
            self.assertEqual(origin.source_parent_id, source.id)
            self.assertEqual(origin.value, source.weights.as_dict()[origin.gene])
            self.assertEqual(values[origin.gene], origin.value)

    def test_mutation_records_changes_and_clamps_to_bounds(self) -> None:
        mutated, events = mutate(
            PieceWeights(100, 320, 330, 500, 900),
            mutation_rate=1,
            mutation_strength=25,
            rng=random.Random(11),
        )

        self.assertEqual(len(events), 5)
        for event in events:
            self.assertEqual(event.new_value - event.old_value, event.delta)
            low, high = WEIGHT_BOUNDS[event.gene]
            self.assertGreaterEqual(event.new_value, low)
            self.assertLessEqual(event.new_value, high)
        for gene, value in mutated.as_dict().items():
            self.assertGreaterEqual(value, WEIGHT_BOUNDS[gene][0])
            self.assertLessEqual(value, WEIGHT_BOUNDS[gene][1])


class GeneticFitnessTests(unittest.TestCase):
    def test_candidate_weights_use_engine_material_evaluator_and_fixed_king(self) -> None:
        board = chess.Board("7k/8/8/8/8/8/P7/K7 w - - 0 1")
        self.assertEqual(evaluate_with_weights(board, PieceWeights(155, 320, 330, 500, 900)), 155)

    def test_games_are_color_balanced_and_stop_at_ply_limit(self) -> None:
        result = evaluate_fitness(
            PieceWeights(100, 320, 330, 500, 900),
            games_per_individual=2,
            search_depth=1,
            max_plies=1,
        )

        self.assertEqual([game.candidate_color for game in result.games], ["white", "black"])
        self.assertEqual([game.outcome for game in result.games], ["draw", "draw"])
        self.assertTrue(all(game.plies == 1 and game.termination == "ply_limit" for game in result.games))
        self.assertEqual((result.fitness, result.wins, result.draws, result.losses), (0, 0, 2, 0))

    def test_game_trace_and_live_events_follow_play_order(self) -> None:
        events: list[dict[str, object]] = []
        result = evaluate_fitness(
            PieceWeights(100, 320, 330, 500, 900),
            games_per_individual=2,
            search_depth=1,
            max_plies=2,
            candidate_id="I0002",
            on_event=events.append,
        )

        event_types = [event["type"] for event in events]
        self.assertEqual(event_types[0], "game_started")
        self.assertEqual(event_types[-1], "candidate_game_finished")
        for game_index in (1, 2):
            started = next(i for i, event in enumerate(events) if event.get("type") == "game_started" and event.get("game_index") == game_index)
            finished = next(i for i, event in enumerate(events) if event.get("type") == "game_finished" and event.get("game_index") == game_index)
            move_indexes = [i for i, event in enumerate(events) if event.get("type") == "move_played" and event.get("game_index") == game_index]
            self.assertLess(started, move_indexes[0])
            self.assertLess(move_indexes[-1], finished)
            self.assertEqual([result.games[game_index - 1].trace.moves[index].ply for index in range(len(move_indexes))], list(range(1, len(move_indexes) + 1)))
            for move in result.games[game_index - 1].trace.moves:
                self.assertTrue(chess.Board(move.fen).is_valid())

        self.assertEqual(result.fitness, result.wins - result.losses)
        self.assertEqual([game.trace.result for game in result.games], [game.outcome for game in result.games])
        started = next(event for event in events if event["type"] == "game_started")
        self.assertEqual((started["opponent_id"], started["opponent_type"], started["color"]), ("BASELINE", "baseline", "white"))

    def test_fitness_aggregates_multiple_opponents_with_both_candidate_colors(self) -> None:
        opponents = [
            FitnessOpponent("BASELINE", "baseline", PieceWeights(100, 320, 330, 500, 900)),
            FitnessOpponent("I0007", "population", PieceWeights(110, 300, 350, 480, 920)),
        ]
        result = evaluate_fitness(
            PieceWeights(120, 310, 340, 510, 880),
            games_per_individual=4,
            search_depth=1,
            max_plies=1,
            opponents=opponents,
        )

        self.assertEqual([(game.opponent_id, game.candidate_color) for game in result.games], [
            ("BASELINE", "white"), ("BASELINE", "black"), ("I0007", "white"), ("I0007", "black"),
        ])
        self.assertTrue(all(game.trace.opponent_weights is not None for game in result.games))
        self.assertEqual(result.fitness, result.wins - result.losses)


class GeneticEvolutionTests(unittest.TestCase):
    @staticmethod
    def config(**changes: object) -> ExperimentConfig:
        options: dict[str, object] = {
            "population_size": 4,
            "games_per_individual": 8,
            "search_depth": 1,
            "mutation_rate": 1.0,
            "mutation_strength": 10,
            "elite_count": 1,
            "tournament_size": 2,
            "max_plies": 2,
            "generations": 3,
            "seed": 42,
        }
        options.update(changes)
        return ExperimentConfig(**options)  # type: ignore[arg-type]

    def test_generation_exposes_children_crossover_mutation_and_elites(self) -> None:
        evolution = GeneticEvolution(self.config())
        initial = evolution.start()
        generation_events: list[dict[str, object]] = []
        next_step = evolution.step_generation(on_event=generation_events.append)

        self.assertEqual(initial.generation, 1)
        self.assertEqual(next_step.generation, 2)
        self.assertEqual(len(next_step.population), 4)
        self.assertEqual(len(next_step.elite_individuals), 1)
        self.assertEqual(len(next_step.children), 3)
        self.assertEqual(len(next_step.parent_pairs), 3)
        self.assertEqual(
            set(next_step.selected_parents),
            {parent for pair in next_step.parent_pairs for parent in (pair.parent_a_id, pair.parent_b_id)},
        )
        source = initial.best_individual
        previous_top_ids = {item.id for item in sorted(initial.population, key=lambda item: item.fitness or 0, reverse=True)[:5]}
        elite = next_step.elite_individuals[0]
        self.assertEqual(elite.elite_from, source.id)
        self.assertEqual(elite.weights, source.weights)
        started_games = [event for event in generation_events if event["type"] == "game_started"]
        self.assertEqual(len(started_games), len(next_step.population) * self.config().games_per_individual)
        for individual in next_step.population:
            matchups = [event for event in started_games if event["candidate_id"] == individual.id]
            self.assertEqual(len(matchups), 8)
            self.assertEqual({event["opponent_type"] for event in matchups}, {"baseline", "elite"})
            self.assertEqual({event["color"] for event in matchups}, {"white", "black"})
            sampled_elites = {event["opponent_id"] for event in matchups if event["opponent_type"] == "elite"}
            self.assertEqual(len(sampled_elites), 3)
            self.assertTrue(sampled_elites.issubset({f"ELITE-{item_id}" for item_id in previous_top_ids}))
        for child in next_step.children:
            self.assertEqual(len(child.gene_origins), 5)
            self.assertEqual(len(child.mutations), 5)
            for gene, value in child.individual.weights.as_dict().items():
                low, high = WEIGHT_BOUNDS[gene]
                self.assertGreaterEqual(value, low)
                self.assertLessEqual(value, high)

    def test_same_seed_reproduces_population_and_events(self) -> None:
        def run_once() -> tuple[object, ...]:
            evolution = GeneticEvolution(self.config())
            evolution.start()
            step = evolution.step_generation()
            return (
                [(item.id, item.weights, item.fitness) for item in step.population],
                step.selected_parents,
                step.parent_pairs,
                [(child.parent_a_id, child.parent_b_id, child.gene_origins, child.mutations) for child in step.children],
            )

        self.assertEqual(run_once(), run_once())

    def test_opponents_are_sampled_without_replacement_from_population_then_previous_top_five(self) -> None:
        evolution = GeneticEvolution(self.config(population_size=6, generations=2, max_plies=1))
        generation_one_events: list[dict[str, object]] = []
        first = evolution.start(on_event=generation_one_events.append)
        generation_two_events: list[dict[str, object]] = []
        second = evolution.step_generation(on_event=generation_two_events.append)

        previous_top_five = {
            item.id
            for item in sorted(first.population, key=lambda item: item.fitness or 0, reverse=True)[:5]
        }
        for candidate, events, expected_type, eligible_ids in [
            (item, generation_one_events, "population", {item.id for item in first.population})
            for item in first.population
        ]:
            games = [event for event in events if event.get("type") == "game_started" and event.get("candidate_id") == candidate.id]
            sampled = {event["opponent_id"] for event in games if event["opponent_type"] != "baseline"}
            self.assertEqual(len(sampled), 3)
            self.assertEqual({event["opponent_type"] for event in games}, {"baseline", expected_type})
            self.assertNotIn(candidate.id, sampled)
            self.assertTrue(sampled.issubset(eligible_ids))

        for candidate in second.population:
            games = [event for event in generation_two_events if event.get("type") == "game_started" and event.get("candidate_id") == candidate.id]
            sampled = {str(event["opponent_id"]).removeprefix("ELITE-") for event in games if event["opponent_type"] != "baseline"}
            self.assertEqual(len(sampled), 3)
            self.assertTrue(sampled.issubset(previous_top_five))

    def test_candidate_and_generation_completion_events_are_ordered(self) -> None:
        events: list[dict[str, object]] = []
        evolution = GeneticEvolution(self.config(population_size=2, games_per_individual=4, generations=1, max_plies=1))
        step = evolution.start(on_event=events.append)

        event_types = [event["type"] for event in events]
        self.assertEqual(event_types[0], "evaluation_started")
        self.assertEqual(event_types[-1], "generation_complete")
        self.assertEqual(event_types.count("candidate_started"), 2)
        self.assertEqual(event_types.count("candidate_finished"), 2)
        for candidate in step.population:
            started = next(i for i, event in enumerate(events) if event.get("type") == "candidate_started" and event.get("candidate_id") == candidate.id)
            finished = next(i for i, event in enumerate(events) if event.get("type") == "candidate_finished" and event.get("candidate_id") == candidate.id)
            self.assertLess(started, finished)
            self.assertEqual(candidate.fitness, candidate.wins - candidate.losses)
        self.assertEqual(len(evolution.state.game_traces), 8)
        self.assertEqual({trace.opponent_id for trace in evolution.state.game_traces if trace.candidate_id == step.population[0].id}, {"BASELINE", next(item.id for item in step.population if item.id != step.population[0].id)})


class GeneticLabApiTests(unittest.TestCase):
    def tearDown(self) -> None:
        reset_experiment()

    def test_start_step_state_and_reset_endpoints_expose_history(self) -> None:
        request = StartExperimentRequest(
            population_size=2,
            games_per_individual=4,
            search_depth=1,
            mutation_rate=0.5,
            mutation_strength=5,
            elite_count=0,
            tournament_size=2,
            max_plies=1,
            generations=2,
            seed=9,
        )

        started = start_experiment(request)
        self.assertEqual(started["generation"], 1)
        self.assertEqual(len(started["population"]), 2)
        stepped = step_experiment()
        self.assertEqual(stepped["generation"], 2)
        self.assertEqual(len(stepped["history"]), 2)
        self.assertEqual(get_experiment_state()["generation"], 2)
        self.assertEqual(reset_experiment()["status"], "ready")

    def test_start_stream_emits_events_and_finishes_with_replay_state(self) -> None:
        request = {
            "population_size": 2,
            "games_per_individual": 4,
            "search_depth": 1,
            "mutation_rate": 0.5,
            "mutation_strength": 5,
            "elite_count": 0,
            "tournament_size": 2,
            "max_plies": 1,
            "generations": 1,
            "seed": 9,
        }
        response = start_experiment_stream(StartExperimentRequest(**request))

        self.assertTrue(response.media_type.startswith("text/event-stream"))

        async def consume_stream() -> str:
            chunks: list[str] = []
            async for chunk in response.body_iterator:
                chunks.append(chunk.decode() if isinstance(chunk, bytes) else chunk)
            return "".join(chunks)

        body = asyncio.run(consume_stream())
        events = [json.loads(frame.removeprefix("data: ")) for frame in body.strip().split("\n\n") if frame.startswith("data: ")]
        event_types = [event["type"] for event in events]
        self.assertEqual(event_types[0], "evaluation_started")
        self.assertIn("game_started", event_types)
        self.assertIn("move_played", event_types)
        self.assertIn("candidate_game_finished", event_types)
        self.assertEqual(event_types[-2:], ["generation_complete", "state"])
        state_event = events[-1]["state"]
        self.assertEqual(state_event["status"], "complete")
        self.assertEqual(len(state_event["game_traces"]), 8)
        game_started = next(event for event in events if event["type"] == "game_started")
        self.assertEqual(game_started["opponent_id"], "BASELINE")
        self.assertEqual(game_started["opponent_type"], "baseline")
        self.assertTrue(any(event.get("opponent_type") == "population" for event in events if event["type"] == "game_started"))

    def test_trained_candidate_returns_a_legal_chess_move(self) -> None:
        fen = chess.STARTING_FEN
        response = play_genetic_move(
            GeneticMoveRequest(
                fen=fen,
                depth=1,
                weights=GeneticWeightsRequest(
                    pawn=90,
                    knight=340,
                    bishop=320,
                    rook=510,
                    queen=930,
                ),
            )
        )

        self.assertEqual(response["algorithm"], "genetic")
        self.assertIn(chess.Move.from_uci(response["move"]), chess.Board(fen).legal_moves)


if __name__ == "__main__":
    unittest.main()
