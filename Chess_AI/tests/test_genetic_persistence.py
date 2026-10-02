"""Checkpoint, resume, and reusable-genome persistence coverage."""

from __future__ import annotations

import asyncio
import json
import random
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from app.api import genetic_lab as genetic_lab_api
from app.api.genetic_lab import CandidateSetSaveRequest, ExperimentReference, StartExperimentRequest, StartFromBankRequest
from app.learning.genetic.evolution import GeneticEvolution
from app.learning.genetic.models import ExperimentConfig, Individual, PieceWeights
from app.learning.genetic import persistence
from app.learning.genetic.persistence import (
    bank_individuals,
    checkpoint_for_experiment,
    individuals_from_bank,
    load_checkpoint,
    restore_experiment,
    save_bank_individuals,
    save_checkpoint,
)
from app.learning.time_management.runtime_policy import TimeManagementDecision


class GeneticPersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root_patcher = patch.object(persistence, "DATA_ROOT", Path(self.temp.name))
        self.root_patcher.start()
        self.addCleanup(self.root_patcher.stop)
        self.addCleanup(self.temp.cleanup)
        decision = TimeManagementDecision(
            predicted_time_ms=None,
            final_search_budget_ms=0.0,
            safety_cap_applied=True,
            fallback_used=True,
            fallback_reason="test_zero_budget",
            probe_time_ms=0.0,
            inference_latency_ms=0.0,
            policy_latency_ms=0.0,
            model_version=None,
        )
        policy_patcher = patch("app.learning.genetic.fitness.decide_search_budget", return_value=decision)
        policy_patcher.start()
        self.addCleanup(policy_patcher.stop)

    @staticmethod
    def config(**changes: object) -> ExperimentConfig:
        values: dict[str, object] = {
            "population_size": 4,
            "games_per_individual": 4,
            "time_control_ms": 1_000,
            "mutation_rate": 0.25,
            "mutation_strength": 5,
            "elite_count": 1,
            "tournament_size": 2,
            "max_plies": 1,
            "generations": 4,
            "seed": 2026,
        }
        values.update(changes)
        return ExperimentConfig(**values)  # type: ignore[arg-type]

    def completed_experiment(self) -> GeneticEvolution:
        experiment = GeneticEvolution(self.config())
        experiment.start()
        experiment.step_generation()
        return experiment

    def test_checkpoint_roundtrip_preserves_state_lineage_rng_and_next_generation(self) -> None:
        experiment = self.completed_experiment()
        saved = save_checkpoint(checkpoint_for_experiment(experiment))
        loaded = restore_experiment(load_checkpoint(saved["experiment_id"]))

        self.assertEqual(loaded.experiment_id, experiment.experiment_id)
        self.assertEqual(loaded.state.generation, experiment.state.generation)
        self.assertEqual(loaded.state.status, "ready")
        self.assertEqual(loaded.next_individual_id, experiment.next_individual_id)
        self.assertEqual(loaded.rng.getstate(), experiment.rng.getstate())
        self.assertEqual(loaded.opponent_rng.getstate(), experiment.opponent_rng.getstate())
        self.assertEqual(
            [(item.id, item.weights, item.fitness, item.wins, item.draws, item.losses, item.games_played) for item in loaded.state.population],
            [(item.id, item.weights, item.fitness, item.wins, item.draws, item.losses, item.games_played) for item in experiment.state.population],
        )
        self.assertEqual([step.generation for step in loaded.state.history], [1, 2])
        loaded_children = loaded.state.history[-1].children
        original_children = experiment.state.history[-1].children
        self.assertEqual([asdict(child) for child in loaded_children], [asdict(child) for child in original_children])
        self.assertTrue(all(not trace.moves for trace in loaded.state.game_traces))

        expected_next = experiment.step_generation()
        resumed_next = loaded.step_generation()
        self.assertEqual(resumed_next.generation, 3)
        self.assertEqual([item.id for item in resumed_next.population], [item.id for item in expected_next.population])
        self.assertEqual([item.weights for item in resumed_next.population], [item.weights for item in expected_next.population])
        self.assertEqual(resumed_next.parent_pairs, expected_next.parent_pairs)
        self.assertEqual([step.generation for step in loaded.state.history], [1, 2, 3])

    def test_checkpoint_rejects_unsupported_schema_version(self) -> None:
        checkpoint = checkpoint_for_experiment(self.completed_experiment())
        checkpoint["schema_version"] = 999
        with self.assertRaisesRegex(ValueError, "schema version"):
            restore_experiment(checkpoint)

    def test_save_and_load_api_restore_the_selected_experiment(self) -> None:
        experiment = self.completed_experiment()
        with (
            patch.object(genetic_lab_api, "_experiment", experiment),
            patch.object(genetic_lab_api, "_stream_active", False),
        ):
            summary = genetic_lab_api.save_current_experiment()
            replacement = GeneticEvolution(self.config(seed=99))
            with patch.object(genetic_lab_api, "_experiment", replacement):
                state = genetic_lab_api.load_saved_experiment(ExperimentReference(experiment_id=summary["experiment_id"]))

        self.assertEqual(state["experiment_id"], experiment.experiment_id)
        self.assertEqual(state["generation"], 2)
        self.assertEqual([step["generation"] for step in state["history"]], [1, 2])
        self.assertEqual(state["population"][0]["weights"], asdict(experiment.state.population[0].weights))

    def test_tied_best_candidates_are_saved_with_lineage_and_candidate_set(self) -> None:
        experiment = GeneticEvolution(self.config(elite_count=0))
        experiment.start()
        with (
            patch.object(genetic_lab_api, "_experiment", experiment),
            patch.object(genetic_lab_api, "_stream_active", False),
        ):
            result = genetic_lab_api.save_tied_best_candidate_set(CandidateSetSaveRequest(name="all tied best"))

        self.assertEqual(len(result["individuals"]), 4)
        self.assertEqual(len(result["candidate_set"]["bank_ids"]), 4)
        self.assertTrue(all(item["fitness"] == 0 for item in result["individuals"]))
        self.assertEqual({item["source_experiment_id"] for item in result["individuals"]}, {experiment.experiment_id})
        self.assertEqual(len(bank_individuals()), 4)
        self.assertEqual(result["candidate_set"]["schema_version"], persistence.SCHEMA_VERSION)

    def test_start_from_bank_keeps_saved_genomes_and_assigns_fresh_ids_to_fillers(self) -> None:
        source_weights = PieceWeights(96, 310, 324, 558, 889)
        record = {
            "individual_id": "I0035",
            "source_experiment_id": "exp_source_001",
            "source_generation": 5,
            "chromosome": source_weights.as_dict(),
            "fitness": 4,
            "wins": 6,
            "draws": 0,
            "losses": 2,
            "games_played": 8,
            "lineage": {"kind": "offspring", "parent_a_id": "I0010", "parent_b_id": "I0011"},
        }
        saved = save_bank_individuals([record])[0]
        immutable_bank_copy = json.loads(json.dumps(saved))
        seeds = individuals_from_bank([saved["bank_id"]])
        experiment = GeneticEvolution(self.config(), initial_candidates=seeds)
        initial = experiment.start()

        seed = next(item for item in initial.population if item.id == "I0035")
        self.assertEqual(seed.weights, source_weights)
        self.assertEqual(seed.generation, 1)
        self.assertEqual(seed.fitness, 0)
        self.assertEqual(seed.games_played, 4)
        self.assertEqual(seed.bank_source_experiment_id, "exp_source_001")
        self.assertEqual(seed.bank_source_generation, 5)
        self.assertEqual(seed.bank_source_lineage, record["lineage"])
        self.assertEqual(len(initial.population), 4)
        self.assertEqual([item.id for item in initial.population], ["I0035", "I0036", "I0037", "I0038"])
        self.assertNotEqual(experiment.experiment_id, "exp_source_001")
        self.assertEqual(bank_individuals()[0], immutable_bank_copy)

    def test_duplicate_source_ids_are_rejected_when_building_a_seed_population(self) -> None:
        weights = PieceWeights(100, 320, 330, 500, 900).as_dict()
        records = [
            {"individual_id": "I0001", "source_experiment_id": f"exp_{suffix}", "source_generation": 1,
             "chromosome": weights, "fitness": 0, "wins": 0, "draws": 4, "losses": 0, "games_played": 4, "lineage": None}
            for suffix in ("a", "b")
        ]
        saved = save_bank_individuals(records)
        with self.assertRaisesRegex(ValueError, "reuse an individual ID"):
            individuals_from_bank([item["bank_id"] for item in saved])

    def test_start_from_bank_stream_creates_a_new_experiment_without_editing_bank(self) -> None:
        source = save_bank_individuals([{
            "individual_id": "I0042",
            "source_experiment_id": "exp_bank_source",
            "source_generation": 4,
            "chromosome": {"pawn": 105, "knight": 330, "bishop": 325, "rook": 515, "queen": 910},
            "fitness": 3,
            "wins": 5,
            "draws": 1,
            "losses": 2,
            "games_played": 8,
            "lineage": {"kind": "elite_clone", "parent_id": "I0038"},
        }])[0]
        bank_before = json.loads(json.dumps(bank_individuals()))
        request = StartFromBankRequest(
            config=StartExperimentRequest(**asdict(self.config(generations=2))),
            bank_ids=[source["bank_id"]],
        )
        with (
            patch.object(genetic_lab_api, "_experiment", None),
            patch.object(genetic_lab_api, "_stream_active", False),
        ):
            response = genetic_lab_api.start_from_bank_stream(request)

            async def consume() -> str:
                chunks = []
                async for chunk in response.body_iterator:
                    chunks.append(chunk.decode() if isinstance(chunk, bytes) else chunk)
                return "".join(chunks)

            body = asyncio.run(consume())

        events = [json.loads(frame.removeprefix("data: ")) for frame in body.strip().split("\n\n") if frame.startswith("data: ")]
        state = next(event["state"] for event in events if event["type"] == "state")
        seeded = next(item for item in state["population"] if item["id"] == "I0042")
        self.assertEqual(seeded["weights"], source["chromosome"])
        self.assertEqual(seeded["bank_source_experiment_id"], "exp_bank_source")
        self.assertEqual(seeded["bank_source_lineage"], source["lineage"])
        self.assertNotEqual(state["experiment_id"], source["source_experiment_id"])
        self.assertEqual(bank_individuals(), bank_before)


if __name__ == "__main__":
    unittest.main()
