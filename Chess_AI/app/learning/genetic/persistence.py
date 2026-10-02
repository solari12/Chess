"""Versioned JSON checkpoints and a durable bank of reusable GA genomes."""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.learning.genetic.evolution import GeneticEvolution
from app.learning.genetic.models import (
    CandidateGameTrace,
    ChildEvent,
    EvolutionState,
    EvolutionStep,
    ExperimentConfig,
    GameMoveTrace,
    GeneOrigin,
    Individual,
    MutationEvent,
    ParentPair,
    PieceWeights,
)
from app.learning.genetic.reporting import summarize_game_traces

SCHEMA_VERSION = 1
DATA_ROOT = Path(__file__).resolve().parents[3] / "data" / "genetic_lab"
_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,100}$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def new_experiment_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"exp_{timestamp}_{uuid4().hex[:8]}"


def checkpoint_for_experiment(experiment: GeneticEvolution) -> dict[str, Any]:
    state = asdict(experiment.state)
    for trace in state["game_traces"]:
        trace["moves"] = []
    config = asdict(experiment.config)
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": {
            "experiment_id": experiment.experiment_id,
            "created_at": experiment.created_at,
            "updated_at": experiment.updated_at,
            "random_seed": experiment.config.seed,
            "current_generation": experiment.state.generation,
            **config,
            "fitness_definition": "wins - losses",
            "termination_settings": {"max_plies": experiment.config.max_plies},
            "best_fitness": max((item.fitness for item in experiment.state.population if item.fitness is not None), default=None),
            "tied_best_individual_ids": _tied_best_ids(experiment.state.population),
            "population_status": [
                {"individual_id": item.id, "status": "evaluated" if item.fitness is not None else "pending"}
                for item in experiment.state.population
            ],
        },
        "state": state,
        "initial_candidates": [asdict(candidate) for candidate in experiment.initial_candidates],
        "game_report": summarize_game_traces(experiment.state.game_traces),
        "next_individual_id": experiment.next_individual_id,
        "rng_state": _json_tuple(experiment.rng.getstate()),
        "opponent_rng_state": _json_tuple(experiment.opponent_rng.getstate()),
    }


def restore_experiment(checkpoint: dict[str, Any]) -> GeneticEvolution:
    _check_schema(checkpoint)
    metadata = checkpoint.get("experiment")
    state_payload = checkpoint.get("state")
    if not isinstance(metadata, dict) or not isinstance(state_payload, dict):
        raise ValueError("Checkpoint must include experiment metadata and state")
    config_payload = state_payload.get("config")
    if not isinstance(config_payload, dict):
        config_payload = {key: metadata[key] for key in ExperimentConfig.__dataclass_fields__ if key in metadata}
    config = ExperimentConfig(**config_payload)
    initial_candidates = [_individual(item) for item in checkpoint.get("initial_candidates", [])]
    experiment = GeneticEvolution(
        config,
        experiment_id=str(metadata["experiment_id"]),
        created_at=str(metadata["created_at"]),
        initial_candidates=initial_candidates,
    )
    experiment.updated_at = str(metadata.get("updated_at", experiment.created_at))
    population = [_individual(item) for item in state_payload.get("population", [])]
    history = [_evolution_step(item) for item in state_payload.get("history", [])]
    traces = [_game_trace(item) for item in state_payload.get("game_traces", [])]
    generation = int(state_payload.get("generation", metadata.get("current_generation", 0)))
    if generation != int(metadata.get("current_generation", generation)):
        raise ValueError("Checkpoint generation metadata does not match its state")
    if [step.generation for step in history] != list(range(1, generation + 1)):
        raise ValueError("Checkpoint history must contain each completed generation exactly once")
    status = "complete" if generation >= config.generations else "ready"
    if generation == 0 and state_payload.get("status") == "complete":
        raise ValueError("An empty checkpoint cannot be complete")
    experiment.state = EvolutionState(
        status=status,
        config=config,
        generation=generation,
        population=population,
        history=history,
        game_traces=traces,
    )
    experiment.next_individual_id = int(checkpoint.get("next_individual_id", _next_id_after(population, history)))
    experiment.rng.setstate(_to_tuple(checkpoint["rng_state"]))
    experiment.opponent_rng.setstate(_to_tuple(checkpoint["opponent_rng_state"]))
    return experiment


def save_checkpoint(checkpoint: dict[str, Any]) -> dict[str, Any]:
    _check_schema(checkpoint)
    experiment_id = _safe_id(str(checkpoint["experiment"]["experiment_id"]))
    checkpoint = json.loads(json.dumps(checkpoint))
    checkpoint["experiment"]["updated_at"] = utc_now()
    directory = DATA_ROOT / "experiments" / experiment_id
    _atomic_json(directory / "checkpoint.json", checkpoint)
    return experiment_summary(checkpoint)


def load_checkpoint(experiment_id: str) -> dict[str, Any]:
    path = DATA_ROOT / "experiments" / _safe_id(experiment_id) / "checkpoint.json"
    if not path.is_file():
        raise FileNotFoundError(f"Saved experiment '{experiment_id}' was not found")
    payload = _read_json(path)
    _check_schema(payload)
    return payload


def list_experiments() -> list[dict[str, Any]]:
    root = DATA_ROOT / "experiments"
    if not root.exists():
        return []
    summaries = []
    for path in root.glob("*/checkpoint.json"):
        try:
            payload = _read_json(path)
            _check_schema(payload)
            summaries.append(experiment_summary(payload))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return sorted(summaries, key=lambda item: item["updated_at"], reverse=True)


def experiment_summary(checkpoint: dict[str, Any]) -> dict[str, Any]:
    metadata = checkpoint["experiment"]
    state = checkpoint["state"]
    population = state.get("population", [])
    fitness_values = [item["fitness"] for item in population if item.get("fitness") is not None]
    return {
        "experiment_id": metadata["experiment_id"],
        "created_at": metadata["created_at"],
        "updated_at": metadata.get("updated_at", metadata["created_at"]),
        "current_generation": int(metadata.get("current_generation", 0)),
        "generation_limit": int(metadata.get("generations", 0)),
        "population_size": int(metadata.get("population_size", len(population))),
        "best_fitness": max(fitness_values) if fitness_values else None,
    }


def bank_individuals() -> list[dict[str, Any]]:
    path = DATA_ROOT / "individual_bank" / "individuals.json"
    if not path.is_file():
        return []
    payload = _read_json(path)
    _check_schema(payload)
    items = payload.get("individuals")
    if not isinstance(items, list):
        raise ValueError("Individual bank file has an invalid individuals list")
    return sorted(items, key=lambda item: item["created_at"], reverse=True)


def save_bank_individuals(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    existing = bank_individuals()
    by_source = {(item["source_experiment_id"], item["individual_id"]): item for item in existing}
    saved = []
    for record in records:
        key = (record["source_experiment_id"], record["individual_id"])
        if key in by_source:
            saved.append(by_source[key])
            continue
        item = {"bank_id": f"ib_{uuid4().hex}", "created_at": utc_now(), "tags": [], "notes": "", **record}
        existing.append(item)
        by_source[key] = item
        saved.append(item)
    _atomic_json(DATA_ROOT / "individual_bank" / "individuals.json", {
        "schema_version": SCHEMA_VERSION,
        "individuals": existing,
    })
    return saved


def save_candidate_set(
    *,
    name: str,
    source_experiment_id: str,
    source_generation: int,
    bank_ids: list[str],
) -> dict[str, Any]:
    if not name.strip():
        raise ValueError("Candidate set name cannot be empty")
    if not bank_ids or len(bank_ids) != len(set(bank_ids)):
        raise ValueError("Candidate sets require unique bank individuals")
    available = {item["bank_id"] for item in bank_individuals()}
    if any(bank_id not in available for bank_id in bank_ids):
        raise FileNotFoundError("A candidate set entry is missing from the Individual Bank")
    candidate_set = {
        "schema_version": SCHEMA_VERSION,
        "candidate_set_id": f"set_{uuid4().hex}",
        "name": name.strip(),
        "source_experiment_id": source_experiment_id,
        "source_generation": source_generation,
        "bank_ids": list(bank_ids),
        "created_at": utc_now(),
    }
    _atomic_json(DATA_ROOT / "candidate_sets" / f"{candidate_set['candidate_set_id']}.json", candidate_set)
    return candidate_set


def list_candidate_sets() -> list[dict[str, Any]]:
    root = DATA_ROOT / "candidate_sets"
    if not root.exists():
        return []
    results = []
    for path in root.glob("set_*.json"):
        try:
            payload = _read_json(path)
            _check_schema(payload)
            results.append(payload)
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return sorted(results, key=lambda item: item["created_at"], reverse=True)


def individuals_from_bank(bank_ids: list[str]) -> list[Individual]:
    if not bank_ids:
        raise ValueError("Select at least one Individual Bank entry")
    if len(bank_ids) != len(set(bank_ids)):
        raise ValueError("The selected Individual Bank entries contain duplicates")
    requested = set(bank_ids)
    records = [item for item in bank_individuals() if item["bank_id"] in requested]
    if len(records) != len(requested):
        raise FileNotFoundError("One or more selected Individual Bank entries were not found")
    stable_ids = [item["individual_id"] for item in records]
    if len(stable_ids) != len(set(stable_ids)):
        raise ValueError("Selected bank entries reuse an individual ID from different experiments; select only one of those copies")
    return [
        Individual(
            id=item["individual_id"],
            weights=PieceWeights(**item["chromosome"]),
            fitness=None,
            generation=1,
            bank_source_experiment_id=item["source_experiment_id"],
            bank_source_generation=int(item["source_generation"]),
            bank_source_individual_id=item["individual_id"],
            bank_source_lineage=item.get("lineage"),
        )
        for item in records
    ]


def _individual(payload: dict[str, Any]) -> Individual:
    values = dict(payload)
    values["weights"] = PieceWeights(**values["weights"])
    return Individual(**values)


def _evolution_step(payload: dict[str, Any]) -> EvolutionStep:
    return EvolutionStep(
        generation=int(payload["generation"]),
        population=[_individual(item) for item in payload["population"]],
        selected_parents=list(payload.get("selected_parents", [])),
        parent_pairs=[ParentPair(**item) for item in payload.get("parent_pairs", [])],
        children=[_child_event(item) for item in payload.get("children", [])],
        elite_individuals=[_individual(item) for item in payload.get("elite_individuals", [])],
        best_individual=_individual(payload["best_individual"]),
        average_fitness=float(payload["average_fitness"]),
    )


def _child_event(payload: dict[str, Any]) -> ChildEvent:
    return ChildEvent(
        child_id=payload.get("child_id", payload["individual"]["id"]),
        generation=int(payload.get("generation", payload["individual"]["generation"])),
        individual=_individual(payload["individual"]),
        parent_a_id=payload["parent_a_id"],
        parent_b_id=payload["parent_b_id"],
        crossover_weights=PieceWeights(**payload["crossover_weights"]),
        pre_mutation_weights=PieceWeights(**payload.get("pre_mutation_weights", payload["crossover_weights"])),
        post_mutation_weights=PieceWeights(**payload.get("post_mutation_weights", payload["individual"]["weights"])),
        gene_origins=[GeneOrigin(**item) for item in payload["gene_origins"]],
        mutations=[MutationEvent(**item) for item in payload.get("mutations", [])],
    )


def _game_trace(payload: dict[str, Any]) -> CandidateGameTrace:
    values = dict(payload)
    values["candidate_weights"] = PieceWeights(**values["candidate_weights"])
    values["baseline_weights"] = PieceWeights(**values["baseline_weights"])
    if values.get("opponent_weights") is not None:
        values["opponent_weights"] = PieceWeights(**values["opponent_weights"])
    values["moves"] = [GameMoveTrace(**move) for move in values.get("moves", [])]
    return CandidateGameTrace(**values)


def _next_id_after(population: list[Individual], history: list[EvolutionStep]) -> int:
    ids = [individual.id for individual in population]
    for step in history:
        ids.extend(individual.id for individual in step.population)
    numbers = [int(match.group(1)) for individual_id in ids if (match := re.fullmatch(r"I(\d+)", individual_id))]
    return max(numbers, default=0) + 1


def _tied_best_ids(population: list[Individual]) -> list[str]:
    evaluated = [item for item in population if item.fitness is not None]
    if not evaluated:
        return []
    best = max(item.fitness for item in evaluated if item.fitness is not None)
    return [item.id for item in evaluated if item.fitness == best]


def _check_schema(payload: dict[str, Any]) -> None:
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        found = payload.get("schema_version") if isinstance(payload, dict) else None
        raise ValueError(f"Unsupported Genetic Lab JSON schema version: {found}")


def _safe_id(value: str) -> str:
    if not _SAFE_ID.fullmatch(value):
        raise ValueError("Invalid saved experiment ID")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path.name}")
    return payload


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = ""
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp") as stream:
            temporary_name = stream.name
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.replace(temporary_name, path)
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _json_tuple(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_tuple(item) for item in value]
    if isinstance(value, list):
        return [_json_tuple(item) for item in value]
    return value


def _to_tuple(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_to_tuple(item) for item in value)
    return value
