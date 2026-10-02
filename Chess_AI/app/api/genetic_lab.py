"""HTTP endpoints for the separate Genetic Algorithm Laboratory."""

import json
from copy import deepcopy
from dataclasses import asdict
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from typing import Any, Iterator

import chess
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator
from starlette.responses import StreamingResponse

from app.engine.alpha_beta import alpha_beta
from app.learning.genetic.fitness import evaluate_with_weights
from app.learning.genetic.evolution import GeneticEvolution
from app.learning.genetic.models import ExperimentConfig, PieceWeights
from app.learning.genetic.models import Individual
from app.learning.genetic.persistence import (
    bank_individuals,
    checkpoint_for_experiment,
    individuals_from_bank,
    list_candidate_sets,
    list_experiments,
    load_checkpoint,
    restore_experiment,
    save_bank_individuals,
    save_candidate_set,
    save_checkpoint,
)
from app.learning.genetic.reporting import summarize_game_traces

router = APIRouter(prefix="/api/lab/genetic", tags=["genetic-laboratory"])
_experiment: GeneticEvolution | None = None
_experiment_lock = Lock()
_stream_active = False
_active_stream_snapshot: dict[str, Any] | None = None
_active_lineage_payload: dict[str, Any] | None = None
_active_experiment_checkpoint: dict[str, Any] | None = None
_cancel_requested = Event()
_stream_finished = Event()
_stream_finished.set()


class EvaluationCancelled(Exception):
    """Raised at an evaluation callback boundary when reset is requested."""


class StartExperimentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    population_size: int = Field(default=10, ge=2, le=20)
    games_per_individual: int = Field(default=8, ge=4, le=8)
    time_control_ms: int = Field(default=5_000, ge=1_000, le=600_000)
    mutation_rate: float = Field(default=0.15, ge=0, le=1)
    mutation_strength: int = Field(default=25, ge=1, le=100)
    elite_count: int = Field(default=2, ge=0)
    tournament_size: int = Field(default=3, ge=1)
    max_plies: int = Field(default=200, ge=1, le=200)
    generations: int = Field(default=20, ge=1, le=100)
    seed: int | None = None

    @model_validator(mode="after")
    def validate_population_options(self) -> "StartExperimentRequest":
        if self.games_per_individual % 2:
            raise ValueError("games_per_individual must be even and at least 4 to compare multiple opponents with both colors")
        if self.elite_count >= self.population_size:
            raise ValueError("elite_count must be less than population_size")
        if self.population_size < self.games_per_individual // 2:
            raise ValueError("population_size must provide enough unique opponents for games_per_individual")
        if self.tournament_size > self.population_size:
            raise ValueError("tournament_size cannot exceed population_size")
        return self

    def to_config(self) -> ExperimentConfig:
        return ExperimentConfig(**self.model_dump())


class ExperimentReference(BaseModel):
    experiment_id: str


class BankSaveRequest(BaseModel):
    experiment_id: str
    generation: int = Field(ge=1)
    individual_ids: list[str] = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)
    notes: str = Field(default="", max_length=2_000)


class CandidateSetSaveRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)


class StartFromBankRequest(BaseModel):
    config: StartExperimentRequest
    bank_ids: list[str] = Field(min_length=1)


class GeneticWeightsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pawn: int = Field(ge=50, le=200)
    knight: int = Field(ge=150, le=500)
    bishop: int = Field(ge=150, le=500)
    rook: int = Field(ge=300, le=800)
    queen: int = Field(ge=700, le=1200)

    def to_piece_weights(self) -> PieceWeights:
        return PieceWeights(**self.model_dump())


class GeneticMoveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fen: str = Field(min_length=1, max_length=256)
    depth: int = Field(default=3, ge=1, le=4)
    weights: GeneticWeightsRequest


@router.post("/play-move")
def play_genetic_move(request: GeneticMoveRequest) -> dict[str, object]:
    try:
        board = chess.Board(request.fen)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=f"Invalid FEN: {error}") from error

    weights = request.weights.to_piece_weights()
    result = alpha_beta(
        board,
        request.depth,
        evaluator=lambda position: evaluate_with_weights(position, weights),
    )
    if result.move is None:
        raise HTTPException(status_code=409, detail="The position has no legal move")
    return {
        "move": result.move.uci(),
        "algorithm": "genetic",
        "depth": request.depth,
        "score": result.score,
        "nodes": result.nodes,
        "time_ms": result.time_ms,
    }


def _state_payload(*, allow_active_state: bool = False) -> dict[str, Any]:
    if _experiment is None:
        return {
            "experiment_id": None,
            "created_at": None,
            "updated_at": None,
            "status": "ready",
            "generation": 0,
            "generation_limit": 0,
            "config": None,
            "population": [],
            "history": [],
            "active_lineage": None,
            "game_traces": [],
            "game_report": summarize_game_traces([]),
        }
    if _stream_active and not allow_active_state and _active_stream_snapshot is not None:
        payload = deepcopy(_active_stream_snapshot)
        payload["status"] = "running"
        payload["active_lineage"] = deepcopy(_active_lineage_payload)
        payload["game_report"] = summarize_game_traces(
            _experiment.state.game_traces if _experiment is not None else []
        )
        return payload
    payload = asdict(_experiment.state)
    payload["experiment_id"] = _experiment.experiment_id
    payload["created_at"] = _experiment.created_at
    payload["updated_at"] = _experiment.updated_at
    payload["active_lineage"] = None
    payload["generation_limit"] = _experiment.config.generations
    payload["game_report"] = summarize_game_traces(_experiment.state.game_traces)
    return payload


@router.post("/start")
def start_experiment(request: StartExperimentRequest) -> dict[str, Any]:
    global _experiment
    try:
        experiment = GeneticEvolution(request.to_config())
        with _experiment_lock:
            if _stream_active:
                raise HTTPException(status_code=409, detail="An evaluation stream is already active")
            experiment.start()
            _experiment = experiment
            return _state_payload()
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.post("/start/stream")
def start_experiment_stream(request: StartExperimentRequest) -> StreamingResponse:
    global _experiment
    try:
        experiment = GeneticEvolution(request.to_config())
        with _experiment_lock:
            if _stream_active:
                raise HTTPException(status_code=409, detail="An evaluation stream is already active")
            _experiment = experiment
            _claim_stream()
        return _stream_response(experiment, "start", reserved=True)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.get("/experiments")
def saved_experiments() -> list[dict[str, Any]]:
    try:
        return list_experiments()
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.post("/experiments/save")
def save_current_experiment() -> dict[str, Any]:
    with _experiment_lock:
        if _experiment is None:
            raise HTTPException(status_code=409, detail="There is no experiment to save")
        checkpoint = _active_experiment_checkpoint if _stream_active else checkpoint_for_experiment(_experiment)
        if checkpoint is None:
            raise HTTPException(status_code=409, detail="The last completed generation is not available to save")
    try:
        return save_checkpoint(checkpoint)
    except (ValueError, OSError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.post("/experiments/load")
def load_saved_experiment(request: ExperimentReference) -> dict[str, Any]:
    global _experiment
    with _experiment_lock:
        if _stream_active:
            raise HTTPException(status_code=409, detail="Wait for the active generation before loading an experiment")
        try:
            checkpoint = load_checkpoint(request.experiment_id)
            restored = restore_experiment(checkpoint)
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except (ValueError, KeyError, TypeError, OSError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        if _experiment is not None and _experiment.state.generation > 0 and _experiment.experiment_id != restored.experiment_id:
            try:
                save_checkpoint(checkpoint_for_experiment(_experiment))
            except (ValueError, OSError) as error:
                raise HTTPException(status_code=500, detail=f"Could not preserve the current experiment: {error}") from error
        _experiment = restored
        return _state_payload()


@router.get("/bank")
def get_individual_bank() -> list[dict[str, Any]]:
    try:
        return bank_individuals()
    except (ValueError, OSError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.get("/candidate-sets")
def get_candidate_sets() -> list[dict[str, Any]]:
    try:
        return list_candidate_sets()
    except (ValueError, OSError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.post("/candidate-sets/save-best")
def save_tied_best_candidate_set(request: CandidateSetSaveRequest) -> dict[str, Any]:
    with _experiment_lock:
        if _experiment is None or _experiment.state.generation == 0:
            raise HTTPException(status_code=409, detail="Complete the first generation before saving candidates")
        evaluated = [item for item in _experiment.state.population if item.fitness is not None]
        if not evaluated:
            raise HTTPException(status_code=409, detail="The current population has no evaluated candidates")
        best_fitness = max(item.fitness for item in evaluated if item.fitness is not None)
        best = [item for item in evaluated if item.fitness == best_fitness]
        source_experiment_id = _experiment.experiment_id
        source_generation = _experiment.state.generation
        records = [_bank_record(_experiment, item, [], "") for item in best]
    try:
        saved = save_bank_individuals(records)
        candidate_set = save_candidate_set(
            name=request.name,
            source_experiment_id=source_experiment_id,
            source_generation=source_generation,
            bank_ids=[item["bank_id"] for item in saved],
        )
        return {"candidate_set": candidate_set, "individuals": saved}
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except (ValueError, OSError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.post("/bank/save")
def save_individuals_to_bank(request: BankSaveRequest) -> list[dict[str, Any]]:
    with _experiment_lock:
        if _experiment is None or _experiment.state.generation == 0:
            raise HTTPException(status_code=409, detail="Complete the first generation before saving individuals")
        if request.experiment_id != _experiment.experiment_id:
            raise HTTPException(status_code=409, detail="The selected individuals belong to a different experiment; refresh the population and try again")
        generation_snapshot = next(
            (step for step in _experiment.state.history if step.generation == request.generation),
            None,
        )
        if generation_snapshot is None:
            raise HTTPException(status_code=404, detail="The selected generation is not available in this experiment")
        population_by_id = {individual.id: individual for individual in generation_snapshot.population}
        if len(request.individual_ids) != len(set(request.individual_ids)):
            raise HTTPException(status_code=422, detail="Selected individual IDs contain duplicates")
        selected = [population_by_id.get(individual_id) for individual_id in request.individual_ids]
        if any(individual is None for individual in selected):
            raise HTTPException(status_code=404, detail="One or more selected individuals are not in the selected generation")
        records = [
            _bank_record(_experiment, individual, request.tags, request.notes)
            for individual in selected
            if individual is not None
        ]
    try:
        return save_bank_individuals(records)
    except (ValueError, OSError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.post("/start-from-bank/stream")
def start_from_bank_stream(request: StartFromBankRequest) -> StreamingResponse:
    global _experiment
    try:
        candidates = individuals_from_bank(request.bank_ids)
        config = request.config.to_config()
        if len(candidates) > config.population_size:
            raise HTTPException(status_code=422, detail="Selected bank individuals exceed the new population size")
        experiment = GeneticEvolution(config, initial_candidates=candidates)
        with _experiment_lock:
            if _stream_active:
                raise HTTPException(status_code=409, detail="An evaluation stream is already active")
            if _experiment is not None and _experiment.state.generation > 0:
                # Preserve the outgoing in-memory experiment before switching
                # the active slot to a new experiment seeded from the bank.
                save_checkpoint(checkpoint_for_experiment(_experiment))
            _experiment = experiment
            _claim_stream()
        return _stream_response(experiment, "start", reserved=True)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except OSError as error:
        raise HTTPException(status_code=500, detail=f"Could not preserve the current experiment: {error}") from error


@router.post("/step")
def step_experiment() -> dict[str, Any]:
    with _experiment_lock:
        if _stream_active:
            raise HTTPException(status_code=409, detail="An evaluation stream is already active")
        if _experiment is None:
            raise HTTPException(status_code=409, detail="Start an experiment before stepping")
        try:
            _experiment.step_generation()
            return _state_payload()
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error


@router.post("/step/stream")
def step_experiment_stream() -> StreamingResponse:
    with _experiment_lock:
        if _experiment is None:
            raise HTTPException(status_code=409, detail="Start an experiment before stepping")
        if _stream_active:
            raise HTTPException(status_code=409, detail="An evaluation stream is already active")
        if _experiment.state.status == "complete":
            raise HTTPException(status_code=409, detail="Experiment is complete")
        experiment = _experiment
        _claim_stream()
    return _stream_response(experiment, "step_generation", reserved=True)


@router.post("/reset")
def reset_experiment() -> dict[str, Any]:
    global _experiment
    with _experiment_lock:
        if _stream_active:
            _cancel_requested.set()
            stream_finished = _stream_finished
        else:
            _experiment = None
            return _state_payload()

    # Let the worker leave its current search/move callback before replacing its state.
    stream_finished.wait()
    with _experiment_lock:
        _experiment = None
        return _state_payload()


@router.get("/state")
def get_experiment_state() -> dict[str, Any]:
    with _experiment_lock:
        return _state_payload()


def _stream_response(
    experiment: GeneticEvolution,
    action: str,
    *,
    reserved: bool = False,
) -> StreamingResponse:
    if not reserved:
        with _experiment_lock:
            _claim_stream()

    events: Queue[dict[str, object] | None] = Queue(maxsize=64)
    disconnected = Event()

    def publish(event: dict[str, object]) -> None:
        global _active_lineage_payload
        if _cancel_requested.is_set():
            raise EvaluationCancelled()
        if event.get("type") == "generation_lineage" and isinstance(event.get("lineage"), dict):
            with _experiment_lock:
                _active_lineage_payload = deepcopy(event["lineage"])
        while not disconnected.is_set():
            try:
                events.put(event, timeout=0.1)
                return
            except Full:
                continue

    def run_evaluation() -> None:
        global _stream_active, _active_stream_snapshot, _active_lineage_payload, _active_experiment_checkpoint
        try:
            with _experiment_lock:
                if _experiment is not experiment:
                    publish({"type": "stream_error", "detail": "Experiment was replaced"})
                    return
            getattr(experiment, action)(on_event=publish)
            with _experiment_lock:
                state = _state_payload(allow_active_state=True)
            publish({"type": "state", "state": state})
        except EvaluationCancelled:
            with _experiment_lock:
                state = _state_payload(allow_active_state=True)
            state["status"] = "ready"
            try:
                events.put({"type": "state", "state": state}, timeout=0.1)
            except Full:
                pass
        except Exception as error:  # Send worker errors through the same stream.
            publish({"type": "stream_error", "detail": str(error)})
        finally:
            with _experiment_lock:
                _stream_active = False
                _active_stream_snapshot = None
                _active_lineage_payload = None
                _active_experiment_checkpoint = None
                _stream_finished.set()
            while not disconnected.is_set():
                try:
                    events.put(None, timeout=0.1)
                    break
                except Full:
                    continue

    Thread(target=run_evaluation, name="genetic-lab-evaluation", daemon=True).start()

    def event_body() -> Iterator[str]:
        try:
            while True:
                try:
                    event = events.get(timeout=15)
                except Empty:
                    yield ": keep-alive\n\n"
                    continue
                if event is None:
                    break
                yield f"data: {json.dumps(event, separators=(',', ':'))}\n\n"
        finally:
            disconnected.set()

    return StreamingResponse(
        event_body(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _claim_stream() -> None:
    global _stream_active, _active_stream_snapshot, _active_lineage_payload, _active_experiment_checkpoint
    if _stream_active:
        raise HTTPException(status_code=409, detail="An evaluation stream is already active")
    _active_stream_snapshot = _state_payload()
    _active_lineage_payload = None
    _active_experiment_checkpoint = checkpoint_for_experiment(_experiment) if _experiment is not None else None
    _cancel_requested.clear()
    _stream_finished.clear()
    _stream_active = True


def _bank_record(
    experiment: GeneticEvolution,
    individual: Individual,
    tags: list[str],
    notes: str,
) -> dict[str, Any]:
    lineage: dict[str, Any] | None = None
    for step in reversed(experiment.state.history):
        child = next((event for event in step.children if event.child_id == individual.id), None)
        if child is not None:
            lineage = {"kind": "offspring", **asdict(child)}
            break
        elite = next((item for item in step.elite_individuals if item.id == individual.id), None)
        if elite is not None:
            lineage = {"kind": "elite_clone", "individual": asdict(elite), "parent_id": elite.elite_from}
            break
    if lineage is None and individual.bank_source_experiment_id:
        lineage = {
            "kind": "bank_seed",
            "source_experiment_id": individual.bank_source_experiment_id,
            "source_generation": individual.bank_source_generation,
            "source_individual_id": individual.bank_source_individual_id or individual.id,
            "source_lineage": individual.bank_source_lineage,
        }
    return {
        "individual_id": individual.id,
        "source_experiment_id": experiment.experiment_id,
        "source_generation": individual.generation,
        "chromosome": individual.weights.as_dict(),
        "fitness": individual.fitness,
        "wins": individual.wins,
        "draws": individual.draws,
        "losses": individual.losses,
        "games_played": individual.games_played,
        "lineage": lineage,
        "tags": list(tags),
        "notes": notes,
    }
