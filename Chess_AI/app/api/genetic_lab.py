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
from app.learning.genetic.reporting import summarize_game_traces

router = APIRouter(prefix="/api/lab/genetic", tags=["genetic-laboratory"])
_experiment: GeneticEvolution | None = None
_experiment_lock = Lock()
_stream_active = False
_active_stream_snapshot: dict[str, Any] | None = None
_active_lineage_payload: dict[str, Any] | None = None
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
            "status": "ready",
            "generation": 0,
            "generation_limit": 0,
            "config": None,
            "population": [],
            "history": [],
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
        global _stream_active, _active_stream_snapshot, _active_lineage_payload
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
    global _stream_active, _active_stream_snapshot, _active_lineage_payload
    if _stream_active:
        raise HTTPException(status_code=409, detail="An evaluation stream is already active")
    _active_stream_snapshot = _state_payload()
    _active_lineage_payload = None
    _cancel_requested.clear()
    _stream_finished.clear()
    _stream_active = True
