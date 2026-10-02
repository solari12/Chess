"""Candidate self-play fitness using Phase 2H budgets and iterative search."""

from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Literal, Sequence

import chess

from app.engine.evaluation import evaluate
from app.engine.iterative_search import MAX_ITERATIVE_DEPTH, IterativeSearchResult, iterative_search
from app.learning.genetic.models import (
    BASELINE_WEIGHTS,
    CandidateGameTrace,
    GameMoveTrace,
    PieceWeights,
)
from app.learning.time_management.runtime_policy import decide_search_budget
from app.learning.time_management.runtime_shadow import load_shadow_model

EventCallback = Callable[[dict[str, object]], None]
OpponentType = Literal["baseline", "population", "elite"]


@dataclass(frozen=True)
class FitnessOpponent:
    id: str
    type: OpponentType
    weights: PieceWeights


@dataclass(frozen=True)
class GameResult:
    outcome: Literal["win", "draw", "loss"]
    candidate_color: str
    opponent_id: str
    opponent_type: OpponentType
    plies: int
    termination: str
    trace: CandidateGameTrace


@dataclass(frozen=True)
class FitnessResult:
    fitness: int
    wins: int
    draws: int
    losses: int
    games: tuple[GameResult, ...]


def evaluate_with_weights(board: chess.Board, weights: PieceWeights) -> int:
    """Score material plus light positional guidance for laboratory games."""
    score = evaluate(board, piece_value_map(weights))
    for square, piece in board.piece_map().items():
        sign = 1 if piece.color == chess.WHITE else -1
        file_index = chess.square_file(square)
        rank = chess.square_rank(square)
        relative_rank = rank if piece.color == chess.WHITE else 7 - rank
        centrality = max(0, 7 - int(abs(file_index - 3.5) + abs(rank - 3.5)))

        if piece.piece_type == chess.PAWN:
            positional = relative_rank * 5 + centrality * 2
        elif piece.piece_type == chess.KNIGHT:
            positional = centrality * 5 + (8 if relative_rank >= 2 else 0)
        elif piece.piece_type == chess.BISHOP:
            positional = centrality * 3 + (5 if relative_rank >= 2 else 0)
        elif piece.piece_type == chess.ROOK:
            positional = 12 if relative_rank == 6 else 0
        elif piece.piece_type == chess.QUEEN:
            positional = centrality
        else:
            positional = 0
        score += sign * positional
    return score


def piece_value_map(weights: PieceWeights) -> dict[int, int]:
    values = weights.as_dict()
    return {
        chess.PAWN: values["pawn"],
        chess.KNIGHT: values["knight"],
        chess.BISHOP: values["bishop"],
        chess.ROOK: values["rook"],
        chess.QUEEN: values["queen"],
        chess.KING: 20_000,
    }


def _play_game(
    candidate_weights: PieceWeights,
    candidate_color: chess.Color,
    time_control_ms: int,
    max_plies: int,
    opponent: FitnessOpponent,
    *,
    generation: int,
    candidate_id: str,
    game_index: int,
    games_total: int,
    on_event: EventCallback | None,
) -> GameResult:
    board = chess.Board()
    plies = 0
    remaining_time = {chess.WHITE: time_control_ms, chess.BLACK: time_control_ms}
    timeout_color: chess.Color | None = None
    candidate_color_name = "white" if candidate_color else "black"
    trace = CandidateGameTrace(
        generation=generation,
        candidate_id=candidate_id,
        candidate_weights=candidate_weights,
        candidate_color=candidate_color_name,
        game_index=game_index,
        games_total=games_total,
        baseline_weights=BASELINE_WEIGHTS,
        opponent_id=opponent.id,
        opponent_type=opponent.type,
        opponent_weights=opponent.weights,
    )

    _emit(
        on_event,
        type="game_started",
        generation=generation,
        candidate_id=candidate_id,
        candidate_weights=candidate_weights.as_dict(),
        candidate_color=candidate_color_name,
        opponent_id=opponent.id,
        opponent_type=opponent.type,
        opponent_weights=opponent.weights.as_dict(),
        color=candidate_color_name,
        game_index=game_index,
        games_total=games_total,
        baseline_weights=BASELINE_WEIGHTS.as_dict(),
    )

    # Keep the original automatic game-over behavior. Claimable draws are
    # recorded as their cause when the game reaches an automatic terminal state.
    while plies < max_plies and not board.is_game_over(claim_draw=False):
        mover = board.turn
        side_weights = candidate_weights if mover == candidate_color else opponent.weights

        def weighted_iterative_search(
            position: chess.Board,
            budget_ms: float,
            *,
            max_depth: int,
        ) -> IterativeSearchResult:
            return iterative_search(
                position,
                budget_ms,
                max_depth=max_depth,
                evaluator=lambda candidate_position: evaluate_with_weights(candidate_position, side_weights),
                repetition_penalty=12,
            )

        clock_before_search = remaining_time[mover]
        move_started = perf_counter()
        decision = decide_search_budget(
            board,
            remaining_time_ms=clock_before_search,
            max_depth=MAX_ITERATIVE_DEPTH,
            search_fn=weighted_iterative_search,
        )
        if decision.final_search_budget_ms > 0:
            result = weighted_iterative_search(
                board,
                decision.final_search_budget_ms,
                max_depth=MAX_ITERATIVE_DEPTH,
            )
        else:
            legal_move = next(iter(board.legal_moves), None)
            result = IterativeSearchResult(legal_move, 0, 0, 0, 0.0, False, ())

        elapsed_ms = max(
            (perf_counter() - move_started) * 1_000,
            decision.policy_latency_ms + result.time_ms,
        )
        remaining_time[mover] = max(0, int(clock_before_search - elapsed_ms))
        if elapsed_ms >= clock_before_search:
            timeout_color = mover
            break
        if result.move is None:
            break

        san = board.san(result.move)
        uci = result.move.uci()
        board.push(result.move)
        plies += 1
        move = GameMoveTrace(
            ply=plies,
            move_number=board.fullmove_number,
            color="white" if mover else "black",
            san=san,
            uci=uci,
            fen=board.fen(),
            evaluation=round(result.score / 100, 2),
            nodes=result.nodes,
            depth=result.completed_depth,
            search_budget_ms=decision.final_search_budget_ms,
            search_time_ms=result.time_ms,
            policy_time_ms=decision.policy_latency_ms,
            remaining_time_ms=remaining_time[mover],
        )
        trace.moves.append(move)
        _emit(
            on_event,
            type="move_played",
            generation=generation,
            candidate_id=candidate_id,
            game_index=game_index,
            opponent_id=opponent.id,
            opponent_type=opponent.type,
            candidate_color=candidate_color_name,
            ply=move.ply,
            move_number=move.move_number,
            color=move.color,
            move=move.san,
            san=move.san,
            uci=move.uci,
            fen=move.fen,
            evaluation=move.evaluation,
            nodes=move.nodes,
            depth=move.depth,
            search_budget_ms=move.search_budget_ms,
            search_time_ms=move.search_time_ms,
            policy_time_ms=move.policy_time_ms,
            remaining_time_ms=move.remaining_time_ms,
        )

    if timeout_color is not None:
        opponent_can_mate = not board.has_insufficient_material(not timeout_color)
        if opponent_can_mate:
            candidate_won = candidate_color != timeout_color
            game_result = "win" if candidate_won else "loss"
        else:
            game_result = "draw"
        termination = "timeout"
    elif plies >= max_plies and not board.is_game_over(claim_draw=False):
        game_result: Literal["win", "draw", "loss"] = "draw"
        termination = "max_plies"
    else:
        outcome = board.result(claim_draw=False)
        if outcome in {"1/2-1/2", "*"}:
            game_result = "draw"
        else:
            candidate_won = (outcome == "1-0") == candidate_color
            game_result = "win" if candidate_won else "loss"
        game_outcome = board.outcome(claim_draw=False)
        termination = _termination_category(game_outcome)

    fitness_delta = {"win": 1, "draw": 0, "loss": -1}[game_result]
    trace.result = game_result
    trace.plies = plies
    trace.fitness_delta = fitness_delta
    trace.termination = termination
    _emit(
        on_event,
        type="game_finished",
        generation=generation,
        candidate_id=candidate_id,
        game_index=game_index,
        opponent_id=opponent.id,
        opponent_type=opponent.type,
        candidate_color=candidate_color_name,
        color=candidate_color_name,
        result=game_result,
        plies=plies,
        termination=termination,
        fitness_delta=fitness_delta,
    )
    return GameResult(game_result, candidate_color_name, opponent.id, opponent.type, plies, termination, trace)


def _termination_category(outcome: chess.Outcome | None) -> str:
    """Name the actual python-chess ending without changing game stopping rules."""
    if outcome is None:
        return "other"
    termination = outcome.termination
    if termination == chess.Termination.CHECKMATE:
        return "checkmate"
    if termination == chess.Termination.STALEMATE:
        return "stalemate"
    if termination == chess.Termination.THREEFOLD_REPETITION:
        return "threefold_repetition"
    if termination == chess.Termination.FIVEFOLD_REPETITION:
        return "fivefold_repetition"
    if termination == chess.Termination.FIFTY_MOVES:
        return "fifty_move_rule"
    if termination == chess.Termination.SEVENTYFIVE_MOVES:
        return "seventyfive_move_rule"
    return "other"


def evaluate_fitness(
    candidate_weights: PieceWeights,
    games_per_individual: int = 2,
    time_control_ms: int = 5_000,
    max_plies: int = 200,
    baseline_weights: PieceWeights = BASELINE_WEIGHTS,
    *,
    generation: int = 1,
    candidate_id: str = "candidate",
    opponents: Sequence[FitnessOpponent] | None = None,
    on_event: EventCallback | None = None,
) -> FitnessResult:
    if games_per_individual < 2 or games_per_individual % 2:
        raise ValueError("games_per_individual must be an even number of at least 2")
    if not 1_000 <= time_control_ms <= 600_000 or max_plies < 1:
        raise ValueError("time_control_ms must be between 1000 and 600000; max_plies must be positive")

    opponent_pool = list(opponents) if opponents is not None else [
        FitnessOpponent("BASELINE", "baseline", baseline_weights)
    ]
    if not opponent_pool:
        raise ValueError("opponent pool cannot be empty")
    if len(opponent_pool) * 2 != games_per_individual:
        raise ValueError("games_per_individual must equal two games per opponent")

    # Keep the one-time cold model load outside any candidate's chess clock so
    # candidate order cannot decide who pays the startup cost.
    try:
        load_shadow_model()
    except Exception:
        # Runtime policy owns the established, clock-capped failure fallback.
        pass

    games: list[GameResult] = []
    wins = draws = losses = 0
    game_index = 0
    for opponent in opponent_pool:
        for candidate_color in (chess.WHITE, chess.BLACK):
            game_index += 1
            game = _play_game(
                candidate_weights,
                candidate_color,
                time_control_ms,
                max_plies,
                opponent,
                generation=generation,
                candidate_id=candidate_id,
                game_index=game_index,
                games_total=games_per_individual,
                on_event=on_event,
            )
            games.append(game)
            wins += game.outcome == "win"
            draws += game.outcome == "draw"
            losses += game.outcome == "loss"
            _emit(
                on_event,
                type="candidate_game_finished",
                generation=generation,
                candidate_id=candidate_id,
                game_index=game_index,
                opponent_id=opponent.id,
                opponent_type=opponent.type,
                color=game.candidate_color,
                record={"wins": wins, "draws": draws, "losses": losses, "games_played": game_index},
                fitness=wins - losses,
            )

    return FitnessResult(wins - losses, wins, draws, losses, tuple(games))


def _emit(callback: EventCallback | None, **event: object) -> None:
    if callback is not None:
        callback(event)
