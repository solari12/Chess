"""Candidate self-play fitness using the existing Alpha-Beta search."""

from dataclasses import dataclass
from typing import Callable, Literal, Sequence

import chess

from app.engine.alpha_beta import alpha_beta
from app.engine.evaluation import evaluate
from app.learning.genetic.models import (
    BASELINE_WEIGHTS,
    CandidateGameTrace,
    GameMoveTrace,
    PieceWeights,
)

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
    search_depth: int,
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

    while plies < max_plies and not board.is_game_over(claim_draw=False):
        mover = board.turn
        side_weights = candidate_weights if mover == candidate_color else opponent.weights
        result = alpha_beta(
            board,
            search_depth,
            evaluator=lambda position: evaluate_with_weights(position, side_weights),
            repetition_penalty=12,
        )
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
            depth=search_depth,
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
        )

    if plies >= max_plies and not board.is_game_over(claim_draw=False):
        game_result: Literal["win", "draw", "loss"] = "draw"
        termination = "ply_limit"
    else:
        outcome = board.result(claim_draw=False)
        if outcome in {"1/2-1/2", "*"}:
            game_result = "draw"
        else:
            candidate_won = (outcome == "1-0") == candidate_color
            game_result = "win" if candidate_won else "loss"
        game_outcome = board.outcome(claim_draw=False)
        termination = game_outcome.termination.name.lower() if game_outcome else "unknown"

    fitness_delta = {"win": 1, "draw": 0, "loss": -1}[game_result]
    trace.result = game_result
    trace.plies = plies
    trace.fitness_delta = fitness_delta
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


def evaluate_fitness(
    candidate_weights: PieceWeights,
    games_per_individual: int = 2,
    search_depth: int = 2,
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
    if search_depth < 1 or max_plies < 1:
        raise ValueError("search_depth and max_plies must be positive")

    opponent_pool = list(opponents) if opponents is not None else [
        FitnessOpponent("BASELINE", "baseline", baseline_weights)
    ]
    if not opponent_pool:
        raise ValueError("opponent pool cannot be empty")
    if len(opponent_pool) * 2 != games_per_individual:
        raise ValueError("games_per_individual must equal two games per opponent")

    games: list[GameResult] = []
    wins = draws = losses = 0
    game_index = 0
    for opponent in opponent_pool:
        for candidate_color in (chess.WHITE, chess.BLACK):
            game_index += 1
            game = _play_game(
                candidate_weights,
                candidate_color,
                search_depth,
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
