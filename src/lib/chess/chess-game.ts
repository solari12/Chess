import { Chess, type Move, type PieceSymbol, type Square } from "chess.js";
import type { CapturedPieces, GameStatus, PieceAnimation } from "@/types/chess";

export function createChessGame() {
  return new Chess();
}

export function legalMovesFrom(game: Chess, square: Square) {
  return game.moves({ square, verbose: true });
}

export function tryMove(game: Chess, from: Square, to: Square, promotion?: PieceSymbol): Move | null {
  try {
    return game.move({ from, to, ...(promotion ? { promotion } : {}) });
  } catch {
    return null;
  }
}

export function getGameStatus(game: Chess): GameStatus {
  if (game.isCheckmate()) return "checkmate";
  if (game.isStalemate()) return "stalemate";
  if (game.isDraw()) return "draw";
  if (game.isCheck()) return "check";
  return "playing";
}

export function getCapturedPieces(game: Chess): CapturedPieces {
  const captured: CapturedPieces = { w: [], b: [] };
  for (const move of game.history({ verbose: true })) {
    if (move.captured) captured[move.color].push(move.captured);
  }
  return captured;
}

export function getCheckedKingSquare(game: Chess): Square | null {
  if (!game.isCheck()) return null;
  for (const rank of game.board()) {
    for (const piece of rank) {
      if (piece?.type === "k" && piece.color === game.turn()) return piece.square;
    }
  }
  return null;
}

export function getLastMove(game: Chess): Move | null {
  const history = game.history({ verbose: true });
  return history.at(-1) ?? null;
}

export function toPieceAnimation(move: Move, id: number): PieceAnimation {
  if (!move.flags.includes("k") && !move.flags.includes("q")) return { id, move };
  const row = move.color === "w" ? "1" : "8";
  return {
    id,
    move,
    rook: move.flags.includes("k")
      ? { from: `h${row}` as Square, to: `f${row}` as Square }
      : { from: `a${row}` as Square, to: `d${row}` as Square },
  };
}

export function getPromotionChoices(game: Chess, from: Square, to: Square): PieceSymbol[] {
  return legalMovesFrom(game, from)
    .filter((move) => move.to === to && move.promotion)
    .map((move) => move.promotion as PieceSymbol);
}
