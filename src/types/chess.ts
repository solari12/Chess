import type { Color, Move, PieceSymbol, Square } from "chess.js";

export type GameStatus = "playing" | "check" | "checkmate" | "stalemate" | "draw";

export type Side = Color;

export type PlayerRole = "human" | "minimax" | "alpha_beta";

export type PlayerRoles = Record<Side, PlayerRole>;

export type ChessViewMode =
  | "3d"
  | "transitioning-to-2d"
  | "2d"
  | "transitioning-to-3d";

export interface CapturedPieces {
  w: PieceSymbol[];
  b: PieceSymbol[];
}

export interface CastleRookTransition {
  from: Square;
  to: Square;
}

export interface PieceAnimation {
  id: number;
  move: Move;
  rook?: CastleRookTransition;
}

export interface PromotionRequest {
  from: Square;
  to: Square;
  color: Color;
}
