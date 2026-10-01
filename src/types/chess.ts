import type { Color, Move, PieceSymbol, Square } from "chess.js";

export type GameStatus = "playing" | "check" | "checkmate" | "stalemate" | "draw" | "timeout";
export type GameTermination = "checkmate" | "stalemate" | "draw" | "timeout" | null;

export type Side = Color;

export type PlayerRole = "human" | "minimax" | "alpha_beta" | "genetic";

export interface GeneticPlayerProfile {
  candidate_id: string;
  fitness: number;
  weights: {
    pawn: number;
    knight: number;
    bishop: number;
    rook: number;
    queen: number;
  };
}

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
