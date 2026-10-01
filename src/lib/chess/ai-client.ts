import type { GeneticPlayerProfile, PlayerRole } from "@/types/chess";
import type { PieceSymbol, Square } from "chess.js";

export interface ParsedUciMove {
  from: Square;
  to: Square;
  promotion?: PieceSymbol;
}

export function parseUciMove(uci: string): ParsedUciMove | null {
  const match = /^([a-h][1-8])([a-h][1-8])([qrbn])?$/.exec(uci);
  if (!match) return null;
  const [, from, to, promotion] = match;
  return {
    from: from as Square,
    to: to as Square,
    ...(promotion ? { promotion: promotion as PieceSymbol } : {}),
  };
}

export interface AIResponse {
  move: string;
  algorithm: "minimax" | "alpha-beta" | "genetic";
  depth: number;
  score: number;
  nodes: number;
  time_ms: number;
}

export async function requestAIMove(
  fen: string,
  role: PlayerRole,
  depth: number,
  signal?: AbortSignal,
  geneticProfile?: GeneticPlayerProfile | null,
): Promise<AIResponse | null> {
  if (role === "human") return null;

  const baseUrl = (process.env.NEXT_PUBLIC_CHESS_AI_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");
  const endpoint = role === "genetic" ? "/api/lab/genetic/play-move" : "/api/ai/move";
  let body: object;
  if (role === "genetic") {
    if (!geneticProfile) throw new Error("Choose a trained genetic candidate before starting this game.");
    body = { fen, depth, weights: geneticProfile.weights };
  } else {
    body = { fen, algorithm: role, depth };
  }
  const response = await fetch(`${baseUrl}${endpoint}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`Chess AI request failed (${response.status}): ${detail}`);
  }

  const result: unknown = await response.json();
  if (
    typeof result !== "object" ||
    result === null ||
    !("move" in result) ||
    typeof result.move !== "string" ||
    !("algorithm" in result) ||
    typeof result.algorithm !== "string" ||
    !("time_ms" in result) ||
    typeof result.time_ms !== "number" ||
    !Number.isFinite(result.time_ms)
  ) {
    throw new Error("Chess AI returned an invalid response");
  }

  const responseAlgorithm = result.algorithm === "alpha_beta" ? "alpha-beta" : result.algorithm;
  const expectedAlgorithm = role === "alpha_beta" ? "alpha-beta" : role;
  if (responseAlgorithm !== expectedAlgorithm) {
    throw new Error("Chess AI returned an invalid response");
  }

  return { ...result, algorithm: responseAlgorithm } as AIResponse;
}
