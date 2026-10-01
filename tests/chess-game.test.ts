import { describe, expect, it } from "vitest";
import { Chess, type Square } from "chess.js";
import { getCapturedPieces, getCheckedKingSquare, getGameStatus, legalMovesFrom, toPieceAnimation, tryMove } from "@/lib/chess/chess-game";

const sq = (square: string) => square as Square;
const move = (game: Chess, from: string, to: string, promotion?: "q" | "r" | "b" | "n") => {
  const result = tryMove(game, sq(from), sq(to), promotion);
  expect(result, `${from}-${to} should be a legal move`).not.toBeNull();
  return result;
};

describe("chess.js adapter", () => {
  it("starts with the standard position and generates pawn and knight moves", () => {
    const game = new Chess();
    expect(game.board().flat().filter(Boolean)).toHaveLength(32);
    expect(legalMovesFrom(game, sq("e2")).map(({ to }) => to)).toEqual(["e3", "e4"]);
    expect(legalMovesFrom(game, sq("g1")).map(({ to }) => to)).toEqual(["f3", "h3"]);
    expect(tryMove(game, sq("e2"), sq("e5"))).toBeNull();
  });

  it("handles bishop, rook, queen, and king movement without jumping", () => {
    const bishop = new Chess("7k/8/8/8/8/8/8/2B1K3 w - - 0 1");
    expect(legalMovesFrom(bishop, sq("c1")).some(({ to }) => to === "f4")).toBe(true);
    const rook = new Chess("7k/8/8/8/8/8/8/R3K3 w - - 0 1");
    expect(legalMovesFrom(rook, sq("a1")).some(({ to }) => to === "a4")).toBe(true);
    const queen = new Chess("7k/8/8/8/8/8/8/3QK3 w - - 0 1");
    expect(legalMovesFrom(queen, sq("d1")).some(({ to }) => to === "d4")).toBe(true);
    expect(legalMovesFrom(queen, sq("d1")).some(({ to }) => to === "g4")).toBe(true);
    const king = new Chess("7k/8/8/8/8/8/8/4K3 w - - 0 1");
    expect(legalMovesFrom(king, sq("e1")).some(({ to }) => to === "f2")).toBe(true);
    const checkedSquare = new Chess("3r3k/8/8/8/8/8/8/4K3 w - - 0 1");
    expect(legalMovesFrom(checkedSquare, sq("e1")).some(({ to }) => to === "d1")).toBe(false);
  });

  it("rejects pinned moves, records captures, and switches turns", () => {
    const game = new Chess();
    move(game, "e2", "e4");
    expect(game.turn()).toBe("b");
    move(game, "d7", "d5");
    move(game, "e4", "d5");
    expect(getCapturedPieces(game).w).toEqual(["p"]);
    expect(getCapturedPieces(game).b).toEqual([]);
    const pinned = new Chess("4r2k/8/8/8/8/8/4R3/4K3 w - - 0 1");
    expect(legalMovesFrom(pinned, sq("e2")).some(({ to }) => to === "d2")).toBe(false);
    expect(legalMovesFrom(pinned, sq("e2")).some(({ to }) => to === "e3")).toBe(true);
  });

  it("identifies check, checkmate, stalemate, and other draw states", () => {
    const check = new Chess("4r2k/8/8/8/8/8/8/4K3 w - - 0 1");
    expect(getGameStatus(check)).toBe("check");
    expect(getCheckedKingSquare(check)).toBe("e1");
    const foolsMate = new Chess();
    move(foolsMate, "f2", "f3"); move(foolsMate, "e7", "e5");
    move(foolsMate, "g2", "g4"); move(foolsMate, "d8", "h4");
    expect(getGameStatus(foolsMate)).toBe("checkmate");
    expect(foolsMate.isGameOver()).toBe(true);
    const stalemate = new Chess("k7/8/1QK5/8/8/8/8/8 b - - 0 1");
    expect(getGameStatus(stalemate)).toBe("stalemate");
    expect(getCheckedKingSquare(stalemate)).toBeNull();
    const insufficient = new Chess("7k/8/8/8/8/8/8/4K2B w - - 0 1");
    expect(getGameStatus(insufficient)).toBe("draw");
  });

  it.each([
    { color: "White", turn: "w", from: "e1", to: "g1", rookFrom: "h1", rookTo: "f1", flag: "k" },
    { color: "White", turn: "w", from: "e1", to: "c1", rookFrom: "a1", rookTo: "d1", flag: "q" },
    { color: "Black", turn: "b", from: "e8", to: "g8", rookFrom: "h8", rookTo: "f8", flag: "k" },
    { color: "Black", turn: "b", from: "e8", to: "c8", rookFrom: "a8", rookTo: "d8", flag: "q" },
  ] as const)("allows $color castling from King to $to and moves both pieces", ({ turn, from, to, rookFrom, rookTo, flag }) => {
    const game = new Chess(`r3k2r/8/8/8/8/8/8/R3K2R ${turn} KQkq - 0 1`);
    expect(legalMovesFrom(game, sq(from)).some(({ to: destination, flags }) => destination === to && flags.includes(flag))).toBe(true);

    const castling = move(game, from, to);
    expect(castling?.flags.includes(flag)).toBe(true);
    expect(game.get(to)?.type).toBe("k");
    expect(game.get(rookTo)?.type).toBe("r");
    expect(game.get(from)).toBeUndefined();
    expect(game.get(rookFrom)).toBeUndefined();
    expect(toPieceAnimation(castling!, 1).rook).toEqual({ from: rookFrom, to: rookTo });
  });

  it("rejects castling through attack", () => {
    const throughAttack = new Chess("k4r2/8/8/8/8/8/8/4K2R w K - 0 1");
    expect(legalMovesFrom(throughAttack, sq("e1")).some(({ flags }) => flags.includes("k"))).toBe(false);
  });

  it("keeps White king-side castling available after clearing the path through normal moves", () => {
    const game = new Chess();
    for (const [from, to] of [["e2", "e4"], ["e7", "e5"], ["g1", "f3"], ["b8", "c6"], ["f1", "c4"], ["g8", "f6"]]) {
      move(game, from, to);
    }

    expect(legalMovesFrom(game, sq("e1")).map(({ to }) => to)).toContain("g1");
    const castling = move(game, "e1", "g1");
    expect(castling?.san).toBe("O-O");
    expect(game.get("g1")?.type).toBe("k");
    expect(game.get("f1")?.type).toBe("r");
  });

  it("performs en passant, promotion, undo, and reset", () => {
    const game = new Chess();
    move(game, "e2", "e4"); move(game, "a7", "a6");
    move(game, "e4", "e5"); move(game, "d7", "d5");
    const enPassant = legalMovesFrom(game, sq("e5")).find(({ to }) => to === "d6");
    expect(enPassant?.flags.includes("e")).toBe(true);
    move(game, "e5", "d6");
    expect(game.get("d5")).toBeUndefined();
    expect(getCapturedPieces(game).w).toEqual(["p"]);

    const promotion = new Chess("4k3/P7/8/8/8/8/8/4K3 w - - 0 1");
    const promoted = move(promotion, "a7", "a8", "n");
    expect(promoted?.san).toBe("a8=N");
    expect(promotion.get("a8")?.type).toBe("n");
    promotion.undo();
    expect(promotion.get("a7")?.type).toBe("p");
    promotion.reset();
    expect(promotion.fen()).toBe(new Chess().fen());
  });
});

