import { afterEach, describe, expect, it, vi } from "vitest";
import { Chess } from "chess.js";
import { parseUciMove, requestAIMove } from "@/lib/chess/ai-client";
import { tryMove } from "@/lib/chess/chess-game";

describe("chess AI client", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("does not call the API for a human role", async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);

    await expect(requestAIMove("fen", "human", 3)).resolves.toBeNull();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it.each(["minimax", "alpha_beta"] as const)("requests an AI move for %s", async (role) => {
    const responseData = {
      move: "e2e4",
      algorithm: role,
      depth: 3,
      score: 0,
      nodes: 123,
      time_ms: 4.5,
    };
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify(responseData), { status: 200 }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await expect(requestAIMove("start-fen", role, 3)).resolves.toEqual(responseData);
    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:8000/api/ai/move",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ fen: "start-fen", algorithm: role, depth: 3 }),
      }),
    );
  });

  it("parses normal and promotion UCI moves and rejects malformed moves", () => {
    expect(parseUciMove("e7e5")).toEqual({ from: "e7", to: "e5" });
    expect(parseUciMove("e7e8q")).toEqual({ from: "e7", to: "e8", promotion: "q" });
    expect(parseUciMove("e2e9")).toBeNull();
  });

  it("applies AI captures, checkmate, and promotion through chess.js", () => {
    const capture = new Chess();
    capture.move("e4");
    capture.move("d5");
    const captureMove = parseUciMove("e4d5");
    expect(captureMove).not.toBeNull();
    expect(tryMove(capture, captureMove!.from, captureMove!.to)).not.toBeNull();
    expect(capture.history()).toEqual(["e4", "d5", "exd5"]);

    const mate = new Chess();
    mate.move("f3");
    mate.move("e5");
    mate.move("g4");
    const mateMove = parseUciMove("d8h4");
    expect(mateMove).not.toBeNull();
    expect(tryMove(mate, mateMove!.from, mateMove!.to)).not.toBeNull();
    expect(mate.isCheckmate()).toBe(true);

    const promotion = new Chess("k7/4P3/8/8/8/8/8/4K3 w - - 0 1");
    const promotionMove = parseUciMove("e7e8q");
    expect(promotionMove).not.toBeNull();
    expect(tryMove(promotion, promotionMove!.from, promotionMove!.to, promotionMove!.promotion)).not.toBeNull();
    expect(promotion.get("e8")?.type).toBe("q");
  });
});
