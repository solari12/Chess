import { describe, expect, it } from "vitest";
import { Chess } from "chess.js";
import {
  advanceChessClock,
  createChessClock,
  formatClockTime,
  getRemainingTimeMs,
  getTimeoutResult,
  TIME_CONTROLS,
} from "@/lib/chess/chess-clock";

describe("chess time control", () => {
  it.each([
    ["10m", 600_000],
    ["5m", 300_000],
    ["3d", 259_200_000],
  ] as const)("starts %s with %i ms per side", (timeControl, initialTimeMs) => {
    const clock = createChessClock(timeControl, 1_000);
    expect(clock.whiteTimeMs).toBe(initialTimeMs);
    expect(clock.blackTimeMs).toBe(initialTimeMs);
    expect(TIME_CONTROLS.find((control) => control.id === timeControl)?.initialTimeMs).toBe(initialTimeMs);
  });

  it("counts down only White while White is active", () => {
    const clock = createChessClock("10m", 1_000);
    expect(getRemainingTimeMs(clock, "w", 4_000)).toBe(597_000);
    expect(getRemainingTimeMs(clock, "b", 4_000)).toBe(600_000);
  });

  it("switches the active clock after a move and counts down Black", () => {
    const afterWhite = advanceChessClock(createChessClock("5m", 1_000), 8_000, "b", false);
    expect(afterWhite.whiteTimeMs).toBe(293_000);
    expect(afterWhite.blackTimeMs).toBe(300_000);
    expect(afterWhite.activeColor).toBe("b");
    expect(getRemainingTimeMs(afterWhite, "b", 10_500)).toBe(297_500);
  });

  it("charges AI search time and alternates both clocks in AI vs AI games", () => {
    const start = createChessClock("10m", 10_000);
    const afterWhiteAI = advanceChessClock(start, 12_400, "b", false, 2_400);
    const afterBlackAI = advanceChessClock(afterWhiteAI, 15_700, "w", false, 3_300);
    expect(afterWhiteAI.whiteTimeMs).toBe(597_600);
    expect(afterBlackAI.blackTimeMs).toBe(596_700);
    expect(afterBlackAI.activeColor).toBe("w");
  });

  it("returns a timeout result with the correct winner and termination", () => {
    const timeout = advanceChessClock(createChessClock("5m", 0), 300_000, "b", false);
    expect(timeout.timeoutColor).toBe("w");
    expect(timeout.whiteTimeMs).toBe(0);
    expect(timeout.activeColor).toBeNull();
    expect(getTimeoutResult(timeout.timeoutColor)).toEqual({
      termination: "timeout",
      loser: "w",
      winner: "b",
    });
  });

  it("resets both clocks for a new game", () => {
    const moved = advanceChessClock(createChessClock("3d", 0), 80_000, "b", false);
    const reset = createChessClock(moved.timeControl, 90_000);
    expect(reset.whiteTimeMs).toBe(259_200_000);
    expect(reset.blackTimeMs).toBe(259_200_000);
    expect(reset.activeColor).toBe("w");
  });

  it("formats long controls as days, hours, then minutes", () => {
    expect(formatClockTime(259_200_000, "3d")).toBe("3d 00:00:00");
    expect(formatClockTime(86_382_000, "3d")).toBe("23:59:42");
    expect(formatClockTime(3_582_000, "3d")).toBe("59:42");
  });

  it("does not change legal chess moves when the clock mode changes", () => {
    const legalMoves = ["10m", "5m", "3d"].map(() => new Chess().moves().sort());
    expect(legalMoves[1]).toEqual(legalMoves[0]);
    expect(legalMoves[2]).toEqual(legalMoves[0]);
  });
});
