import type { Color } from "chess.js";

export const TIME_CONTROLS = [
  { id: "10m", label: "10 min", initialTimeMs: 600_000 },
  { id: "5m", label: "5 min", initialTimeMs: 300_000 },
  { id: "3d", label: "3 days", initialTimeMs: 259_200_000 },
] as const;

export type TimeControlId = (typeof TIME_CONTROLS)[number]["id"];

export interface ChessClockState {
  timeControl: TimeControlId;
  whiteTimeMs: number;
  blackTimeMs: number;
  activeColor: Color | null;
  activeSince: number | null;
  timeoutColor: Color | null;
}

export interface ClockMoveSnapshot {
  move: string;
  fen: string;
  timestamp: number;
  elapsedMs: number;
  whiteTimeMs: number;
  blackTimeMs: number;
}

export interface TimeoutResult {
  termination: "timeout";
  loser: Color;
  winner: Color;
}

export function createChessClock(timeControl: TimeControlId, now: number): ChessClockState {
  const initialTimeMs = TIME_CONTROLS.find((control) => control.id === timeControl)!.initialTimeMs;
  return {
    timeControl,
    whiteTimeMs: initialTimeMs,
    blackTimeMs: initialTimeMs,
    activeColor: "w",
    activeSince: now,
    timeoutColor: null,
  };
}

export function getRemainingTimeMs(clock: ChessClockState, color: Color, now: number): number {
  const savedTime = color === "w" ? clock.whiteTimeMs : clock.blackTimeMs;
  if (clock.activeColor !== color || clock.activeSince === null) return savedTime;
  return Math.max(0, savedTime - Math.max(0, now - clock.activeSince));
}

export function advanceChessClock(
  clock: ChessClockState,
  now: number,
  nextTurn: Color,
  gameOver: boolean,
  elapsedOverrideMs?: number,
): ChessClockState {
  const activeColor = clock.activeColor ?? nextTurn;
  const activeTime = activeColor === "w" ? clock.whiteTimeMs : clock.blackTimeMs;
  const elapsed = Math.max(0, elapsedOverrideMs ?? (
    clock.activeColor === activeColor && clock.activeSince !== null
      ? now - clock.activeSince
      : 0
  ));
  const remaining = Math.max(0, activeTime - elapsed);
  const whiteTimeMs = activeColor === "w" ? remaining : clock.whiteTimeMs;
  const blackTimeMs = activeColor === "b" ? remaining : clock.blackTimeMs;
  const timedOut = remaining <= 0;

  return {
    ...clock,
    whiteTimeMs,
    blackTimeMs,
    activeColor: timedOut || gameOver ? null : nextTurn,
    activeSince: timedOut || gameOver ? null : now,
    timeoutColor: timedOut ? activeColor : null,
  };
}

export function freezeClockAt(
  clock: ChessClockState,
  now: number,
  timeoutColor: Color | null = null,
): ChessClockState {
  const whiteTimeMs = getRemainingTimeMs(clock, "w", now);
  const blackTimeMs = getRemainingTimeMs(clock, "b", now);
  if (timeoutColor === "w") return { ...clock, whiteTimeMs: 0, blackTimeMs, activeColor: null, activeSince: null, timeoutColor };
  if (timeoutColor === "b") return { ...clock, whiteTimeMs, blackTimeMs: 0, activeColor: null, activeSince: null, timeoutColor };
  return { ...clock, whiteTimeMs, blackTimeMs, activeColor: null, activeSince: null, timeoutColor: null };
}

export function getTimeoutResult(loser: Color | null): TimeoutResult | null {
  if (!loser) return null;
  return { termination: "timeout", loser, winner: loser === "w" ? "b" : "w" };
}

export function formatClockTime(timeMs: number, timeControl: TimeControlId): string {
  const totalSeconds = Math.ceil(Math.max(0, timeMs) / 1_000);
  const days = Math.floor(totalSeconds / 86_400);
  const hours = Math.floor((totalSeconds % 86_400) / 3_600);
  const minutes = Math.floor((totalSeconds % 3_600) / 60);
  const seconds = totalSeconds % 60;
  const twoDigits = (value: number) => String(value).padStart(2, "0");

  if (timeControl === "3d") {
    if (days > 0) return `${days}d ${twoDigits(hours)}:${twoDigits(minutes)}:${twoDigits(seconds)}`;
    if (hours > 0) return `${twoDigits(hours)}:${twoDigits(minutes)}:${twoDigits(seconds)}`;
  }
  return `${twoDigits(minutes + hours * 60)}:${twoDigits(seconds)}`;
}
