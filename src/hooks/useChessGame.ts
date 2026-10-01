"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Chess, type Color, type PieceSymbol, type Square } from "chess.js";
import {
  createChessGame,
  getCapturedPieces,
  getCheckedKingSquare,
  getGameStatus,
  getLastMove,
  getPromotionChoices,
  legalMovesFrom,
  toPieceAnimation,
  tryMove,
} from "@/lib/chess/chess-game";
import { parseUciMove, requestAIMove } from "@/lib/chess/ai-client";
import {
  advanceChessClock,
  createChessClock,
  freezeClockAt,
  getRemainingTimeMs,
  getTimeoutResult,
  type ChessClockState,
  type ClockMoveSnapshot,
  type TimeControlId,
} from "@/lib/chess/chess-clock";
import type { GameTermination, GeneticPlayerProfile, PieceAnimation, PlayerRole, PlayerRoles, PromotionRequest } from "@/types/chess";

const AI_DEPTH = 3;

export function useChessGame() {
  const [game] = useState<Chess>(createChessGame);
  const animationId = useRef(0);
  const aiRequestRef = useRef<{
    key: string;
    controller: AbortController;
    lease: number;
    cancelPending: boolean;
  } | null>(null);
  const [playerRoles, setPlayerRoles] = useState<PlayerRoles>({ w: "human", b: "human" });
  const playerRolesRef = useRef(playerRoles);
  const [geneticProfile, setGeneticProfile] = useState<GeneticPlayerProfile | null>(null);
  const [aiThinking, setAiThinking] = useState(false);
  const [timeControl, setTimeControlState] = useState<TimeControlId>("10m");
  const [clock, setClock] = useState<ChessClockState>(() => createChessClock("10m", Date.now()));
  const clockRef = useRef(clock);
  const [clockNow, setClockNow] = useState(() => Date.now());
  const [clockHistory, setClockHistory] = useState<ClockMoveSnapshot[]>([]);
  const clockHistoryRef = useRef(clockHistory);
  const [positionKey, setPositionKey] = useState(() => game.fen());
  const positionRevision = useRef(0);
  const [selectedSquare, setSelectedSquare] = useState<Square | null>(null);
  const [promotionRequest, setPromotionRequest] = useState<PromotionRequest | null>(null);
  const [animation, setAnimation] = useState<PieceAnimation | null>(null);

  const history = game.history({ verbose: true });
  const legalMoves = selectedSquare ? legalMovesFrom(game, selectedSquare) : [];
  const legalDestinations = [...new Set(legalMoves.map((move) => move.to))];
  const captured = getCapturedPieces(game);
  const status = clock.timeoutColor ? "timeout" : getGameStatus(game);
  const termination: GameTermination = clock.timeoutColor
    ? "timeout"
    : game.isCheckmate()
      ? "checkmate"
      : game.isStalemate()
        ? "stalemate"
        : game.isDraw()
          ? "draw"
          : null;
  const currentRole = playerRoles[game.turn()];
  const checkedKingSquare = getCheckedKingSquare(game);
  const lastMove = getLastMove(game);

  const writeClock = useCallback((next: ChessClockState) => {
    clockRef.current = next;
    setClock(next);
  }, []);

  const refreshPositionKey = useCallback(() => {
    positionRevision.current += 1;
    setPositionKey(`${game.fen()}:${positionRevision.current}`);
  }, [game]);

  const commitMove = useCallback((from: Square, to: Square, promotion?: PieceSymbol, elapsedOverrideMs?: number) => {
    const currentClock = clockRef.current;
    if (currentClock.timeoutColor || currentClock.activeColor !== game.turn()) return false;
    const now = Date.now();
    const elapsedSinceTurnStart = currentClock.activeSince === null ? 0 : Math.max(0, now - currentClock.activeSince);
    const elapsedMs = elapsedOverrideMs === undefined
      ? elapsedSinceTurnStart
      : Math.max(elapsedSinceTurnStart, elapsedOverrideMs);
    const activeRemaining = game.turn() === "w" ? currentClock.whiteTimeMs : currentClock.blackTimeMs;
    if (activeRemaining - elapsedMs <= 0) {
      writeClock(freezeClockAt(currentClock, now, game.turn()));
      return false;
    }

    const move = tryMove(game, from, to, promotion);
    if (!move) return false;
    const nextClock = advanceChessClock(currentClock, now, game.turn(), game.isGameOver(), elapsedMs);
    writeClock(nextClock);
    const clockSnapshot: ClockMoveSnapshot = {
      move: `${move.from}${move.to}${move.promotion ?? ""}`,
      fen: game.fen(),
      timestamp: Date.now(),
      elapsedMs,
      whiteTimeMs: nextClock.whiteTimeMs,
      blackTimeMs: nextClock.blackTimeMs,
    };
    const nextHistory = [...clockHistoryRef.current, clockSnapshot];
    clockHistoryRef.current = nextHistory;
    setClockHistory(nextHistory);
    const id = ++animationId.current;
    setAnimation(toPieceAnimation(move, id));
    window.setTimeout(() => setAnimation((current) => current?.id === id ? null : current), 280);
    setSelectedSquare(null);
    setPromotionRequest(null);
    refreshPositionKey();
    return true;
  }, [game, refreshPositionKey, writeClock]);

  const resetGame = useCallback((nextTimeControl: TimeControlId = timeControl) => {
    aiRequestRef.current?.controller.abort();
    aiRequestRef.current = null;
    setAiThinking(false);
    game.reset();
    setSelectedSquare(null);
    setPromotionRequest(null);
    setAnimation(null);
    setTimeControlState(nextTimeControl);
    writeClock(createChessClock(nextTimeControl, Date.now()));
    clockHistoryRef.current = [];
    setClockHistory([]);
    refreshPositionKey();
  }, [game, refreshPositionKey, timeControl, writeClock]);

  const chooseTimeControl = useCallback((nextTimeControl: TimeControlId) => {
    resetGame(nextTimeControl);
  }, [resetGame]);

  useEffect(() => {
    const timer = window.setInterval(() => {
      const now = Date.now();
      setClockNow((displayedNow) => Math.floor(displayedNow / 1_000) === Math.floor(now / 1_000) ? displayedNow : now);
      const current = clockRef.current;
      if (
        current.activeColor !== null
        && current.activeSince !== null
        && !current.timeoutColor
        && !game.isGameOver()
        && getRemainingTimeMs(current, current.activeColor, now) <= 0
      ) {
        writeClock(freezeClockAt(current, now, current.activeColor));
        setSelectedSquare(null);
        setPromotionRequest(null);
      }
    }, 200);
    return () => window.clearInterval(timer);
  }, [game, writeClock]);

  const setPlayerRole = useCallback((color: Color, role: PlayerRole) => {
    if (playerRolesRef.current[color] !== role) {
      const next = { ...playerRolesRef.current, [color]: role };
      playerRolesRef.current = next;
      setPlayerRoles(next);
    }
    setSelectedSquare(null);
    setPromotionRequest(null);
  }, []);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      const savedProfile = window.localStorage.getItem("chess.genetic-player-profile");
      if (savedProfile) {
        try {
          const parsed: unknown = JSON.parse(savedProfile);
          if (isGeneticPlayerProfile(parsed)) setGeneticProfile(parsed);
        } catch {
          window.localStorage.removeItem("chess.genetic-player-profile");
        }
      }
      if (window.localStorage.getItem("chess.genetic-auto-play") === "black") {
        window.localStorage.removeItem("chess.genetic-auto-play");
        setPlayerRole("b", "genetic");
      }
    }, 0);
    return () => window.clearTimeout(timer);
  }, [setPlayerRole]);

  useEffect(() => {
    if (currentRole === "human" || game.isGameOver() || clock.timeoutColor) return;

    const fen = game.fen();
    const turn = game.turn();
    const key = `${fen}:${currentRole}:${currentRole === "genetic" ? geneticProfile?.candidate_id ?? "missing" : ""}`;
    if (aiRequestRef.current?.key === key) {
      const existingRequest = aiRequestRef.current;
      existingRequest.lease += 1;
      existingRequest.cancelPending = false;
      return () => {
        const lease = existingRequest.lease;
        existingRequest.cancelPending = true;
        queueMicrotask(() => {
          if (existingRequest.lease !== lease || aiRequestRef.current !== existingRequest) return;
          existingRequest.controller.abort();
          aiRequestRef.current = null;
          setAiThinking(false);
        });
      };
    }

    aiRequestRef.current?.controller.abort();
    const controller = new AbortController();
    const request = { key, controller, lease: 0, cancelPending: false };
    aiRequestRef.current = request;
    setAiThinking(true);

    void requestAIMove(fen, currentRole, AI_DEPTH, controller.signal, geneticProfile)
      .then((result) => {
        if (!result || aiRequestRef.current !== request || controller.signal.aborted || request.cancelPending) return;
        if (clockRef.current.timeoutColor || game.isGameOver() || game.turn() !== turn || game.fen() !== fen || playerRolesRef.current[turn] !== currentRole) return;

        const move = parseUciMove(result.move);
        if (!move) {
          console.error(`Chess AI returned malformed UCI move: ${result.move}`);
          return;
        }

        if (!commitMove(move.from, move.to, move.promotion, result.time_ms)) {
          console.error(`Chess AI returned an illegal move for the current position: ${result.move}`);
        }
      })
      .catch((error: unknown) => {
        if (!controller.signal.aborted && !request.cancelPending && aiRequestRef.current === request) {
          console.error("Unable to get a move from the chess AI service:", error);
        }
      })
      .finally(() => {
        if (aiRequestRef.current === request) {
          aiRequestRef.current = null;
          setAiThinking(false);
        }
      });

    return () => {
      const lease = request.lease;
      request.cancelPending = true;
      queueMicrotask(() => {
        if (request.lease !== lease || aiRequestRef.current !== request) return;
        controller.abort();
        aiRequestRef.current = null;
        setAiThinking(false);
      });
    };
  }, [clock.timeoutColor, commitMove, currentRole, game, geneticProfile, playerRoles, positionKey]);

  const clockTimes = {
    w: getRemainingTimeMs(clock, "w", clockNow),
    b: getRemainingTimeMs(clock, "b", clockNow),
  };

  const selectSquare = useCallback((square: Square) => {
    if (clockRef.current.timeoutColor || game.isGameOver() || promotionRequest || playerRoles[game.turn()] !== "human") return;
    const piece = game.get(square);
    if (!piece || piece.color !== game.turn()) {
      setSelectedSquare(null);
      return;
    }
    setSelectedSquare(square);
  }, [game, playerRoles, promotionRequest]);

  const handleSquareClick = useCallback((square: Square) => {
    if (clockRef.current.timeoutColor || game.isGameOver() || promotionRequest || playerRoles[game.turn()] !== "human") return;
    if (selectedSquare) {
      const destinationMoves = selectedSquare ? legalMovesFrom(game, selectedSquare).filter((move) => move.to === square) : [];
      if (destinationMoves.length) {
        const promotion = destinationMoves.some((move) => Boolean(move.promotion));
        if (promotion) {
          setPromotionRequest({ from: selectedSquare, to: square, color: game.turn() });
          return;
        }
        commitMove(selectedSquare, square);
        return;
      }
    }
    selectSquare(square);
  }, [commitMove, game, playerRoles, promotionRequest, selectSquare, selectedSquare]);

  const choosePromotion = useCallback((piece: PieceSymbol) => {
    if (!promotionRequest || playerRoles[game.turn()] !== "human") return false;
    return commitMove(promotionRequest.from, promotionRequest.to, piece);
  }, [commitMove, game, playerRoles, promotionRequest]);

  const cancelPromotion = useCallback(() => setPromotionRequest(null), []);

  const undo = useCallback(() => {
    if (clockRef.current.timeoutColor) return false;
    if (!game.undo()) return false;
    setSelectedSquare(null);
    setPromotionRequest(null);
    setAnimation(null);
    const now = Date.now();
    const frozen = freezeClockAt(clockRef.current, now);
    const nextClock = { ...frozen, activeColor: game.isGameOver() ? null : game.turn(), activeSince: game.isGameOver() ? null : now };
    writeClock(nextClock);
    const nextHistory = clockHistoryRef.current.slice(0, -1);
    if (nextHistory.length > 0) {
      const lastIndex = nextHistory.length - 1;
      nextHistory[lastIndex] = {
        ...nextHistory[lastIndex],
        whiteTimeMs: nextClock.whiteTimeMs,
        blackTimeMs: nextClock.blackTimeMs,
      };
    }
    clockHistoryRef.current = nextHistory;
    setClockHistory(nextHistory);
    refreshPositionKey();
    return true;
  }, [game, refreshPositionKey, writeClock]);

  const newGame = useCallback(() => {
    resetGame();
  }, [resetGame]);

  return {
    game,
    playerRoles,
    geneticProfile,
    setPlayerRole,
    currentRole,
    aiThinking,
    positionKey,
    board: game.board(),
    turn: game.turn() as Color,
    history,
    captured,
    status,
    termination,
    timeControl,
    chooseTimeControl,
    clock,
    clockTimes,
    clockHistory,
    timeoutLoser: clock.timeoutColor,
    timeoutResult: getTimeoutResult(clock.timeoutColor),
    clockActiveColor: clock.activeColor,
    checkedKingSquare,
    lastMove,
    moveNumber: game.moveNumber(),
    moveCount: history.length,
    selectedSquare,
    legalMoves,
    legalDestinations,
    promotionRequest,
    promotionChoices: promotionRequest ? getPromotionChoices(game, promotionRequest.from, promotionRequest.to) : [],
    animation,
    selectSquare,
    handleSquareClick,
    choosePromotion,
    cancelPromotion,
    undo,
    newGame,
  };
}

function isGeneticPlayerProfile(value: unknown): value is GeneticPlayerProfile {
  if (typeof value !== "object" || value === null) return false;
  const candidate = value as Partial<GeneticPlayerProfile>;
  const weights = candidate.weights;
  return typeof candidate.candidate_id === "string"
    && typeof candidate.fitness === "number"
    && typeof weights === "object"
    && weights !== null
    && ["pawn", "knight", "bishop", "rook", "queen"].every((gene) =>
      Number.isInteger(weights[gene as keyof typeof weights]),
    );
}
