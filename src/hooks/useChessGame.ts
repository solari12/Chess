"use client";

import { useCallback, useRef, useState } from "react";
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
import type { PieceAnimation, PromotionRequest } from "@/types/chess";

export function useChessGame() {
  const [game] = useState<Chess>(createChessGame);
  const animationId = useRef(0);
  const [positionKey, setPositionKey] = useState(() => game.fen());
  const [selectedSquare, setSelectedSquare] = useState<Square | null>(null);
  const [promotionRequest, setPromotionRequest] = useState<PromotionRequest | null>(null);
  const [animation, setAnimation] = useState<PieceAnimation | null>(null);

  const history = game.history({ verbose: true });
  const legalMoves = selectedSquare ? legalMovesFrom(game, selectedSquare) : [];
  const legalDestinations = [...new Set(legalMoves.map((move) => move.to))];
  const captured = getCapturedPieces(game);
  const status = getGameStatus(game);
  const checkedKingSquare = getCheckedKingSquare(game);
  const lastMove = getLastMove(game);

  const commitMove = useCallback((from: Square, to: Square, promotion?: PieceSymbol) => {
    const move = tryMove(game, from, to, promotion);
    if (!move) return false;
    const id = ++animationId.current;
    setAnimation(toPieceAnimation(move, id));
    window.setTimeout(() => setAnimation((current) => current?.id === id ? null : current), 280);
    setSelectedSquare(null);
    setPromotionRequest(null);
    setPositionKey(game.fen());
    return true;
  }, [game]);

  const selectSquare = useCallback((square: Square) => {
    if (game.isGameOver() || promotionRequest) return;
    const piece = game.get(square);
    if (!piece || piece.color !== game.turn()) {
      setSelectedSquare(null);
      return;
    }
    setSelectedSquare(square);
  }, [game, promotionRequest]);

  const handleSquareClick = useCallback((square: Square) => {
    if (game.isGameOver() || promotionRequest) return;
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
  }, [commitMove, game, promotionRequest, selectSquare, selectedSquare]);

  const choosePromotion = useCallback((piece: PieceSymbol) => {
    if (!promotionRequest) return false;
    return commitMove(promotionRequest.from, promotionRequest.to, piece);
  }, [commitMove, promotionRequest]);

  const cancelPromotion = useCallback(() => setPromotionRequest(null), []);

  const undo = useCallback(() => {
    if (!game.undo()) return false;
    setSelectedSquare(null);
    setPromotionRequest(null);
    setAnimation(null);
    setPositionKey(game.fen());
    return true;
  }, [game]);

  const newGame = useCallback(() => {
    game.reset();
    setSelectedSquare(null);
    setPromotionRequest(null);
    setAnimation(null);
    setPositionKey(game.fen());
  }, [game]);

  return {
    game,
    positionKey,
    board: game.board(),
    turn: game.turn() as Color,
    history,
    captured,
    status,
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
