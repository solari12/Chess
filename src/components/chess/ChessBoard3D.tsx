"use client";

import { Text } from "@react-three/drei";
import { useFrame } from "@react-three/fiber";
import { useRef, type MutableRefObject } from "react";
import type { Piece, Square } from "chess.js";
import type { Group } from "three";
import { ChessPiece3D } from "@/components/chess/ChessPiece3D";
import type { PieceAnimation } from "@/types/chess";

interface ChessBoard3DProps {
  board: (Piece | null)[][];
  selectedSquare: Square | null;
  legalDestinations: Square[];
  lastMove: { from: Square; to: Square } | null;
  checkedKingSquare: Square | null;
  animation: PieceAnimation | null;
  flipped: boolean;
  flattenProgress: MutableRefObject<number>;
  onSquareClick: (square: Square) => void;
}

function BoardSquare({
  square,
  file,
  rank,
  selected,
  legal,
  capture,
  lastMove,
  check,
  onClick,
}: {
  square: Square;
  file: number;
  rank: number;
  selected: boolean;
  legal: boolean;
  capture: boolean;
  lastMove: boolean;
  check: boolean;
  onClick: (square: Square) => void;
}) {
  const x = file - 3.5;
  const z = 3.5 - (rank - 1);
  const dark = (file + rank) % 2 === 0;
  return (
    <group position={[x, 0, z]} onClick={(event) => { event.stopPropagation(); onClick(square); }}>
      <mesh receiveShadow castShadow>
        <boxGeometry args={[0.99, 0.13, 0.99]} />
        <meshStandardMaterial color={dark ? "#624b37" : "#e8dfcb"} roughness={dark ? 0.55 : 0.68} />
      </mesh>
      {lastMove && <mesh position={[0, 0.071, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[0.99, 0.99]} />
        <meshBasicMaterial color="#c4a45f" transparent opacity={0.25} depthWrite={false} />
      </mesh>}
      {selected && <mesh position={[0, 0.076, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[0.99, 0.99]} />
        <meshBasicMaterial color="#99ae71" transparent opacity={0.43} depthWrite={false} />
      </mesh>}
      {check && <mesh position={[0, 0.083, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <circleGeometry args={[0.49, 40]} />
        <meshBasicMaterial color="#bd6253" transparent opacity={0.45} depthWrite={false} />
      </mesh>}
      {legal && !capture && <mesh position={[0, 0.11, 0]}>
        <cylinderGeometry args={[0.12, 0.12, 0.035, 24]} />
        <meshStandardMaterial color="#637e4e" roughness={0.42} metalness={0.15} />
      </mesh>}
      {legal && capture && <mesh position={[0, 0.105, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <torusGeometry args={[0.39, 0.035, 8, 40]} />
        <meshStandardMaterial color="#b77f4b" roughness={0.38} metalness={0.32} />
      </mesh>}
    </group>
  );
}

function BoardCoordinates() {
  const files = "abcdefgh";
  return <group>
    {Array.from(files).map((file, index) => <Text key={`file-${file}`} position={[index - 3.5, 0.018, 4.09]} rotation={[-Math.PI / 2, 0, 0]} fontSize={0.15} color="#ded5c2" anchorX="center" anchorY="middle">{file}</Text>)}
    {Array.from({ length: 8 }, (_, index) => <Text key={`rank-${index}`} position={[-4.09, 0.018, 3.5 - index]} rotation={[-Math.PI / 2, 0, 0]} fontSize={0.15} color="#ded5c2" anchorX="center" anchorY="middle">{String(8 - index)}</Text>)}
  </group>;
}

export function ChessBoard3D({ board, selectedSquare, legalDestinations, lastMove, checkedKingSquare, animation, flipped, flattenProgress, onSquareClick }: ChessBoard3DProps) {
  const boardDepth = useRef<Group>(null);
  useFrame(() => {
    if (boardDepth.current) {
      boardDepth.current.scale.y = 1 - flattenProgress.current * 0.25;
    }
  });

  const legalSet = new Set(legalDestinations);
  const pieces: { square: Square; piece: Piece }[] = [];
  for (let row = 0; row < 8; row += 1) {
    for (let file = 0; file < 8; file += 1) {
      const piece = board[row][file];
      if (piece) pieces.push({ square: `${String.fromCharCode(97 + file)}${8 - row}` as Square, piece });
    }
  }

  return (
    <group rotation={[0, flipped ? Math.PI : 0, 0]}>
      <group ref={boardDepth}>
      <mesh position={[0, -0.17, 0]} receiveShadow castShadow>
        <boxGeometry args={[8.62, 0.34, 8.62]} />
        <meshStandardMaterial color="#392d22" roughness={0.5} metalness={0.16} />
      </mesh>
      <mesh position={[0, 0.005, 0]}>
        <boxGeometry args={[8.42, 0.025, 8.42]} />
        <meshStandardMaterial color="#b58b50" roughness={0.36} metalness={0.62} />
      </mesh>
      {Array.from({ length: 8 }, (_, row) => Array.from({ length: 8 }, (_, file) => {
        const square = `${String.fromCharCode(97 + file)}${8 - row}` as Square;
        const piece = board[row][file];
        return <BoardSquare key={square} square={square} file={file} rank={8 - row}
          selected={selectedSquare === square} legal={legalSet.has(square)} capture={Boolean(piece)}
          lastMove={lastMove?.from === square || lastMove?.to === square} check={checkedKingSquare === square}
          onClick={onSquareClick} />;
      }))}
      </group>
      {pieces.map(({ square, piece }) => (
        <ChessPiece3D key={square} type={piece.type} color={piece.color} square={square}
          selected={selectedSquare === square} animation={animation} flattenProgress={flattenProgress} onClick={onSquareClick} />
      ))}
      <BoardCoordinates />
    </group>
  );
}
