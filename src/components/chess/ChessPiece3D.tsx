"use client";

import { memo, useEffect, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import type { MutableRefObject } from "react";
import { CylinderGeometry, Group, MeshStandardMaterial, SphereGeometry, Vector3 } from "three";
import type { PieceSymbol, Square } from "chess.js";
import type { PieceAnimation } from "@/types/chess";

const whiteMaterial = new MeshStandardMaterial({ color: "#f1ecdf", roughness: 0.3, metalness: 0.12 });
const blackMaterial = new MeshStandardMaterial({ color: "#414641", roughness: 0.34, metalness: 0.3 });
const whiteTrim = new MeshStandardMaterial({ color: "#a47b45", roughness: 0.3, metalness: 0.58 });
const blackTrim = new MeshStandardMaterial({ color: "#bd9b68", roughness: 0.32, metalness: 0.52 });
const selectedMaterial = new MeshStandardMaterial({ color: "#9eae76", roughness: 0.32, metalness: 0.2 });
const commonBase = new CylinderGeometry(0.25, 0.31, 0.11, 28);
const commonFoot = new CylinderGeometry(0.21, 0.25, 0.08, 28);
const commonBand = new CylinderGeometry(0.205, 0.205, 0.045, 28);
const sphereHead = new SphereGeometry(0.115, 18, 14);

function squarePosition(square: Square) {
  const file = square.charCodeAt(0) - 97;
  const rank = Number(square[1]) - 1;
  return new Vector3(file - 3.5, 0.07, 3.5 - rank);
}

function AnimatedChessPiece({
  type,
  color,
  square,
  selected,
  animation,
  flattenProgress,
  onClick,
}: {
  type: PieceSymbol;
  color: "w" | "b";
  square: Square;
  selected: boolean;
  animation: PieceAnimation | null;
  flattenProgress: MutableRefObject<number>;
  onClick: (square: Square) => void;
}) {
  const group = useRef<Group>(null);
  const progressStart = useRef<number | null>(null);
  const moving = useRef<{ from: Vector3; to: Vector3 } | null>(null);
  const white = color === "w";
  const body = white ? whiteMaterial : blackMaterial;
  const trim = white ? whiteTrim : blackTrim;
  const moveFrom = animation?.move.to === square ? animation.move.from : animation?.rook?.to === square ? animation.rook.from : null;
  const moveTo = moveFrom ? squarePosition(square) : null;

  useEffect(() => {
    if (!group.current) return;
    if (moveFrom && moveTo) {
      const from = squarePosition(moveFrom);
      moving.current = { from, to: moveTo };
      progressStart.current = null;
      group.current.position.copy(from);
    } else {
      moving.current = null;
      group.current.position.copy(squarePosition(square));
    }
  }, [animation?.id, moveFrom, moveTo, square]);

  useFrame(({ clock }) => {
    if (!group.current) return;
    group.current.scale.y = 1 - flattenProgress.current * 0.9;
    if (!moving.current) return;
    const now = clock.getElapsedTime();
    progressStart.current ??= now;
    const t = Math.min((now - progressStart.current) / 0.24, 1);
    const eased = 1 - Math.pow(1 - t, 3);
    group.current.position.lerpVectors(moving.current.from, moving.current.to, eased);
    group.current.position.y += Math.sin(Math.PI * t) * 0.28;
  });

  return (
    <group ref={group} dispose={null} position={squarePosition(square)} onClick={(event) => { event.stopPropagation(); onClick(square); }}>
      {selected && <mesh position={[0, 0.025, 0]} rotation={[-Math.PI / 2, 0, 0]} material={selectedMaterial}>
        <torusGeometry args={[0.34, 0.035, 8, 36]} />
      </mesh>}
      <mesh position={[0, 0.055, 0]} geometry={commonBase} material={body} castShadow receiveShadow />
      <mesh position={[0, 0.15, 0]} geometry={commonFoot} material={trim} castShadow />
      <mesh position={[0, 0.205, 0]} geometry={commonBand} material={body} />
      <PieceSilhouette type={type} body={body} trim={trim} />
    </group>
  );
}

function PieceSilhouette({ type, body, trim }: { type: PieceSymbol; body: MeshStandardMaterial; trim: MeshStandardMaterial }) {
  if (type === "p") return <>
    <mesh position={[0, 0.34, 0]} material={body} castShadow><cylinderGeometry args={[0.11, 0.18, 0.29, 20]} /></mesh>
    <mesh position={[0, 0.52, 0]} material={trim}><cylinderGeometry args={[0.105, 0.14, 0.075, 20]} /></mesh>
    <mesh position={[0, 0.66, 0]} geometry={sphereHead} material={body} castShadow />
  </>;

  if (type === "r") return <>
    <mesh position={[0, 0.34, 0]} material={body} castShadow><cylinderGeometry args={[0.18, 0.2, 0.3, 24]} /></mesh>
    <mesh position={[0, 0.52, 0]} material={trim}><cylinderGeometry args={[0.21, 0.18, 0.075, 24]} /></mesh>
    <mesh position={[0, 0.63, 0]} material={body} castShadow><cylinderGeometry args={[0.19, 0.19, 0.18, 24]} /></mesh>
    {[-0.12, -0.04, 0.04, 0.12].map((x) => <mesh key={x} position={[x, 0.755, 0]} material={trim} castShadow><boxGeometry args={[0.055, 0.085, 0.23]} /></mesh>)}
  </>;

  if (type === "n") return <>
    <mesh position={[0, 0.37, 0]} material={body} castShadow><cylinderGeometry args={[0.12, 0.2, 0.35, 20]} /></mesh>
    <mesh position={[0.025, 0.62, 0]} rotation={[0, 0, -0.23]} scale={[0.82, 1.45, 0.72]} geometry={sphereHead} material={body} castShadow />
    <mesh position={[0.055, 0.77, 0]} rotation={[0, 0, -0.35]} material={body} castShadow><coneGeometry args={[0.105, 0.22, 4]} /></mesh>
    <mesh position={[0.1, 0.61, 0.11]} scale={[0.35, 0.35, 0.35]} geometry={sphereHead} material={trim} />
  </>;

  if (type === "b") return <>
    <mesh position={[0, 0.39, 0]} material={body} castShadow><cylinderGeometry args={[0.12, 0.19, 0.38, 22]} /></mesh>
    <mesh position={[0, 0.65, 0]} material={body} castShadow><coneGeometry args={[0.16, 0.35, 24]} /></mesh>
    <mesh position={[0, 0.86, 0]} geometry={sphereHead} material={trim} />
    <mesh position={[0, 0.91, 0]} material={body}><sphereGeometry args={[0.045, 12, 10]} /></mesh>
  </>;

  if (type === "q") return <>
    <mesh position={[0, 0.39, 0]} material={body} castShadow><cylinderGeometry args={[0.12, 0.2, 0.4, 24]} /></mesh>
    <mesh position={[0, 0.67, 0]} material={trim}><torusGeometry args={[0.15, 0.025, 8, 24]} /></mesh>
    <mesh position={[0, 0.78, 0]} material={body} castShadow><coneGeometry args={[0.18, 0.23, 24]} /></mesh>
    {Array.from({ length: 5 }, (_, index) => {
      const angle = (index / 5) * Math.PI * 2;
      return <mesh key={index} position={[Math.cos(angle) * 0.13, 0.92, Math.sin(angle) * 0.13]} geometry={sphereHead} material={trim} scale={0.48} castShadow />;
    })}
    <mesh position={[0, 0.95, 0]} geometry={sphereHead} material={trim} scale={0.55} />
  </>;

  return <>
    <mesh position={[0, 0.39, 0]} material={body} castShadow><cylinderGeometry args={[0.12, 0.2, 0.4, 24]} /></mesh>
    <mesh position={[0, 0.68, 0]} material={trim}><torusGeometry args={[0.155, 0.026, 8, 24]} /></mesh>
    <mesh position={[0, 0.82, 0]} material={body} castShadow><coneGeometry args={[0.13, 0.25, 24]} /></mesh>
    <mesh position={[0, 1.02, 0]} material={trim}><boxGeometry args={[0.075, 0.29, 0.075]} /></mesh>
    <mesh position={[0, 1.02, 0]} material={trim}><boxGeometry args={[0.24, 0.065, 0.075]} /></mesh>
    <mesh position={[0, 1.02, 0]} material={trim}><boxGeometry args={[0.075, 0.065, 0.24]} /></mesh>
  </>;
}

function animationIdForPiece(props: { square: Square; animation: PieceAnimation | null }) {
  if (!props.animation) return null;
  if (props.animation.move.to === props.square || props.animation.rook?.to === props.square) return props.animation.id;
  return null;
}

export const ChessPiece3D = memo(AnimatedChessPiece, (previous, next) =>
  previous.type === next.type && previous.color === next.color && previous.square === next.square && previous.selected === next.selected && previous.flattenProgress === next.flattenProgress && animationIdForPiece(previous) === animationIdForPiece(next) && previous.onClick === next.onClick,
);
