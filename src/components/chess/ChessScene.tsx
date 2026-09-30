"use client";

import { OrbitControls, PerspectiveCamera } from "@react-three/drei";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { Suspense, useCallback, useEffect, useRef } from "react";
import type { Piece, Square } from "chess.js";
import { Vector3 } from "three";
import { ChessBoard3D } from "@/components/chess/ChessBoard3D";
import type { PieceAnimation, ChessViewMode } from "@/types/chess";

interface ChessSceneProps {
  board: (Piece | null)[][];
  selectedSquare: Square | null;
  legalDestinations: Square[];
  lastMove: { from: Square; to: Square } | null;
  checkedKingSquare: Square | null;
  animation: PieceAnimation | null;
  flipped: boolean;
  resetCameraKey: number;
  viewMode: ChessViewMode;
  onTransitionComplete: (mode: "3d" | "2d") => void;
  onSquareClick: (square: Square) => void;
}

interface OrbitControlsApi {
  target: Vector3;
  update: () => void;
  reset: () => void;
}

interface CameraTransition {
  startedAt: number;
  duration: number;
  fromPosition: Vector3;
  toPosition: Vector3;
  fromTarget: Vector3;
  toTarget: Vector3;
  toMode: "3d" | "2d";
}

const default3DPosition = new Vector3(6.2, 9.4, 10.8);
const boardCenter = new Vector3(0, 0, 0);
const topDownPosition = new Vector3(0, 14.5, 0);
const transitionDuration = 850;

function SceneCameraRig({
  resetKey,
  viewMode,
  onTransitionComplete,
  flattenProgress,
}: {
  resetKey: number;
  viewMode: ChessViewMode;
  onTransitionComplete: (mode: "3d" | "2d") => void;
  flattenProgress: React.MutableRefObject<number>;
}) {
  const controls = useRef<OrbitControlsApi | null>(null);
  const transition = useRef<CameraTransition | null>(null);
  const saved3DView = useRef<{ position: Vector3; target: Vector3 } | null>(null);
  const previousResetKey = useRef(resetKey);
  const pendingReset = useRef(false);
  const onCompleteRef = useRef(onTransitionComplete);
  const { camera, clock } = useThree();

  onCompleteRef.current = onTransitionComplete;

  useEffect(() => {
    const activeControls = controls.current;
    if (!activeControls) return;

    if (viewMode === "transitioning-to-2d") {
      saved3DView.current = {
        position: camera.position.clone(),
        target: activeControls.target.clone(),
      };
      transition.current = {
        startedAt: clock.elapsedTime,
        duration: transitionDuration / 1000,
        fromPosition: camera.position.clone(),
        toPosition: topDownPosition.clone(),
        fromTarget: activeControls.target.clone(),
        toTarget: boardCenter.clone(),
        toMode: "2d",
      };
    } else if (viewMode === "transitioning-to-3d") {
      const returnView = saved3DView.current ?? {
        position: default3DPosition.clone(),
        target: boardCenter.clone(),
      };
      transition.current = {
        startedAt: clock.elapsedTime,
        duration: transitionDuration / 1000,
        fromPosition: camera.position.clone(),
        toPosition: returnView.position.clone(),
        fromTarget: activeControls.target.clone(),
        toTarget: returnView.target.clone(),
        toMode: "3d",
      };
    } else {
      transition.current = null;
      flattenProgress.current = viewMode === "2d" ? 1 : 0;
    }
  }, [camera, clock, flattenProgress, viewMode]);

  useEffect(() => {
    if (previousResetKey.current === resetKey && !pendingReset.current) return;
    previousResetKey.current = resetKey;

    if (viewMode === "transitioning-to-2d" || viewMode === "transitioning-to-3d") {
      pendingReset.current = true;
      return;
    }
    pendingReset.current = false;

    const activeControls = controls.current;
    if (!activeControls) return;

    if (viewMode === "2d") {
      camera.position.copy(topDownPosition);
      activeControls.target.copy(boardCenter);
      camera.lookAt(boardCenter);
      activeControls.update();
      saved3DView.current = {
        position: default3DPosition.clone(),
        target: boardCenter.clone(),
      };
    } else if (viewMode === "3d") {
      activeControls.reset();
      saved3DView.current = null;
    }
  }, [camera, resetKey, viewMode]);

  useFrame(() => {
    const activeTransition = transition.current;
    const activeControls = controls.current;
    if (!activeTransition || !activeControls) return;

    const progress = Math.min(
      (clock.elapsedTime - activeTransition.startedAt) / activeTransition.duration,
      1,
    );
    const eased = progress < 0.5
      ? 4 * progress * progress * progress
      : 1 - Math.pow(-2 * progress + 2, 3) / 2;

    camera.position.lerpVectors(
      activeTransition.fromPosition,
      activeTransition.toPosition,
      eased,
    );
    activeControls.target.lerpVectors(
      activeTransition.fromTarget,
      activeTransition.toTarget,
      eased,
    );
    activeControls.update();
    flattenProgress.current = activeTransition.toMode === "2d" ? eased : 1 - eased;

    if (progress >= 1) {
      camera.position.copy(activeTransition.toPosition);
      activeControls.target.copy(activeTransition.toTarget);
      activeControls.update();
      flattenProgress.current = activeTransition.toMode === "2d" ? 1 : 0;
      transition.current = null;
      onCompleteRef.current(activeTransition.toMode);
    }
  });

  const transitioning = viewMode === "transitioning-to-2d" || viewMode === "transitioning-to-3d";
  const is2D = viewMode === "2d";

  return (
    <OrbitControls
      ref={(value) => { controls.current = value as unknown as OrbitControlsApi | null; }}
      makeDefault
      enableDamping={!transitioning}
      dampingFactor={0.08}
      enabled={!transitioning}
      enableRotate={!is2D}
      enablePan={false}
      minDistance={is2D ? 11 : 8.1}
      maxDistance={18}
      minPolarAngle={viewMode === "3d" ? 0.52 : 0.001}
      maxPolarAngle={is2D ? 0.001 : viewMode === "3d" ? 1.28 : Math.PI}
    />
  );
}

export function ChessScene(props: ChessSceneProps) {
  const clickRef = useRef(props.onSquareClick);
  const viewModeRef = useRef(props.viewMode);
  const flattenProgress = useRef(0);
  clickRef.current = props.onSquareClick;
  viewModeRef.current = props.viewMode;

  const handleSquareClick = useCallback((square: Square) => {
    const mode = viewModeRef.current;
    if (mode === "transitioning-to-2d" || mode === "transitioning-to-3d") return;
    clickRef.current(square);
  }, []);

  const { resetCameraKey, viewMode, onTransitionComplete, ...boardProps } = props;

  return (
    <Canvas className="chess-canvas" shadows dpr={[1, 1.5]} gl={{ antialias: true, alpha: false }}>
      <color attach="background" args={["#484a43"]} />
      <PerspectiveCamera makeDefault position={[6.2, 9.4, 10.8]} fov={39} near={0.1} far={60} />
      <SceneCameraRig
        resetKey={resetCameraKey}
        viewMode={viewMode}
        onTransitionComplete={onTransitionComplete}
        flattenProgress={flattenProgress}
      />
      <hemisphereLight args={["#fff8e9", "#555044", 1.5]} />
      <ambientLight intensity={0.28} />
      <directionalLight position={[-5, 10, 6]} intensity={2.2} castShadow shadow-mapSize-width={1536} shadow-mapSize-height={1536} shadow-camera-left={-7} shadow-camera-right={7} shadow-camera-top={7} shadow-camera-bottom={-7} />
      <directionalLight position={[5, 5, -5]} intensity={0.55} color="#ead8b4" />
      <mesh position={[0, -0.55, 0]} receiveShadow>
        <boxGeometry args={[200, 0.25, 200]} />
        <meshStandardMaterial color="#45453e" roughness={0.91} />
      </mesh>
      <Suspense fallback={null}>
        <ChessBoard3D
          {...boardProps}
          flattenProgress={flattenProgress}
          onSquareClick={handleSquareClick}
        />
      </Suspense>
    </Canvas>
  );
}
