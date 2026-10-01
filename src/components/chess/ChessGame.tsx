"use client";

import dynamic from "next/dynamic";
import Link from "next/link";
import { useState } from "react";
import type { Square } from "chess.js";
import {
  ArrowDownUp,
  Camera,
  Crown,
  Dna,
  LayoutGrid,
  RotateCcw,
  Shield,
  Undo2,
} from "lucide-react";
import { AccessibleBoard } from "@/components/chess/AccessibleBoard";
import { SceneErrorBoundary } from "@/components/chess/SceneErrorBoundary";
import { Button } from "@/components/ui/button";
import { formatClockTime, TIME_CONTROLS } from "@/lib/chess/chess-clock";
import {
  CapturedPiecesPanel,
  GameStatusPanel,
  MoveHistoryPanel,
} from "@/components/game-ui/GamePanels";
import { PromotionDialog } from "@/components/game-ui/PromotionDialog";
import { useChessGame } from "@/hooks/useChessGame";
import type { ChessViewMode, PlayerRole } from "@/types/chess";

const roleLabels: Record<PlayerRole, string> = {
  human: "Human",
  minimax: "Minimax",
  alpha_beta: "Alpha-Beta",
  genetic: "Genetic AI",
};

const ChessScene = dynamic(
  () =>
    import("@/components/chess/ChessScene").then(
      (module) => module.ChessScene,
    ),
  {
    ssr: false,
    loading: () => (
      <div className="scene-loading">
        <div className="text-center">
          <div className="mx-auto mb-3 size-8 animate-spin rounded-full border-2 border-[#d6c194]/35 border-t-[#ead5ad]" />
          <p className="m-0 text-xs tracking-wide">
            Đang chuẩn bị bàn cờ 3D...
          </p>
        </div>
      </div>
    ),
  },
);

export function ChessGame() {
  const game = useChessGame();
  const [flipped, setFlipped] = useState(false);
  const [viewMode, setViewMode] = useState<ChessViewMode>("3d");
  const [cameraReset, setCameraReset] = useState(0);
  const isTransitioning = viewMode === "transitioning-to-2d" || viewMode === "transitioning-to-3d";
  const currently2D = viewMode === "2d" || viewMode === "transitioning-to-3d";
  const handleSquareClick = (square: Square) => {
    if (!isTransitioning) game.handleSquareClick(square);
  };
  const startViewTransition = () => {
    if (viewMode === "3d") setViewMode("transitioning-to-2d");
    else if (viewMode === "2d") setViewMode("transitioning-to-3d");
  };

  const winner =
    game.status === "timeout"
      ? game.timeoutLoser === "w" ? "Black" : "White"
      : game.status === "checkmate"
      ? game.turn === "w"
        ? "Black"
        : "White"
      : null;

  return (
    <main className="mx-auto min-h-screen w-full max-w-[1480px] px-4 pb-8 pt-4 sm:px-6 sm:pt-6 lg:px-9 lg:pb-12">
      <header className="mb-5 flex items-center justify-between border-b border-[#d3ccbd] pb-4 sm:mb-8">
        <a
          href="#main"
          className="group inline-flex items-center gap-3 rounded-md focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-amber-700"
        >
          <span className="grid size-10 place-items-center rounded-xl border border-[#a48b5b] bg-[#4c583f] text-[#f2e7cf] shadow-[0_2px_0_#fff8_inset]">
            <Crown size={19} strokeWidth={1.8} />
          </span>

          <span>
            <span className="block font-display text-sm font-bold tracking-[.12em] text-[#30352e]">
              CHESS <span className="text-[#9a7040]">TABLE</span>
            </span>

            <span className="mt-0.5 block font-mono text-[9px] tracking-[.12em] text-[#7c796d]">
              DIGITAL CHESS ROOM
            </span>
          </span>
        </a>

        <div className="flex items-center gap-2">
          <Button asChild variant="secondary" size="sm" className="h-9 text-[11px]">
            <Link href="/laboratory/genetic">
              <Dna size={14} />
              Genetic Algorithm Lab
            </Link>
          </Button>
          <div className="hidden items-center gap-2 rounded-full border border-[#d4cbb8] bg-[#faf8f1] px-3 py-1.5 text-[10px] font-medium tracking-[.06em] text-[#626258] sm:flex">
            <span className="size-1.5 rounded-full bg-[#81905c]" />
            LOCAL PASS &amp; PLAY
          </div>
        </div>
      </header>

      <div
        id="main"
        className="mb-5 flex flex-col justify-between gap-3 sm:mb-6 sm:flex-row sm:items-end"
      >
        <div>
          <p className="mb-1 mt-0 font-mono text-[9px] font-medium tracking-[.18em] text-[#987442]">
            A TABLE FOR TWO | EST. 2026
          </p>

          <h1 className="m-0 font-display text-[clamp(1.7rem,4vw,2.5rem)] font-semibold tracking-[-.055em] text-[#292d29]">
            Sit down. Make your move.
          </h1>

          <p className="mb-0 mt-1.5 text-xs text-[#6e6e64]">
            A 3D chess table for two players.
          </p>
        </div>

        <div className="flex items-center gap-2 self-start rounded-lg border border-[#d4cbb8] bg-[#fbf9f3] px-3 py-2 text-[10px] font-mono tracking-[.04em] text-[#68675c] sm:self-auto">
          <Shield size={14} className="text-[#897246]" />
          CHƠI 2 NGƯỜI | CÙNG THIẾT BỊ
        </div>
      </div>

      <div className="grid min-w-0 grid-cols-1 gap-4 lg:grid-cols-[minmax(0,1.55fr)_minmax(310px,.75fr)] lg:gap-5">
        <section
          className="min-w-0 rounded-[22px] border border-[#554b3e] bg-[#41423c] p-2.5 shadow-[0_18px_48px_#34302620,0_1px_0_#fff3_inset] sm:rounded-[26px] sm:p-4"
          aria-label="Bàn cờ"
        >
          <PlayerRail
            name="Quân đen"
            descriptor={`${roleLabels[game.playerRoles.b]} · ĐI SAU`}
            active={
              game.turn === "b" &&
              game.status !== "checkmate" &&
              game.status !== "stalemate" &&
              game.status !== "draw" &&
              game.status !== "timeout"
            }
            captured={game.captured.b}
            side="b"
            clockText={formatClockTime(game.clockTimes.b, game.timeControl)}
            clockActive={game.clockActiveColor === "b"}
            clockWarning={game.clockTimes.b < 60_000}
            clockCritical={game.clockTimes.b < 10_000}
          />

          <div className="relative mx-auto aspect-square w-full max-w-[740px] overflow-hidden rounded-2xl border border-[#706758] bg-[#484a43] shadow-[0_8px_24px_#17181355]">
            <SceneErrorBoundary>
              <ChessScene
                board={game.board}
                selectedSquare={game.selectedSquare}
                legalDestinations={game.legalDestinations}
                lastMove={game.lastMove}
                checkedKingSquare={game.checkedKingSquare}
                animation={game.animation}
                flipped={flipped}
                resetCameraKey={cameraReset}
                viewMode={viewMode}
                onTransitionComplete={setViewMode}
                onSquareClick={handleSquareClick}
              />
            </SceneErrorBoundary>

            {winner && (
              <div className="pointer-events-none absolute inset-x-0 bottom-4 flex justify-center px-3">
                <div className="pointer-events-auto flex items-center gap-3 rounded-xl border border-[#ddc99f] bg-[#fffdf4]/95 px-4 py-3 shadow-xl backdrop-blur">
                  <span className="grid size-8 place-items-center rounded-lg bg-[#edf1e5] text-[#52653d]">
                    <Crown size={16} />
                  </span>

                  <div>
                    <p className="m-0 text-[9px] font-semibold tracking-[.12em] text-[#8a6935]">
                      {game.status === "timeout" ? "TIME OUT" : "CHIẾU HẾT"}
                    </p>

                    <p className="m-0 text-sm font-semibold text-[#30352e]">
                      {winner} thắng ván này
                    </p>
                  </div>

                  <Button size="sm" onClick={game.newGame}>
                    Ván mới
                  </Button>
                </div>
              </div>
            )}
          </div>

          <PlayerRail
            name="Quân trắng"
            descriptor={`${roleLabels[game.playerRoles.w]} · ĐI TRƯỚC`}
            active={
              game.turn === "w" &&
              game.status !== "checkmate" &&
              game.status !== "stalemate" &&
              game.status !== "draw" &&
              game.status !== "timeout"
            }
            captured={game.captured.w}
            side="w"
            clockText={formatClockTime(game.clockTimes.w, game.timeControl)}
            clockActive={game.clockActiveColor === "w"}
            clockWarning={game.clockTimes.w < 60_000}
            clockCritical={game.clockTimes.w < 10_000}
          />
        </section>

        <aside className="flex min-w-0 flex-col gap-3.5">
          <section className="grid grid-cols-2 gap-3 rounded-2xl border border-[#d5cebf] bg-[#fffdf8] p-4">
            {(["w", "b"] as const).map((color) => (
              <label key={color} className="flex min-w-0 flex-col gap-1.5">
                <span className="text-[9px] font-semibold tracking-[.14em] text-[#77776d]">
                  {color === "w" ? "WHITE" : "BLACK"}
                </span>
                <select
                  aria-label={`${color === "w" ? "White" : "Black"} player role`}
                  value={game.playerRoles[color]}
                  onChange={(event) => game.setPlayerRole(color, event.target.value as PlayerRole)}
                  className="h-9 w-full rounded-lg border border-[#ded7c7] bg-[#f7f4ec] px-2 text-xs text-[#343930] outline-none focus-visible:ring-2 focus-visible:ring-amber-700"
                >
                  {Object.entries(roleLabels)
                    .filter(([role]) => role !== "genetic" || game.geneticProfile !== null)
                    .map(([role, label]) => (
                      <option key={role} value={role}>
                        {role === "genetic" && game.geneticProfile
                          ? `Genetic AI · ${game.geneticProfile.candidate_id}`
                          : label}
                      </option>
                    ))}
                </select>
              </label>
            ))}
            {game.geneticProfile && (
              <p className="col-span-2 m-0 border-t border-[#e4dece] pt-2 text-[10px] leading-4 text-[#69695f]">
                Ready to play: <strong>{game.geneticProfile.candidate_id}</strong> · fitness{" "}
                <strong>{game.geneticProfile.fitness > 0 ? "+" : ""}{game.geneticProfile.fitness}</strong>
                {Object.values(game.playerRoles).includes("genetic") ? " · selected as a player" : " · choose it for White or Black"}
              </p>
            )}
          </section>

          <section className="rounded-2xl border border-[#d5cebf] bg-[#fffdf8] p-4" aria-labelledby="time-control-heading">
            <div className="flex items-baseline justify-between gap-3">
              <div>
                <p className="m-0 text-[9px] font-semibold tracking-[.14em] text-[#77776d]">GAME SETTINGS</p>
                <h2 id="time-control-heading" className="m-0 mt-1 font-display text-sm font-semibold">Time control</h2>
              </div>
              <span className="text-[9px] text-[#77776d]">per side · whole game</span>
            </div>
            <div className="mt-3 grid grid-cols-3 gap-2">
              {TIME_CONTROLS.map((control) => (
                <button
                  key={control.id}
                  type="button"
                  aria-pressed={game.timeControl === control.id}
                  onClick={() => game.chooseTimeControl(control.id)}
                  className={`min-h-9 rounded-lg border px-2 text-xs font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-amber-700 ${game.timeControl === control.id ? "border-[#68764d] bg-[#edf1e5] text-[#465238]" : "border-[#ded7c7] bg-[#f7f4ec] text-[#626258] hover:bg-white"}`}
                >
                  {control.label}
                </button>
              ))}
            </div>
            <p className="mb-0 mt-2 text-[10px] leading-4 text-[#77776d]" title="Search depth and time control are separate settings.">
              Search depth controls how far AI looks ahead; time control sets each side’s whole-game clock. AI search counts against its clock. Changing this starts a new game.
            </p>
          </section>

          <GameStatusPanel
            turn={game.turn}
            status={game.status}
            moveNumber={game.moveNumber}
            aiThinking={game.aiThinking}
            timeoutLoser={game.timeoutLoser}
          />

          <div className="grid grid-cols-2 gap-2 rounded-2xl border border-[#d5cebf] bg-[#f7f4ec] p-2">
            <Button onClick={game.newGame} className="h-10">
              <RotateCcw size={15} />
              Ván mới
            </Button>

            <Button
              variant="secondary"
              onClick={game.undo}
              disabled={!game.moveCount || game.status === "timeout"}
              className="h-10"
            >
              <Undo2 size={15} />
              Đi lại
            </Button>

            <Button
              variant="secondary"
              onClick={() => setFlipped((value) => !value)}
              className="h-9 text-xs"
            >
              <ArrowDownUp size={14} />
              Xoay bàn
            </Button>

            <Button
              variant="secondary"
              onClick={() => setCameraReset((value) => value + 1)}
              className="h-9 text-xs"
            >
              <Camera size={14} />
              Đặt lại góc nhìn
            </Button>

            <Button
              variant="secondary"
              onClick={startViewTransition}
              disabled={isTransitioning}
              aria-pressed={currently2D}
              className="col-span-2 h-9 text-xs"
            >
              <LayoutGrid size={14} />
              {currently2D ? "Chuyển sang bàn 3D" : "Chuyển sang bàn 2D"}
            </Button>
          </div>

          <CapturedPiecesPanel captured={game.captured} />

          <MoveHistoryPanel history={game.history} />
        </aside>
      </div>

      <AccessibleBoard
        board={game.board}
        selected={game.selectedSquare}
        legalDestinations={game.legalDestinations}
        lastMove={game.lastMove}
        onSquareClick={handleSquareClick}
      />

      <footer className="mt-6 flex flex-col justify-between gap-2 border-t border-[#d3ccbd] pt-3 text-[9px] tracking-[.04em] text-[#878579] sm:flex-row">
        <span>LOCAL PLAY | STANDARD RULES | NO ACCOUNT</span>
        <span>WOOD BOARD | 3D CHESS PIECES</span>
      </footer>

      <PromotionDialog
        open={Boolean(game.promotionRequest)}
        color={game.promotionRequest?.color ?? "w"}
        onOpenChange={(open) => {
          if (!open) game.cancelPromotion();
        }}
        onChoose={game.choosePromotion}
      />
    </main>
  );
}

function PlayerRail({
  name,
  descriptor,
  active,
  captured,
  side,
  clockText,
  clockActive,
  clockWarning,
  clockCritical,
}: {
  name: string;
  descriptor: string;
  active: boolean;
  captured: string[];
  side: "w" | "b";
  clockText: string;
  clockActive: boolean;
  clockWarning: boolean;
  clockCritical: boolean;
}) {
  return (
    <div
      className={`flex min-h-[55px] items-center gap-3 px-1.5 py-2 sm:px-2 ${
        side === "b" ? "pt-1" : "pb-1"
      }`}
    >
      <span
        className={`grid size-9 shrink-0 place-items-center rounded-full border ${
          side === "w"
            ? "border-[#d1c9b7] bg-[#f3eddd] text-[#505347]"
            : "border-[#747368] bg-[#353934] text-[#e9e3d4]"
        }`}
      >
        <Crown size={15} strokeWidth={1.7} />
      </span>

      <div className="min-w-[82px]">
        <p className="m-0 text-xs font-semibold text-[#f4f0e5]">
          {name}
        </p>

        <p className="mb-0 mt-0.5 font-mono text-[8px] tracking-[.12em] text-[#b9b4a7]">
          {descriptor}
        </p>
      </div>

      <span
        className="flex flex-1 flex-wrap items-center gap-0.5 text-base leading-none text-[#ded7c6]"
        aria-label={`${name} đã bắt ${captured.length} quân`}
      >
        {captured.map((piece, index) => (
          <CapturedSymbol
            key={`${piece}-${index}`}
            piece={piece}
            capturedBy={side}
          />
        ))}
      </span>

      <span
        role="timer"
        aria-label={`${side === "w" ? "White" : "Black"} clock ${clockText}${clockActive ? ", active" : ""}`}
        className={`min-w-[76px] rounded-md border px-2.5 py-1.5 text-right font-mono text-sm font-semibold tabular-nums ${clockCritical ? "border-[#e5a497] bg-[#8d4438] text-white" : clockWarning ? "border-[#e0ba8c] bg-[#8b6941] text-white" : clockActive ? "border-[#c5d0a5] bg-[#69764f] text-white" : "border-[#77776d] bg-[#353934] text-[#f3eddd]"}`}
      >
        {clockText}
      </span>

      {active && (
        <span className="flex items-center gap-1.5 rounded-full border border-[#929b78] bg-[#77825f]/20 px-2.5 py-1 text-[8px] font-medium tracking-[.1em] text-[#dfe8c9]">
          <span className="size-1.5 animate-pulse rounded-full bg-[#c8d991]" />
          ĐANG ĐI
        </span>
      )}
    </div>
  );
}

function CapturedSymbol({
  piece,
  capturedBy,
}: {
  piece: string;
  capturedBy: "w" | "b";
}) {
  const whiteSymbols: Record<string, string> = {
    p: "\u2659",
    n: "\u2658",
    b: "\u2657",
    r: "\u2656",
    q: "\u2655",
    k: "\u2654",
  };

  const blackSymbols: Record<string, string> = {
    p: "\u265f",
    n: "\u265e",
    b: "\u265d",
    r: "\u265c",
    q: "\u265b",
    k: "\u265a",
  };

  const symbols = capturedBy === "w" ? blackSymbols : whiteSymbols;

  return (
    <span
      className={
        capturedBy === "w"
          ? "chess-symbol text-[#191d19]"
          : "chess-symbol text-[#fff8e8] [text-shadow:0_1px_1px_#252921]"
      }
      aria-hidden="true"
    >
      {symbols[piece]}
    </span>
  );
}
