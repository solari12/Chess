"use client";

import { useEffect, useRef } from "react";
import { Activity, Check, CircleDot, Clock3, Trophy } from "lucide-react";
import type { Move, PieceSymbol } from "chess.js";
import type { CapturedPieces, GameStatus } from "@/types/chess";

const symbols: Record<PieceSymbol, string> = {
  p: "\u265f",
  n: "\u265e",
  b: "\u265d",
  r: "\u265c",
  q: "\u265b",
  k: "\u265a",
};

const pieceNames: Record<PieceSymbol, string> = {
  p: "Tốt",
  n: "Mã",
  b: "Tượng",
  r: "Xe",
  q: "Hậu",
  k: "Vua",
};

export function GameStatusPanel({
  turn,
  status,
  moveNumber,
}: {
  turn: "w" | "b";
  status: GameStatus;
  moveNumber: number;
}) {
  const title =
    status === "checkmate"
      ? "Chiếu hết"
      : status === "stalemate"
        ? "Hòa cờ"
        : status === "draw"
          ? "Hòa cờ"
          : status === "check"
            ? "Đang bị chiếu"
            : "Đang chơi";

  const detail =
    status === "checkmate"
      ? `${turn === "w" ? "Đen" : "Trắng"} thắng ván này.`
      : status === "stalemate" || status === "draw"
        ? "Ván cờ kết thúc với kết quả hòa."
        : status === "check"
          ? `${turn === "w" ? "Trắng" : "Đen"} phải bảo vệ vua.`
          : "Chọn một quân cờ để xem các nước đi hợp lệ.";

  const Icon =
    status === "checkmate"
      ? Trophy
      : status === "check"
        ? Activity
        : status === "draw" || status === "stalemate"
          ? CircleDot
          : Check;

  return (
    <section
      className="rounded-2xl border border-[#d5cebf] bg-[#fffdf8] p-4 shadow-[0_3px_10px_#3029190a]"
      aria-live="polite"
      aria-atomic="true"
    >
      <div className="flex items-start gap-3">
        <span
          className={`grid size-10 shrink-0 place-items-center rounded-xl border ${
            status === "check" || status === "checkmate"
              ? "border-[#d8a094] bg-[#f7eae6] text-[#a94f43]"
              : "border-[#cfd7bd] bg-[#edf1e5] text-[#536743]"
          }`}
        >
          <Icon size={18} aria-hidden="true" />
        </span>

        <div className="min-w-0 flex-1">
          <div className="flex items-center justify-between gap-3">
            <p className="m-0 text-[10px] font-semibold tracking-[.14em] text-[#77776d]">
              {status === "checkmate" ||
              status === "stalemate" ||
              status === "draw"
                ? "KẾT QUẢ VÁN ĐẤU"
                : "LƯỢT ĐI HIỆN TẠI"}
            </p>

            <span className="inline-flex items-center gap-1.5 rounded-full bg-[#f0ede4] px-2 py-1 text-[9px] font-medium text-[#706f65]">
              <Clock3 size={11} />
              NƯỚC {moveNumber}
            </span>
          </div>

          <div className="mt-1 flex items-center gap-2">
            <span
              className={`size-3 rounded-full border ${
                turn === "w"
                  ? "border-[#a39a84] bg-[#fffdf4]"
                  : "border-[#252a25] bg-[#3d433d]"
              }`}
              aria-hidden="true"
            />

            <h2 className="m-0 font-display text-lg font-semibold leading-tight text-[#292d29]">
              {title}
            </h2>
          </div>

          <p className="mb-0 mt-1 text-xs leading-5 text-[#6d6d64]">
            {detail}
          </p>
        </div>
      </div>
    </section>
  );
}

function PieceTray({
  title,
  pieces,
  capturedBy,
}: {
  title: string;
  pieces: PieceSymbol[];
  capturedBy: "w" | "b";
}) {
  const order: PieceSymbol[] = ["q", "r", "b", "n", "p"];

  const ordered = order.flatMap((type) =>
    pieces.filter((piece) => piece === type),
  );

  return (
    <div className="flex min-h-10 items-center gap-2 border-b border-[#e9e4d8] py-2 last:border-b-0">
      <span className="w-[104px] shrink-0 text-[10px] text-[#737267]">
        {title}
      </span>

      <span
        className={`flex min-h-6 flex-wrap items-center gap-0.5 text-xl leading-none ${
          capturedBy === "w" ? "text-[#42463e]" : "text-[#a59d8d]"
        }`}
        aria-label="Các quân cờ đã bị bắt"
      >
        {ordered.length ? (
          ordered.map((piece, index) => (
            <span
              key={`${piece}-${index}`}
              title={pieceNames[piece]}
              aria-hidden="true"
            >
              {symbols[piece]}
            </span>
          ))
        ) : (
          <span className="text-xs text-[#9a988d]">?</span>
        )}
      </span>

      <span className="ml-auto font-mono text-[10px] tabular-nums text-[#878579]">
        {ordered.length || 0}
      </span>
    </div>
  );
}

export function CapturedPiecesPanel({
  captured,
}: {
  captured: CapturedPieces;
}) {
  return (
    <section
      className="rounded-2xl border border-[#d5cebf] bg-[#fffdf8] px-4 py-2"
      aria-label="Các quân cờ đã bị bắt"
    >
      <PieceTray
        title="Trắng đã bắt"
        pieces={captured.w}
        capturedBy="w"
      />

      <PieceTray
        title="Đen đã bắt"
        pieces={captured.b}
        capturedBy="b"
      />
    </section>
  );
}

export function MoveHistoryPanel({ history }: { history: Move[] }) {
  const bottom = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottom.current?.scrollIntoView({
      block: "nearest",
      behavior: "smooth",
    });
  }, [history.length]);

  const rows = Array.from(
    { length: Math.ceil(history.length / 2) },
    (_, index) => ({
      number: index + 1,
      white: history[index * 2],
      black: history[index * 2 + 1],
    }),
  );

  return (
    <section
      className="flex min-h-[260px] flex-col overflow-hidden rounded-2xl border border-[#d5cebf] bg-[#fffdf8]"
      aria-labelledby="move-history-heading"
    >
      <header className="flex items-center justify-between border-b border-[#e9e4d8] px-4 py-3">
        <div>
          <p className="m-0 text-[9px] font-semibold tracking-[.14em] text-[#898677]">
            BIÊN BẢN NƯỚC ĐI
          </p>

          <h2
            id="move-history-heading"
            className="m-0 mt-0.5 font-display text-sm font-semibold"
          >
            Biên bản nước đi
          </h2>
        </div>

        <span className="rounded-md border border-[#ded7c7] bg-[#f5f2e9] px-2 py-1 font-mono text-[10px] text-[#666459]">
          {history.length} nước
        </span>
      </header>

      <div
        className="move-scroll min-h-0 flex-1 overflow-y-auto p-2"
        role="log"
        aria-label="Biên bản nước đi"
      >
        {rows.length ? (
          <div className="grid grid-cols-[36px_1fr_1fr]">
            {rows.map((row) => (
              <div className="contents" key={row.number}>
                <span className="border-b border-[#eee9df] px-2 py-2 font-mono text-[10px] text-[#969184]">
                  {row.number}.
                </span>

                <span
                  className={`border-b border-[#eee9df] px-2 py-2 font-mono text-xs ${
                    row.white?.color === "w"
                      ? "text-[#343930]"
                      : "text-transparent"
                  }`}
                >
                  {row.white?.san ?? ""}
                </span>

                <span className="border-b border-[#eee9df] px-2 py-2 font-mono text-xs text-[#5d6259]">
                  {row.black?.san ?? ""}
                </span>
              </div>
            ))}
          </div>
        ) : (
          <div className="grid min-h-[190px] place-items-center px-4 text-center">
            <div>
              <span className="mx-auto grid size-9 place-items-center rounded-full border border-[#ded7c7] bg-[#f7f4eb] text-[#8a774f]">
                <CircleDot size={16} />
              </span>

              <p className="mb-0 mt-2 text-xs font-medium text-[#56594f]">
                Biên bản đang chờ nước đầu tiên
              </p>

              <p className="mb-0 mt-1 text-[10px] text-[#878579]">
                Chọn một quân trắng để bắt đầu.
              </p>
            </div>
          </div>
        )}

        <div ref={bottom} />
      </div>
    </section>
  );
}