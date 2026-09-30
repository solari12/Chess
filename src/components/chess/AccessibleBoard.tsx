"use client";

import { useRef, useState, type KeyboardEvent } from "react";
import type { Piece, Square } from "chess.js";

const symbols: Record<string, string> = {
  p: "\u265f",
  n: "\u265e",
  b: "\u265d",
  r: "\u265c",
  q: "\u265b",
  k: "\u265a",
};
const names: Record<string, string> = {
  p: "tốt",
  n: "mã",
  b: "tượng",
  r: "xe",
  q: "hậu",
  k: "vua",
};

export function AccessibleBoard({
  board,
  selected,
  legalDestinations,
  lastMove,
  onSquareClick,
}: {
  board: (Piece | null)[][];
  selected: Square | null;
  legalDestinations: Square[];
  lastMove: { from: Square; to: Square } | null;
  onSquareClick: (square: Square) => void;
}) {
  const [focusedSquare, setFocusedSquare] = useState<Square>("e2");
  const squareRefs = useRef(new Map<Square, HTMLButtonElement>());
  const files = "abcdefgh";
  const legal = new Set(legalDestinations);

  const moveFocus = (event: KeyboardEvent<HTMLButtonElement>, square: Square) => {
    const file = square.charCodeAt(0) - 97;
    const rank = Number(square[1]);
    const nextFile = Math.max(
      0,
      Math.min(
        7,
        file +
          (event.key === "ArrowLeft" ? -1 : event.key === "ArrowRight" ? 1 : 0),
      ),
    );
    const nextRank = Math.max(
      1,
      Math.min(
        8,
        rank +
          (event.key === "ArrowUp" ? 1 : event.key === "ArrowDown" ? -1 : 0),
      ),
    );

    if (["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].includes(event.key)) {
      event.preventDefault();
      const nextSquare = `${files[nextFile]}${nextRank}` as Square;
      setFocusedSquare(nextSquare);
      squareRefs.current.get(nextSquare)?.focus();
    }
  };

  return (
    <section
      aria-label="Bàn cờ bàn phím & trợ năng"
      className="sr-only focus-within:not-sr-only focus-within:fixed focus-within:inset-4 focus-within:z-50 focus-within:overflow-auto focus-within:rounded-xl focus-within:border focus-within:border-[#d5cebf] focus-within:bg-[#fffdf8] focus-within:p-4 focus-within:shadow-2xl"
    >
      <h2 className="mb-1 mt-0 text-base font-semibold text-[#30352e]">
        Bàn cờ bàn phím &amp; trợ năng
      </h2>
      <p id="accessible-board-help" className="mb-3 mt-0 text-sm text-[#626258]">
        Dùng phím mũi tên để di chuyển, Enter để chọn quân hoặc ô. Tab để vào bàn cờ.
      </p>
      <div
        role="grid"
        aria-label="Bàn cờ vua"
        aria-describedby="accessible-board-help"
        aria-rowcount={8}
        aria-colcount={8}
        className="mx-auto grid aspect-square w-full max-w-[440px] grid-cols-8 grid-rows-8 overflow-hidden rounded-md border border-[#9d927b]"
      >
        {Array.from({ length: 8 }, (_, row) => (
          <div role="row" key={row} className="contents">
            {Array.from(files).map((file, col) => {
              const square = `${file}${8 - row}` as Square;
              const piece = board[row][col];
              const isLegal = legal.has(square);
              const isSelected = selected === square;
              const isLastMove = lastMove?.from === square || lastMove?.to === square;
              const description = piece
                ? `${piece.color === "w" ? "Trắng" : "Đen"} ${names[piece.type]}`
                : "ô trống";

              return (
                <button
                  key={square}
                  ref={(element) => {
                    if (element) squareRefs.current.set(square, element);
                    else squareRefs.current.delete(square);
                  }}
                  type="button"
                  role="gridcell"
                  tabIndex={focusedSquare === square ? 0 : -1}
                  aria-label={`${square}, ${description}${isSelected ? ", đang được chọn" : ""}${isLegal ? ", nước đi hợp lệ" : ""}${isLastMove ? ", thuộc nước đi gần nhất" : ""}`}
                  aria-selected={isSelected}
                  onFocus={() => setFocusedSquare(square)}
                  onKeyDown={(event) => moveFocus(event, square)}
                  onClick={() => onSquareClick(square)}
                  className={`relative grid h-full min-h-0 min-w-0 place-items-center text-[clamp(1.15rem,6vw,2.2rem)] leading-none focus-visible:z-10 focus-visible:outline-2 focus-visible:outline-offset-[-3px] focus-visible:outline-[#a87643] ${
                    (row + col) % 2 ? "bg-[#74583f]" : "bg-[#e8dfcb]"
                  } ${
                    isLastMove ? "[box-shadow:inset_0_0_0_999px_#c7ad6570]" : ""
                  } ${
                    isSelected ? "[box-shadow:inset_0_0_0_999px_#9aad6b80]" : ""
                  }`}
                >
                  {piece && (
                    <span
                      aria-hidden="true"
                      className={
                        piece.color === "w"
                          ? "chess-symbol text-[#fffdf4] [text-shadow:0_1px_2px_#34382f,0_0_1px_#34382f]"
                          : "chess-symbol text-[#343a35] [text-shadow:0_1px_1px_#c8c3b7]"
                      }
                    >
                      {symbols[piece.type]}
                    </span>
                  )}
                  {isLegal && (
                    <span
                      aria-hidden="true"
                      className={`absolute size-2 rounded-full ${
                        piece
                          ? "inset-1 rounded-none border-2 border-[#fff6] bg-transparent"
                          : "bg-[#526c40]/70"
                      }`}
                    />
                  )}
                </button>
              );
            })}
          </div>
        ))}
      </div>
    </section>
  );
}
