"use client";

import type { PieceSymbol } from "chess.js";
import { Dialog, DialogContent, DialogDescription, DialogTitle } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";

const symbols: Record<PieceSymbol, string> = { p: "\u265f", n: "\u265e", b: "\u265d", r: "\u265c", q: "\u265b", k: "\u265a" };
const options: { piece: PieceSymbol; label: string }[] = [
  { piece: "q", label: "Queen" }, { piece: "r", label: "Rook" }, { piece: "b", label: "Bishop" }, { piece: "n", label: "Knight" },
];

export function PromotionDialog({ open, color, onOpenChange, onChoose }: { open: boolean; color: "w" | "b"; onOpenChange: (open: boolean) => void; onChoose: (piece: PieceSymbol) => void }) {
  return <Dialog open={open} onOpenChange={onOpenChange}>
    <DialogContent onEscapeKeyDown={() => onOpenChange(false)}>
      <DialogTitle>Phong cap tot</DialogTitle>
      <DialogDescription>Chon quan co se thay the tot cua ban.</DialogDescription>
      <div className="mt-5 grid grid-cols-4 gap-2">
        {options.map(({ piece, label }) => <Button key={piece} variant="secondary" className="h-auto min-h-[88px] flex-col gap-1.5 rounded-xl" aria-label={`Phong cap thanh ${label}`} onClick={() => onChoose(piece)}>
          <span className={`chess-symbol text-4xl leading-none ${color === "w" ? "text-[#fffdf4] [text-shadow:0_1px_2px_#34382f,0_0_1px_#34382f]" : "text-[#343a35] [text-shadow:0_1px_1px_#c8c3b7]"}`} aria-hidden="true">{symbols[piece]}</span>
          <span className="text-xs">{label}</span>
        </Button>)}
      </div>
    </DialogContent>
  </Dialog>;
}
