import * as React from "react";
import { Slot } from "@radix-ui/react-slot";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const variants = cva("inline-flex items-center justify-center gap-2 rounded-lg font-medium transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-amber-700 focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-45", {
  variants: {
    variant: {
      default: "bg-[#48563c] text-white shadow-sm hover:bg-[#37452e]",
      secondary: "border border-[#cfc6b3] bg-[#fbf9f3] text-[#3c4038] hover:bg-[#f1ede3]",
      ghost: "text-[#606057] hover:bg-[#ece8de] hover:text-[#292d29]",
      outline: "border border-[#cfc6b3] bg-transparent text-[#3c4038] hover:bg-[#f1ede3]",
      copper: "bg-[#a87643] text-white hover:bg-[#91663b]",
    },
    size: {
      default: "h-10 px-4 text-sm",
      sm: "h-9 px-3 text-xs",
      icon: "size-10 p-0",
    },
  },
  defaultVariants: { variant: "default", size: "default" },
});

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement>, VariantProps<typeof variants> {
  asChild?: boolean;
}

export function Button({ className, variant, size, asChild = false, ...props }: ButtonProps) {
  const Component = asChild ? Slot : "button";
  return <Component className={cn(variants({ variant, size }), className)} {...props} />;
}
