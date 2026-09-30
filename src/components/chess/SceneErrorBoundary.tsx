"use client";

import { Component, type ReactNode } from "react";
import { MonitorX } from "lucide-react";

interface SceneErrorBoundaryProps { children: ReactNode; onError?: (error: Error) => void }
interface SceneErrorBoundaryState { hasError: boolean }

export class SceneErrorBoundary extends Component<SceneErrorBoundaryProps, SceneErrorBoundaryState> {
  state: SceneErrorBoundaryState = { hasError: false };

  static getDerivedStateFromError(): SceneErrorBoundaryState {
    return { hasError: true };
  }

  componentDidCatch(error: Error) {
    this.props.onError?.(error);
  }

  render() {
    if (this.state.hasError) return <div className="grid h-full min-h-[360px] place-items-center rounded-2xl border border-[#82745c] bg-[#3c3d37] p-8 text-center text-[#f5f0e4]">
      <div><MonitorX size={25} className="mx-auto text-[#d4b77f]" /><h2 className="mb-1 mt-3 font-display text-base font-semibold">Khong the mo ban co 3D</h2><p className="m-0 max-w-sm text-xs leading-5 text-[#d1cabb]">Ban van co the tiep tuc van co bang ban co ban phim ben duoi.</p></div>
    </div>;
    return this.props.children;
  }
}
