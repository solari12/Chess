import { createElement } from "react";
import { renderToString } from "react-dom/server";
import { afterEach, describe, expect, it, vi } from "vitest";
import GeneticLaboratoryPage from "@/app/laboratory/genetic/page";
import {
  getGeneticState,
  resetGeneticExperiment,
  startGeneticExperiment,
  stepGeneticExperiment,
  streamGeneticExperiment,
  type GeneticConfig,
  type GeneticState,
} from "@/lib/laboratory/genetic-client";

vi.mock("next/navigation", () => ({ useRouter: () => ({ push: () => undefined }) }));

const readyState: GeneticState = {
  status: "ready",
  generation: 0,
  generation_limit: 0,
  config: null,
  population: [],
  history: [],
  game_traces: [],
};

describe("genetic laboratory page and client", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("renders the laboratory controls and educational panels without starting training", () => {
    const markup = renderToString(createElement(GeneticLaboratoryPage));

    expect(markup).toContain("Genetic Algorithm Laboratory");
    expect(markup).toContain("Population");
    expect(markup).toContain("Evolution visualizer");
    expect(markup).toContain("What happened?");
    expect(markup).toContain("BASELINE is one reference opponent");
    expect(markup).toContain("not a universal measure of chess strength");
    expect(markup).toContain("Start Experiment");
    expect(markup).not.toContain("Evaluating self-play games");
  });

  it("calls the separate state, start, step, and reset laboratory endpoints", async () => {
    const fetchMock = vi.fn().mockImplementation(() =>
      Promise.resolve(new Response(JSON.stringify(readyState), { status: 200 })),
    );
    vi.stubGlobal("fetch", fetchMock);

    await expect(getGeneticState()).resolves.toEqual(readyState);
    const config: GeneticConfig = {
      population_size: 10,
      games_per_individual: 8,
      search_depth: 2,
      mutation_rate: 0.15,
      mutation_strength: 25,
      elite_count: 2,
      tournament_size: 3,
      max_plies: 200,
      generations: 20,
      seed: 42,
    };
    await startGeneticExperiment(config);
    await stepGeneticExperiment();
    await resetGeneticExperiment();

    expect(fetchMock.mock.calls.map(([url]) => url)).toEqual([
      "http://127.0.0.1:8000/api/lab/genetic/state",
      "http://127.0.0.1:8000/api/lab/genetic/start",
      "http://127.0.0.1:8000/api/lab/genetic/step",
      "http://127.0.0.1:8000/api/lab/genetic/reset",
    ]);
  });

  it("consumes ordered real-time events and requires a completed state event", async () => {
    const streamBody = [
      { type: "evaluation_started", generation: 1, candidates_total: 1, completed_candidates: 0, games_per_individual: 8 },
      { type: "game_started", generation: 1, candidate_id: "I0001", candidate_weights: { pawn: 100, knight: 320, bishop: 330, rook: 500, queen: 900 }, candidate_color: "white", color: "white", game_index: 1, games_total: 8, opponent_id: "BASELINE", opponent_type: "baseline", opponent_weights: { pawn: 100, knight: 320, bishop: 330, rook: 500, queen: 900 }, baseline_weights: { pawn: 100, knight: 320, bishop: 330, rook: 500, queen: 900 }, candidate_index: 1, candidates_total: 1, completed_candidates: 0 },
      { type: "move_played", generation: 1, candidate_id: "I0001", candidate_color: "white", opponent_id: "BASELINE", opponent_type: "baseline", game_index: 1, ply: 1, move_number: 1, color: "white", move: "e4", san: "e4", uci: "e2e4", fen: "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1", evaluation: 0.1, nodes: 1, depth: 2, candidate_index: 1, candidates_total: 1, completed_candidates: 0 },
      { type: "state", state: readyState },
    ].map((event) => `data: ${JSON.stringify(event)}\n\n`).join("");
    const fetchMock = vi.fn().mockResolvedValue(new Response(streamBody, {
      status: 200,
      headers: { "Content-Type": "text/event-stream" },
    }));
    vi.stubGlobal("fetch", fetchMock);
    const received: string[] = [];
    const receivedEvents: import("@/lib/laboratory/genetic-client").GeneticEvent[] = [];

    await streamGeneticExperiment("start", null, (event) => { received.push(event.type); receivedEvents.push(event); });

    expect(received).toEqual(["evaluation_started", "game_started", "move_played", "state"]);
    expect(receivedEvents[1]).toMatchObject({ opponent_id: "BASELINE", opponent_type: "baseline", color: "white" });
    expect(fetchMock.mock.calls[0]?.[0]).toBe("http://127.0.0.1:8000/api/lab/genetic/start/stream");
  });

  it("reports deployments that do not serve SSE instead of simulating live progress", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify(readyState), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    })));

    await expect(streamGeneticExperiment("start", null, () => undefined))
      .rejects.toThrow("did not provide a live event stream");
  });
});
