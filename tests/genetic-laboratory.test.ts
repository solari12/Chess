import { createElement } from "react";
import { renderToString } from "react-dom/server";
import { afterEach, describe, expect, it, vi } from "vitest";
import GeneticLaboratoryPage from "@/app/laboratory/genetic/page";
import { EvolutionLineageVisualizer } from "@/app/laboratory/genetic/page";
import {
  candidateTraceById,
  getCandidateSets,
  getGeneticState,
  getIndividualBank,
  getSavedExperiments,
  individualById,
  loadGeneticExperiment,
  lineageEntriesForStep,
  replayCandidateWeights,
  resetGeneticExperiment,
  saveGeneticExperiment,
  saveIndividualsToBank,
  saveTiedBestCandidateSet,
  startGeneticExperiment,
  stepGeneticExperiment,
  streamGeneticExperiment,
  type CandidateGameTrace,
  type EvolutionStep,
  type GeneticConfig,
  type GeneticState,
  type Individual,
  type PieceWeights,
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
  game_report: {
    total_games: 0,
    wins: 0,
    draws: 0,
    losses: 0,
    draws_by_termination_reason: {},
    termination_counts: {},
    average_plies: 0,
    max_plies: 0,
  },
};

const parentAWeights: PieceWeights = { pawn: 96, knight: 310, bishop: 324, rook: 558, queen: 889 };
const parentBWeights: PieceWeights = { pawn: 101, knight: 320, bishop: 330, rook: 501, queen: 900 };
const finalWeights: PieceWeights = { pawn: 96, knight: 305, bishop: 330, rook: 558, queen: 889 };

function individual(id: string, weights: PieceWeights, generation: number, eliteFrom: string | null = null): Individual {
  return { id, weights, fitness: 1, generation, wins: 2, draws: 1, losses: 1, games_played: 4, elite_from: eliteFrom };
}

const parentA = individual("I0001", parentAWeights, 1);
const parentB = individual("I0002", parentBWeights, 1);
const elite = individual("I0011", parentAWeights, 2, "I0001");
const offspring = individual("I0012", finalWeights, 2);
const childEvent = {
  child_id: "I0012",
  generation: 2,
  individual: offspring,
  parent_a_id: "I0001",
  parent_b_id: "I0002",
  crossover_weights: { ...parentAWeights, bishop: 330 },
  pre_mutation_weights: { ...parentAWeights, bishop: 330 },
  post_mutation_weights: finalWeights,
  gene_origins: [
    { gene: "pawn" as const, value: 96, source_parent: "parent_a" as const, source_parent_id: "I0001" },
    { gene: "knight" as const, value: 310, source_parent: "parent_a" as const, source_parent_id: "I0001" },
    { gene: "bishop" as const, value: 330, source_parent: "parent_b" as const, source_parent_id: "I0002" },
    { gene: "rook" as const, value: 558, source_parent: "parent_a" as const, source_parent_id: "I0001" },
    { gene: "queen" as const, value: 889, source_parent: "parent_a" as const, source_parent_id: "I0001" },
  ],
  mutations: [{ gene: "knight" as const, old_value: 310, new_value: 305, delta: -5 }],
};
const nextStep: EvolutionStep = {
  generation: 2,
  population: [elite, offspring],
  selected_parents: ["I0001", "I0002"],
  parent_pairs: [{ parent_a_id: "I0001", parent_b_id: "I0002" }],
  children: [childEvent],
  elite_individuals: [elite],
  best_individual: offspring,
  average_fitness: 1,
};
const previousStep: EvolutionStep = {
  generation: 1,
  population: [parentA, parentB],
  selected_parents: [],
  parent_pairs: [],
  children: [],
  elite_individuals: [],
  best_individual: parentA,
  average_fitness: 1,
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
    expect(markup).toContain("Save Experiment");
    expect(markup).toContain("Load Experiment");
    expect(markup).toContain("Continue Experiment");
    expect(markup).toContain("Individual Bank");
    expect(markup).toContain("Create Experiment from Selected Individuals");
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
      time_control_ms: 5_000,
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
      { type: "generation_lineage", lineage: { generation: 2, population: nextStep.population, children: nextStep.children, elite_individuals: nextStep.elite_individuals } },
      { type: "game_started", generation: 1, candidate_id: "I0001", candidate_weights: { pawn: 100, knight: 320, bishop: 330, rook: 500, queen: 900 }, candidate_color: "white", color: "white", game_index: 1, games_total: 8, opponent_id: "BASELINE", opponent_type: "baseline", opponent_weights: { pawn: 100, knight: 320, bishop: 330, rook: 500, queen: 900 }, baseline_weights: { pawn: 100, knight: 320, bishop: 330, rook: 500, queen: 900 }, candidate_index: 1, candidates_total: 1, completed_candidates: 0 },
      { type: "move_played", generation: 1, candidate_id: "I0001", candidate_color: "white", opponent_id: "BASELINE", opponent_type: "baseline", game_index: 1, ply: 1, move_number: 1, color: "white", move: "e4", san: "e4", uci: "e2e4", fen: "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1", evaluation: 0.1, nodes: 1, depth: 2, search_budget_ms: 150, search_time_ms: 145, policy_time_ms: 35, remaining_time_ms: 4_820, candidate_index: 1, candidates_total: 1, completed_candidates: 0 },
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

    expect(received).toEqual(["evaluation_started", "generation_lineage", "game_started", "move_played", "state"]);
    expect(receivedEvents[1]).toMatchObject({ type: "generation_lineage", lineage: { generation: 2, children: [{ child_id: "I0012" }] } });
    expect(receivedEvents[2]).toMatchObject({ opponent_id: "BASELINE", opponent_type: "baseline", color: "white" });
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

  it("calls checkpoint, load, bank, and tied-best candidate-set endpoints", async () => {
    const fetchMock = vi.fn().mockImplementation(() => Promise.resolve(new Response(JSON.stringify({ saved: true }), { status: 200 })));
    vi.stubGlobal("fetch", fetchMock);

    await getSavedExperiments();
    await saveGeneticExperiment();
    await loadGeneticExperiment("exp_test_001");
    await getIndividualBank();
    await saveIndividualsToBank("exp_test_001", 5, ["I0035", "I0038"]);
    await getCandidateSets();
    await saveTiedBestCandidateSet("all tied best");

    expect(fetchMock.mock.calls.map(([url]) => url)).toEqual([
      "http://127.0.0.1:8000/api/lab/genetic/experiments",
      "http://127.0.0.1:8000/api/lab/genetic/experiments/save",
      "http://127.0.0.1:8000/api/lab/genetic/experiments/load",
      "http://127.0.0.1:8000/api/lab/genetic/bank",
      "http://127.0.0.1:8000/api/lab/genetic/bank/save",
      "http://127.0.0.1:8000/api/lab/genetic/candidate-sets",
      "http://127.0.0.1:8000/api/lab/genetic/candidate-sets/save-best",
    ]);
    expect(fetchMock.mock.calls[2]?.[1]).toMatchObject({ body: JSON.stringify({ experiment_id: "exp_test_001" }) });
    expect(fetchMock.mock.calls[4]?.[1]).toMatchObject({ body: JSON.stringify({ experiment_id: "exp_test_001", generation: 5, individual_ids: ["I0035", "I0038"] }) });
  });

  it("maps offspring and replay chromosomes by stable individual ID, independent of sorting", () => {
    const sortedParents = [parentB, parentA].sort((left, right) => (left.fitness ?? 0) - (right.fitness ?? 0));
    expect(individualById(sortedParents, "I0001")?.weights).toEqual(parentAWeights);
    expect(lineageEntriesForStep(nextStep).find((entry) => entry.id === "I0012")).toMatchObject({
      kind: "offspring",
      parent_a_id: "I0001",
      parent_b_id: "I0002",
      post_mutation_weights: finalWeights,
    });

    const trace: CandidateGameTrace = {
      generation: 2,
      candidate_id: "I0012",
      candidate_weights: finalWeights,
      candidate_color: "white",
      game_index: 1,
      games_total: 4,
      opponent_id: "I0001",
      opponent_type: "population",
      opponent_weights: parentAWeights,
      baseline_weights: parentAWeights,
      result: "draw",
      plies: 12,
      fitness_delta: 0,
      termination: "stalemate",
      moves: [],
    };
    const otherCandidateTrace = { ...trace, candidate_id: "I0011", candidate_weights: parentAWeights };
    const selectedTrace = candidateTraceById([otherCandidateTrace, trace], "I0012", 1);
    expect(selectedTrace?.candidate_id).toBe("I0012");
    expect(replayCandidateWeights(selectedTrace!)).toEqual(finalWeights);
    expect(replayCandidateWeights(selectedTrace!)).not.toEqual(parentAWeights);
  });

  it("reads lineage from a running backend that still sends the previous child event schema", () => {
    const { child_id: _childId, generation: _generation, pre_mutation_weights: _pre, post_mutation_weights: _post, ...legacyChild } = childEvent;
    const legacyStep: EvolutionStep = { ...nextStep, children: [legacyChild] };
    const lineage = lineageEntriesForStep(legacyStep).find((entry) => entry.id === "I0012");
    expect(lineage).toMatchObject({
      kind: "offspring",
      id: "I0012",
      generation: 2,
      pre_mutation_weights: childEvent.crossover_weights,
      post_mutation_weights: finalWeights,
    });
    const markup = renderToString(createElement(EvolutionLineageVisualizer, {
      snapshot: legacyStep,
      previousSnapshot: previousStep,
      selectedId: "I0012",
      onSelect: () => undefined,
    }));
    expect(markup).toContain("Parent A");
    expect(markup).toContain("I0001");
    expect(markup).toContain("Parent B");
    expect(markup).toContain("I0002");
  });

  it("renders backend lineage metadata, crossover, mutations, and elite clones after advancing", () => {
    const defaultMarkup = renderToString(createElement(EvolutionLineageVisualizer, {
      snapshot: nextStep,
      previousSnapshot: previousStep,
      selectedId: null,
      onSelect: () => undefined,
    }));
    expect(defaultMarkup).toContain("Parent A");
    expect(defaultMarkup).toContain("Parent B");

    const offspringMarkup = renderToString(createElement(EvolutionLineageVisualizer, {
      snapshot: nextStep,
      previousSnapshot: previousStep,
      selectedId: "I0012",
      onSelect: () => undefined,
    }));
    expect(offspringMarkup).toContain("I0012");
    expect(offspringMarkup).toContain("Parent A");
    expect(offspringMarkup).toContain("I0001");
    expect(offspringMarkup).toContain("Parent B");
    expect(offspringMarkup).toContain("I0002");
    expect(offspringMarkup).toContain("Crossover");
    expect(offspringMarkup).toContain("pre-mutation");
    expect(offspringMarkup).toContain("Mutation");
    expect(offspringMarkup).toContain("305");
    expect(offspringMarkup).toContain("Final chromosome after mutation");
    expect(offspringMarkup).toContain("330");
    expect(offspringMarkup).toContain("elite clone");

    const eliteMarkup = renderToString(createElement(EvolutionLineageVisualizer, {
      snapshot: nextStep,
      previousSnapshot: previousStep,
      selectedId: "I0011",
      onSelect: () => undefined,
    }));
    expect(eliteMarkup).toContain("Copied from");
    expect(eliteMarkup).toContain("I0001");
    expect(eliteMarkup).toContain("Final elite chromosome");
    expect(eliteMarkup).toContain("carries the parent chromosome forward unchanged");
  });

  it("shows a bank-seeded individual and its source lineage", () => {
    const bankSeed = individual("I0035", parentAWeights, 1);
    bankSeed.bank_source_experiment_id = "exp_source_001";
    bankSeed.bank_source_generation = 5;
    bankSeed.bank_source_individual_id = "I0035";
    bankSeed.bank_source_lineage = { kind: "offspring", parent_a_id: "I0010", parent_b_id: "I0011" };
    const seedStep: EvolutionStep = { ...previousStep, population: [bankSeed], best_individual: bankSeed };
    const lineage = lineageEntriesForStep(seedStep).find((entry) => entry.id === "I0035");
    expect(lineage).toMatchObject({ kind: "bank_seed", source_experiment_id: "exp_source_001", source_generation: 5 });
    const markup = renderToString(createElement(EvolutionLineageVisualizer, {
      snapshot: seedStep,
      previousSnapshot: undefined,
      selectedId: "I0035",
      onSelect: () => undefined,
    }));
    expect(markup).toContain("Individual Bank seed");
    expect(markup).toContain("exp_source_001");
    expect(markup).toContain("generation ");
    expect(markup).toContain("<!-- -->5");
    expect(markup).toContain("View source lineage");
  });
});
