export interface PieceWeights {
  pawn: number;
  knight: number;
  bishop: number;
  rook: number;
  queen: number;
}

export interface Individual {
  id: string;
  weights: PieceWeights;
  fitness: number | null;
  generation: number;
  wins: number;
  draws: number;
  losses: number;
  games_played: number;
  elite_from: string | null;
  bank_source_experiment_id?: string | null;
  bank_source_generation?: number | null;
  bank_source_individual_id?: string | null;
  bank_source_lineage?: Record<string, unknown> | null;
}

export interface GameMoveTrace {
  ply: number;
  move_number: number;
  color: "white" | "black";
  san: string;
  uci: string;
  fen: string;
  evaluation: number;
  nodes: number;
  depth: number;
  search_budget_ms: number;
  search_time_ms: number;
  policy_time_ms: number;
  remaining_time_ms: number;
}

export interface CandidateGameTrace {
  generation: number;
  candidate_id: string;
  candidate_weights: PieceWeights;
  candidate_color: "white" | "black";
  game_index: number;
  games_total: number;
  opponent_id: string;
  opponent_type: "baseline" | "population" | "elite";
  opponent_weights: PieceWeights;
  baseline_weights: PieceWeights;
  result: "win" | "draw" | "loss" | null;
  plies: number;
  fitness_delta: number;
  termination: string | null;
  moves: GameMoveTrace[];
}

export interface GameReport {
  total_games: number;
  wins: number;
  draws: number;
  losses: number;
  draws_by_termination_reason: Record<string, number>;
  termination_counts: Record<string, number>;
  average_plies: number;
  max_plies: number;
}

export interface MutationEvent {
  gene: keyof PieceWeights;
  old_value: number;
  new_value: number;
  delta: number;
}

export interface GeneOrigin {
  gene: keyof PieceWeights;
  value: number;
  source_parent: "parent_a" | "parent_b";
  source_parent_id: string;
}

export interface ChildEvent {
  child_id?: string;
  generation?: number;
  individual: Individual;
  parent_a_id: string;
  parent_b_id: string;
  crossover_weights: PieceWeights;
  pre_mutation_weights?: PieceWeights;
  post_mutation_weights?: PieceWeights;
  gene_origins: GeneOrigin[];
  mutations: MutationEvent[];
}

export type EvolutionLineageEntry =
  | {
      kind: "offspring";
      id: string;
      generation: number;
      individual: Individual;
      parent_a_id: string;
      parent_b_id: string;
      pre_mutation_weights: PieceWeights;
      post_mutation_weights: PieceWeights;
      gene_origins: GeneOrigin[];
      mutations: MutationEvent[];
    }
  | {
      kind: "elite_clone";
      id: string;
      generation: number;
      individual: Individual;
      parent_a_id: string;
      post_mutation_weights: PieceWeights;
    }
  | {
      kind: "bank_seed";
      id: string;
      generation: number;
      individual: Individual;
      source_experiment_id: string;
      source_generation: number;
      source_individual_id: string;
      post_mutation_weights: PieceWeights;
    };

export interface ParentPair {
  parent_a_id: string;
  parent_b_id: string;
}

export interface EvolutionStep {
  generation: number;
  population: Individual[];
  selected_parents: string[];
  parent_pairs: ParentPair[];
  children: ChildEvent[];
  elite_individuals: Individual[];
  best_individual: Individual;
  average_fitness: number;
}

export type EvolutionLineageSnapshot = Pick<
  EvolutionStep,
  "generation" | "population" | "children" | "elite_individuals"
>;

export interface GeneticConfig {
  population_size: number;
  games_per_individual: number;
  time_control_ms: number;
  mutation_rate: number;
  mutation_strength: number;
  elite_count: number;
  tournament_size: number;
  max_plies: number;
  generations: number;
  seed: number | null;
}

export interface GeneticState {
  experiment_id?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
  status: "ready" | "running" | "complete";
  generation: number;
  generation_limit: number;
  config: GeneticConfig | null;
  population: Individual[];
  history: EvolutionStep[];
  active_lineage?: EvolutionLineageSnapshot | null;
  game_traces: CandidateGameTrace[];
  game_report: GameReport;
}

export interface SavedExperimentSummary {
  experiment_id: string;
  created_at: string;
  updated_at: string;
  current_generation: number;
  generation_limit: number;
  population_size: number;
  best_fitness: number | null;
}

export interface BankIndividual {
  bank_id: string;
  individual_id: string;
  source_experiment_id: string;
  source_generation: number;
  chromosome: PieceWeights;
  fitness: number | null;
  wins: number;
  draws: number;
  losses: number;
  games_played: number;
  lineage: Record<string, unknown> | null;
  created_at: string;
  tags: string[];
  notes: string;
}

export interface SavedCandidateSet {
  candidate_set_id: string;
  name: string;
  source_experiment_id: string;
  source_generation: number;
  bank_ids: string[];
  created_at: string;
}

export function individualById(population: Individual[], individualId: string | null): Individual | null {
  return individualId === null ? null : population.find((individual) => individual.id === individualId) ?? null;
}

export function lineageEntriesForStep(step: EvolutionLineageSnapshot): EvolutionLineageEntry[] {
  // Older running API processes omit the explicit child_id/generation and
  // pre/post-mutation fields. The nested individual ID is still stable and
  // lets us render that lineage without relying on a sorted population index.
  const offspringById = new Map(step.children.map((child) => [child.child_id ?? child.individual.id, child]));
  const eliteById = new Map(step.elite_individuals.map((elite) => [elite.id, elite]));
  return step.population.flatMap((individual): EvolutionLineageEntry[] => {
    const child = offspringById.get(individual.id);
    if (child) {
      const childId = child.child_id ?? child.individual.id;
      return [{
        kind: "offspring",
        id: childId,
        generation: child.generation ?? child.individual.generation,
        individual: child.individual,
        parent_a_id: child.parent_a_id,
        parent_b_id: child.parent_b_id,
        pre_mutation_weights: child.pre_mutation_weights ?? child.crossover_weights,
        post_mutation_weights: child.post_mutation_weights ?? child.individual.weights,
        gene_origins: child.gene_origins,
        mutations: child.mutations,
      }];
    }
    const elite = eliteById.get(individual.id);
    if (elite?.elite_from) {
      return [{
        kind: "elite_clone",
        id: elite.id,
        generation: elite.generation,
        individual: elite,
        parent_a_id: elite.elite_from,
        post_mutation_weights: elite.weights,
      }];
    }
    if (individual.bank_source_experiment_id && individual.bank_source_generation !== null && individual.bank_source_generation !== undefined) {
      return [{
        kind: "bank_seed",
        id: individual.id,
        generation: individual.generation,
        individual,
        source_experiment_id: individual.bank_source_experiment_id,
        source_generation: individual.bank_source_generation,
        source_individual_id: individual.bank_source_individual_id ?? individual.id,
        post_mutation_weights: individual.weights,
      }];
    }
    return [];
  });
}

export function replayCandidateWeights(trace: CandidateGameTrace): PieceWeights {
  return trace.candidate_weights;
}

export function candidateTraceById(
  traces: CandidateGameTrace[],
  candidateId: string,
  gameIndex: number,
): CandidateGameTrace | null {
  return traces.find((trace) => trace.candidate_id === candidateId && trace.game_index === gameIndex) ?? null;
}

export type GeneticEvent =
  | { type: "evaluation_started"; generation: number; candidates_total: number; completed_candidates: number; games_per_individual: number }
  | { type: "generation_lineage"; lineage: EvolutionLineageSnapshot }
  | { type: "candidate_started"; generation: number; candidate_id: string; candidate_weights: PieceWeights; candidate_index: number; candidates_total: number; completed_candidates: number }
  | { type: "game_started"; generation: number; candidate_id: string; candidate_weights: PieceWeights; candidate_color: "white" | "black"; color: "white" | "black"; opponent_id: string; opponent_type: "baseline" | "population" | "elite"; opponent_weights: PieceWeights; game_index: number; games_total: number; baseline_weights: PieceWeights; candidate_index: number; candidates_total: number; completed_candidates: number }
  | ({ type: "move_played"; generation: number; candidate_id: string; candidate_color: "white" | "black"; opponent_id: string; opponent_type: "baseline" | "population" | "elite"; game_index: number; move: string } & GameMoveTrace & { candidate_index: number; candidates_total: number; completed_candidates: number })
  | { type: "game_finished"; generation: number; candidate_id: string; candidate_color: "white" | "black"; color: "white" | "black"; opponent_id: string; opponent_type: "baseline" | "population" | "elite"; game_index: number; result: "win" | "draw" | "loss"; plies: number; termination: string; fitness_delta: number; candidate_index: number; candidates_total: number; completed_candidates: number }
  | { type: "candidate_game_finished"; generation: number; candidate_id: string; opponent_id: string; opponent_type: "baseline" | "population" | "elite"; color: "white" | "black"; game_index: number; record: Pick<Individual, "wins" | "draws" | "losses" | "games_played">; fitness: number; candidate_index: number; candidates_total: number; completed_candidates: number }
  | { type: "candidate_finished"; generation: number; candidate_id: string; record: Pick<Individual, "wins" | "draws" | "losses" | "games_played">; fitness: number; candidate_index: number; candidates_total: number; completed_candidates: number }
  | { type: "generation_complete"; generation: number; completed_candidates: number; candidates_total: number }
  | { type: "state"; state: GeneticState }
  | { type: "stream_error"; detail: string };

const baseUrl = (process.env.NEXT_PUBLIC_CHESS_AI_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");
const endpoint = `${baseUrl}/api/lab/genetic`;

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${endpoint}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`Laboratory request failed (${response.status}): ${detail}`);
  }
  return response.json() as Promise<T>;
}

export const getGeneticState = () => request<GeneticState>("/state");
export const getSavedExperiments = () => request<SavedExperimentSummary[]>("/experiments");
export const saveGeneticExperiment = () => request<SavedExperimentSummary>("/experiments/save", { method: "POST" });
export const loadGeneticExperiment = (experimentId: string) => request<GeneticState>("/experiments/load", {
  method: "POST",
  body: JSON.stringify({ experiment_id: experimentId }),
});
export const getIndividualBank = () => request<BankIndividual[]>("/bank");
export const saveIndividualsToBank = (individualIds: string[]) => request<BankIndividual[]>("/bank/save", {
  method: "POST",
  body: JSON.stringify({ individual_ids: individualIds }),
});
export const getCandidateSets = () => request<SavedCandidateSet[]>("/candidate-sets");
export const saveTiedBestCandidateSet = (name: string) => request<{
  candidate_set: SavedCandidateSet;
  individuals: BankIndividual[];
}>("/candidate-sets/save-best", { method: "POST", body: JSON.stringify({ name }) });

export const startGeneticExperiment = (config: GeneticConfig) =>
  request<GeneticState>("/start", { method: "POST", body: JSON.stringify(config) });

export const stepGeneticExperiment = () =>
  request<GeneticState>("/step", { method: "POST" });

export const resetGeneticExperiment = () =>
  request<GeneticState>("/reset", { method: "POST" });

export async function streamGeneticExperiment(
  action: "start" | "step" | "start-from-bank",
  config: GeneticConfig | null,
  onEvent: (event: GeneticEvent) => void,
  signal?: AbortSignal,
  bankIds: string[] = [],
): Promise<void> {
  const path = action === "start"
    ? "/start/stream"
    : action === "start-from-bank" ? "/start-from-bank/stream" : "/step/stream";
  const body = action === "start-from-bank"
    ? JSON.stringify({ config, bank_ids: bankIds })
    : config ? JSON.stringify(config) : undefined;
  const response = await fetch(`${endpoint}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body,
    signal,
  });
  if (!response.ok) {
    throw new Error(`Laboratory stream failed (${response.status}): ${await response.text()}`);
  }
  if (!response.headers.get("content-type")?.includes("text/event-stream") || !response.body) {
    throw new Error("The backend deployment did not provide a live event stream.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let receivedState = false;
  for (;;) {
    const { done, value } = await reader.read();
    buffer += decoder.decode(value, { stream: !done });
    buffer = buffer.replace(/\r\n/g, "\n");
    let boundary = buffer.indexOf("\n\n");
    while (boundary >= 0) {
      const frame = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      const data = frame
        .split("\n")
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trimStart())
        .join("\n");
      if (data) {
        const event = JSON.parse(data) as GeneticEvent;
        if (event.type === "stream_error") throw new Error(event.detail);
        if (event.type === "state") receivedState = true;
        onEvent(event);
      }
      boundary = buffer.indexOf("\n\n");
    }
    if (done) break;
  }
  if (!receivedState) {
    throw new Error("The live stream ended before the backend reported a completed generation.");
  }
}
