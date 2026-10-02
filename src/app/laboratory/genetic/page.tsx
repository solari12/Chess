"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import {
  Activity,
  ArrowLeft,
  Dna,
  FlaskConical,
  GitBranch,
  Pause,
  Play,
  RotateCcw,
  StepForward,
  Swords,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  getGeneticState,
  getCandidateSets,
  getIndividualBank,
  getSavedExperiments,
  loadGeneticExperiment,
  saveGeneticExperiment,
  saveIndividualsToBank,
  saveTiedBestCandidateSet,
  candidateTraceById,
  individualById,
  lineageEntriesForStep,
  replayCandidateWeights,
  resetGeneticExperiment,
  streamGeneticExperiment,
  type CandidateGameTrace,
  type BankIndividual,
  type SavedCandidateSet,
  type SavedExperimentSummary,
  type EvolutionLineageSnapshot,
  type EvolutionStep,
  type GeneticEvent,
  type GeneticConfig,
  type GeneticState,
  type Individual,
  type PieceWeights,
} from "@/lib/laboratory/genetic-client";

const initialConfig: GeneticConfig = {
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

const geneLabels: Record<keyof PieceWeights, string> = {
  pawn: "Pawn",
  knight: "Knight",
  bishop: "Bishop",
  rook: "Rook",
  queen: "Queen",
};
const geneSymbols: Record<keyof PieceWeights, string> = {
  pawn: "♙",
  knight: "♘",
  bishop: "♗",
  rook: "♖",
  queen: "♕",
};
const geneKeys = Object.keys(geneLabels) as (keyof PieceWeights)[];

export default function GeneticLaboratoryPage() {
  const router = useRouter();
  const [state, setState] = useState<GeneticState | null>(null);
  const [config, setConfig] = useState(initialConfig);
  const [operation, setOperation] = useState<"loading" | "start" | "step" | "run" | "reset" | null>("loading");
  const [error, setError] = useState<string | null>(null);
  const [selectedGeneration, setSelectedGeneration] = useState<number | null>(null);
  const [selectedIndividualId, setSelectedIndividualId] = useState<string | null>(null);
  const [selectedChildId, setSelectedChildId] = useState<string | null>(null);
  const [pendingLineage, setPendingLineage] = useState<EvolutionLineageSnapshot | null>(null);
  const [savedExperiments, setSavedExperiments] = useState<SavedExperimentSummary[]>([]);
  const [selectedExperimentId, setSelectedExperimentId] = useState("");
  const [individualBank, setIndividualBank] = useState<BankIndividual[]>([]);
  const [candidateSets, setCandidateSets] = useState<SavedCandidateSet[]>([]);
  const [selectedCandidateSetId, setSelectedCandidateSetId] = useState("");
  const [selectedBankIds, setSelectedBankIds] = useState<string[]>([]);
  const [selectedPopulationIds, setSelectedPopulationIds] = useState<string[]>([]);
  const [archiveBusy, setArchiveBusy] = useState(false);
  const [archiveNotice, setArchiveNotice] = useState<string | null>(null);
  const [paused, setPaused] = useState(false);
  const [liveEvents, setLiveEvents] = useState<GeneticEvent[]>([]);
  const [liveTraces, setLiveTraces] = useState<CandidateGameTrace[]>([]);
  const [progressEvent, setProgressEvent] = useState<GeneticEvent | null>(null);
  const [followLive, setFollowLive] = useState(true);
  const [replay, setReplay] = useState<{ candidate_id: string; game_index: number; ply: number } | null>(null);
  const runEnabled = useRef(false);
  const pausedByUser = useRef(false);
  const requestActive = useRef(false);
  const resetPending = useRef(false);
  const latestState = useRef<GeneticState | null>(null);

  const handleEvent = (event: GeneticEvent) => {
    setLiveEvents((current) => event.type === "evaluation_started" ? [event] : [...current, event]);
    setProgressEvent(event);
    if (event.type === "evaluation_started") {
      setLiveTraces([]);
      setProgressEvent(event);
      setReplay(null);
      setFollowLive(true);
    } else if (event.type === "candidate_started" || event.type === "move_played" || event.type === "candidate_finished") {
      setProgressEvent(event);
    } else if (event.type === "state") {
      latestState.current = event.state;
      setState(event.state);
      setPendingLineage(null);
      setProgressEvent(event);
      setLiveTraces(event.state.game_traces);
    } else if (event.type === "generation_lineage") {
      setPendingLineage(event.lineage);
    }

    if (event.type === "game_started") {
      setLiveTraces((current) => [
        ...current,
        {
          generation: event.generation,
          candidate_id: event.candidate_id,
          candidate_weights: event.candidate_weights,
          candidate_color: event.candidate_color,
          game_index: event.game_index,
          games_total: event.games_total,
          opponent_id: event.opponent_id,
          opponent_type: event.opponent_type,
          opponent_weights: event.opponent_weights,
          baseline_weights: event.baseline_weights,
          result: null,
          plies: 0,
          fitness_delta: 0,
          termination: null,
          moves: [],
        },
      ]);
    } else if (event.type === "move_played") {
      setLiveTraces((current) => current.map((trace) => trace.candidate_id === event.candidate_id && trace.game_index === event.game_index
        ? { ...trace, moves: [...trace.moves, { ply: event.ply, move_number: event.move_number, color: event.color, san: event.san, uci: event.uci, fen: event.fen, evaluation: event.evaluation, nodes: event.nodes, depth: event.depth, search_budget_ms: event.search_budget_ms, search_time_ms: event.search_time_ms, policy_time_ms: event.policy_time_ms, remaining_time_ms: event.remaining_time_ms }] }
        : trace));
    } else if (event.type === "game_finished") {
      setLiveTraces((current) => current.map((trace) => trace.candidate_id === event.candidate_id && trace.game_index === event.game_index
        ? { ...trace, result: event.result, plies: event.plies, fitness_delta: event.fitness_delta, termination: event.termination }
        : trace));
    }
  };

  useEffect(() => {
    let mounted = true;
    getGeneticState()
      .then((result) => {
        if (!mounted) return;
        setState(result);
        setPendingLineage(result.active_lineage ?? null);
        latestState.current = result;
        if (result.config) setConfig(result.config);
        setLiveTraces(result.game_traces ?? []);
      })
      .catch((reason: unknown) => {
        if (mounted) setError(messageFrom(reason));
      })
      .finally(() => {
        if (mounted) setOperation(null);
      });
    void getSavedExperiments().then(setSavedExperiments).catch(() => setSavedExperiments([]));
    void getIndividualBank().then(setIndividualBank).catch((reason: unknown) => {
      if (mounted) setError(`Could not load Individual Bank: ${messageFrom(reason)}`);
    });
    void getCandidateSets().then(setCandidateSets).catch(() => setCandidateSets([]));
    return () => {
      mounted = false;
      runEnabled.current = false;
    };
  }, []);

  useEffect(() => {
    if (operation !== null || state?.status !== "running") return;
    let mounted = true;
    const refreshState = async () => {
      try {
        const result = await getGeneticState();
        if (!mounted) return;
        setState(result);
        setPendingLineage(result.active_lineage ?? null);
        latestState.current = result;
        setLiveTraces(result.game_traces ?? []);
      } catch {
        // Keep showing the last known running state; a later poll can recover.
      }
    };
    const timer = window.setInterval(() => { void refreshState(); }, 1_000);
    return () => {
      mounted = false;
      window.clearInterval(timer);
    };
  }, [operation, state?.status]);

  const snapshot = useMemo(() => {
    if (!state?.history.length) return null;
    const generation = selectedGeneration ?? state.generation;
    return state.history.find((step) => step.generation === generation) ?? state.history.at(-1) ?? null;
  }, [selectedGeneration, state]);
  const visualizerSnapshot = selectedGeneration === null && pendingLineage
    ? pendingLineage
    : snapshot;
  const bestCandidate = state?.history.at(-1)?.best_individual ?? null;
  const previousSnapshot = state?.history.find((step) => step.generation === (visualizerSnapshot?.generation ?? 0) - 1);
  const selectedIndividual = snapshot ? individualById(snapshot.population, selectedIndividualId)
    ?? snapshot?.best_individual
    ?? null : null;
  const busy = operation !== null;
  const status = operation === "run" || operation === "step" || operation === "start"
    ? paused && state?.status !== "complete" ? operation === "run" ? "PAUSING AFTER GEN" : "PAUSED" : "EVOLVING"
    : state?.status === "complete"
      ? "COMPLETE"
      : state?.status === "running"
        ? "EVOLVING"
      : paused
        ? "PAUSED"
        : state?.generation
          ? "READY"
          : "READY TO START";

  const updateConfig = <K extends keyof GeneticConfig>(key: K, value: GeneticConfig[K]) => {
    setConfig((current) => ({ ...current, [key]: value }));
  };

  const useBestCandidateInChess = () => {
    if (!bestCandidate) return;
    window.localStorage.setItem("chess.genetic-player-profile", JSON.stringify({
      candidate_id: bestCandidate.id,
      fitness: bestCandidate.fitness ?? 0,
      weights: bestCandidate.weights,
    }));
    window.localStorage.setItem("chess.genetic-auto-play", "black");
    router.push("/");
  };

  const runStream = async (action: "start" | "step", streamConfig: GeneticConfig | null) => {
    if (requestActive.current) return;
    requestActive.current = true;
    setOperation(action);
    setError(null);
    try {
      await streamGeneticExperiment(action, streamConfig, handleEvent);
    } catch (reason) {
      setError(messageFrom(reason));
    } finally {
      if (!resetPending.current) {
        requestActive.current = false;
        setOperation(null);
      }
    }
  };

  const start = async () => {
    setPaused(false);
    await runStream("start", config);
    setSelectedGeneration(null);
    setSelectedIndividualId(null);
    setSelectedChildId(null);
  };

  const step = async () => {
    setPaused(false);
    await runStream("step", null);
    setSelectedGeneration(null);
    setSelectedIndividualId(null);
    setSelectedChildId(null);
  };

  const refreshArchives = async () => {
    const [experiments, bank, sets] = await Promise.all([
      getSavedExperiments(),
      getIndividualBank(),
      getCandidateSets(),
    ]);
    setSavedExperiments(experiments);
    setIndividualBank(bank);
    setCandidateSets(sets);
  };

  const saveExperiment = async () => {
    setArchiveBusy(true);
    setError(null);
    try {
      const saved = await saveGeneticExperiment();
      await refreshArchives();
      setSelectedExperimentId(saved.experiment_id);
    } catch (reason) {
      setError(messageFrom(reason));
    } finally {
      setArchiveBusy(false);
    }
  };

  const loadExperiment = async () => {
    if (!selectedExperimentId) return;
    setOperation("loading");
    setError(null);
    try {
      const restored = await loadGeneticExperiment(selectedExperimentId);
      setState(restored);
      latestState.current = restored;
      setConfig(restored.config ?? initialConfig);
      setLiveTraces(restored.game_traces ?? []);
      setPendingLineage(restored.active_lineage ?? null);
      setSelectedGeneration(null);
      setSelectedIndividualId(null);
      setSelectedChildId(null);
    } catch (reason) {
      setError(messageFrom(reason));
    } finally {
      setOperation(null);
    }
  };

  const saveSelectedIndividuals = async () => {
    if (!selectedPopulationIds.length || !state?.experiment_id || !snapshot) return;
    setArchiveBusy(true);
    setError(null);
    setArchiveNotice(null);
    try {
      const saved = await saveIndividualsToBank(state.experiment_id, snapshot.generation, selectedPopulationIds);
      const bank = await getIndividualBank();
      setIndividualBank(bank);
      const visibleBankIds = new Set(bank.map((item) => item.bank_id));
      if (saved.some((item) => !visibleBankIds.has(item.bank_id))) {
        throw new Error("The API accepted the save, but the bank read-back did not contain those records. Check that the frontend and backend use the same API server and data directory.");
      }
      setSelectedPopulationIds([]);
      setArchiveNotice(`Saved ${saved.length} individual${saved.length === 1 ? "" : "s"} to the bank.`);
    } catch (reason) {
      setError(messageFrom(reason));
    } finally {
      setArchiveBusy(false);
    }
  };

  const saveTiedBest = async () => {
    if (!state?.experiment_id || !state.population.length) return;
    setArchiveBusy(true);
    setError(null);
    setArchiveNotice(null);
    try {
      const result = await saveTiedBestCandidateSet(`${state.experiment_id} · Gen ${state.generation} tied best`);
      const bank = await getIndividualBank();
      setIndividualBank(bank);
      const visibleBankIds = new Set(bank.map((item) => item.bank_id));
      if (result.individuals.some((item) => !visibleBankIds.has(item.bank_id))) {
        throw new Error("The API accepted the tied-best save, but the bank read-back did not contain those records. Check that the frontend and backend use the same API server and data directory.");
      }
      setCandidateSets(await getCandidateSets());
      setSelectedCandidateSetId(result.candidate_set.candidate_set_id);
      setSelectedBankIds(result.candidate_set.bank_ids);
      setArchiveNotice(`Saved all ${result.individuals.length} tied-best individual${result.individuals.length === 1 ? "" : "s"} to the bank.`);
    } catch (reason) {
      setError(messageFrom(reason));
    } finally {
      setArchiveBusy(false);
    }
  };

  const selectCandidateSet = (candidateSetId: string) => {
    setSelectedCandidateSetId(candidateSetId);
    const candidateSet = candidateSets.find((item) => item.candidate_set_id === candidateSetId);
    setSelectedBankIds(candidateSet?.bank_ids ?? []);
  };

  const startFromSelectedBank = async () => {
    if (!selectedBankIds.length) return;
    if (selectedBankIds.length > config.population_size) {
      setError("Selected bank individuals cannot exceed Population Size.");
      return;
    }
    if (requestActive.current) return;
    requestActive.current = true;
    setOperation("start");
    setError(null);
    setPaused(false);
    try {
      await streamGeneticExperiment("start-from-bank", config, handleEvent, undefined, selectedBankIds);
      await refreshArchives();
      setSelectedGeneration(null);
      setSelectedIndividualId(null);
      setSelectedChildId(null);
      setSelectedPopulationIds([]);
    } catch (reason) {
      setError(messageFrom(reason));
    } finally {
      requestActive.current = false;
      setOperation(null);
    }
  };

  const runGenerations = async (stepLimit: number, allowStart = false) => {
    if (requestActive.current || !state || (!allowStart && !state.generation) || state.status === "complete") return;
    requestActive.current = true;
    runEnabled.current = true;
    pausedByUser.current = false;
    setPaused(false);
    setError(null);
    setOperation("run");
    try {
      for (let index = 0; index < stepLimit && runEnabled.current; index += 1) {
        const isInitialGeneration = (latestState.current?.generation ?? state.generation) === 0;
        await streamGeneticExperiment(isInitialGeneration ? "start" : "step", isInitialGeneration ? config : null, handleEvent);
        setSelectedGeneration(null);
        if (latestState.current?.status === "complete") break;
      }
    } catch (reason) {
      setError(messageFrom(reason));
    } finally {
      runEnabled.current = false;
      setPaused(pausedByUser.current);
      if (!resetPending.current) {
        requestActive.current = false;
        setOperation(null);
      }
    }
  };

  const runTen = () => runGenerations(10);
  const continueExperiment = () => runGenerations(Math.max(1, config.generations - (state?.generation ?? 0)), true);

  const pause = () => {
    runEnabled.current = false;
    pausedByUser.current = true;
    setPaused(true);
  };

  const reset = async () => {
    if (operation === "reset" || operation === "loading") return;
    resetPending.current = true;
    requestActive.current = true;
    runEnabled.current = false;
    setPaused(false);
    pausedByUser.current = false;
    setOperation("reset");
    setError(null);
    try {
      const next = await resetGeneticExperiment();
      setState(next);
      latestState.current = next;
      setLiveTraces([]);
      setLiveEvents([]);
      setProgressEvent(null);
      setSelectedGeneration(null);
      setSelectedIndividualId(null);
      setSelectedChildId(null);
    } catch (reason) {
      setError(messageFrom(reason));
    } finally {
      resetPending.current = false;
      requestActive.current = false;
      setOperation(null);
    }
  };

  const sortedPopulation = snapshot
    ? [...snapshot.population].sort((left, right) => (right.fitness ?? -Infinity) - (left.fitness ?? -Infinity))
    : [];
  const bestFitnessInSnapshot = sortedPopulation.reduce<number | null>((best, individual) => (
    individual.fitness === null ? best : best === null ? individual.fitness : Math.max(best, individual.fitness)
  ), null);
  const tiedBestCount = sortedPopulation.filter((individual) => individual.fitness === bestFitnessInSnapshot).length;
  const displayedTrace = replay
    ? candidateTraceById(liveTraces, replay.candidate_id, replay.game_index)
    : liveTraces.at(-1) ?? null;
  const gameReport = useMemo(() => {
    const completed = liveTraces.filter((trace) => trace.result !== null);
    const drawsByReason: Record<string, number> = {};
    for (const trace of completed) {
      if (trace.result === "draw") {
        const reason = trace.termination ?? "other";
        drawsByReason[reason] = (drawsByReason[reason] ?? 0) + 1;
      }
    }
    const plies = completed.map((trace) => trace.plies);
    return {
      total_games: completed.length,
      wins: completed.filter((trace) => trace.result === "win").length,
      draws: completed.filter((trace) => trace.result === "draw").length,
      losses: completed.filter((trace) => trace.result === "loss").length,
      draws_by_termination_reason: drawsByReason,
      average_plies: plies.length ? plies.reduce((total, ply) => total + ply, 0) / plies.length : 0,
      max_plies: plies.length ? Math.max(...plies) : 0,
    };
  }, [liveTraces]);
  const displayedPly = replay ? Math.min(replay.ply, displayedTrace?.moves.length ?? 0) : displayedTrace?.moves.length ?? 0;
  const displayedMove = displayedTrace?.moves[displayedPly - 1] ?? null;
  const displayedFen = displayedMove?.fen ?? "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

  return (
    <main className="mx-auto min-h-screen w-full max-w-[1500px] px-4 pb-10 pt-5 sm:px-7 lg:px-10">
      <header className="mb-8 flex flex-wrap items-center justify-between gap-4 border-b border-[#d5cebf] pb-5">
        <div className="flex items-center gap-4">
          <Link href="/" aria-label="Back to chess game" className="grid size-10 place-items-center rounded-lg border border-[#d5cebf] bg-[#fbf9f3] text-[#48563c] hover:bg-white">
            <ArrowLeft size={17} />
          </Link>
          <div className="grid size-11 place-items-center rounded-xl bg-[#48563c] text-[#f5f1e6]">
            <Dna size={22} />
          </div>
          <div>
            <p className="m-0 text-[10px] font-semibold uppercase tracking-[.18em] text-[#8c744b]">Chess AI / Learning Lab</p>
            <h1 className="m-0 mt-1 font-display text-2xl font-semibold tracking-[-.04em] sm:text-3xl">Genetic Algorithm Laboratory</h1>
            <p className="mb-0 mt-1 text-sm text-[#706f65]">Evolution of chess piece values</p>
          </div>
        </div>
        <div className="flex items-center gap-3 rounded-lg border border-[#d5cebf] bg-[#fbf9f3] px-4 py-3">
          <span className={`size-2 rounded-full ${status === "EVOLVING" ? "animate-pulse motion-reduce:animate-none bg-[#a87643]" : "bg-[#74845c]"}`} />
          <span className="text-[10px] font-semibold tracking-[.14em] text-[#64645a]">{status}</span>
          <span className="mx-1 h-5 border-l border-[#d5cebf]" />
          <span className="font-mono text-xs text-[#343930]">GEN {String(state?.generation ?? 0).padStart(2, "0")} / {state?.generation_limit || config.generations}</span>
        </div>
      </header>

      {error && (
        <div role="alert" className="mb-5 flex items-start justify-between gap-4 border-l-4 border-[#a94f43] bg-[#f8ece7] px-4 py-3 text-sm text-[#713b34]">
          <span>{error}</span>
          <button type="button" onClick={() => setError(null)} aria-label="Dismiss error" className="font-semibold">×</button>
        </div>
      )}

      <section aria-labelledby="controls-heading" className="mb-7 border-y border-[#d5cebf] py-5">
        <div className="mb-4 flex items-center gap-2">
          <FlaskConical size={17} className="text-[#8c744b]" />
          <h2 id="controls-heading" className="m-0 text-sm font-semibold tracking-wide">Experiment controls</h2>
          <span className="ml-auto text-[10px] uppercase tracking-widest text-[#8b897e]">Start only when ready · no training runs on page load</span>
          <Button onClick={start} disabled={busy || state?.status === "running"} className="gap-2 disabled:opacity-100">{operation === "start" ? "Starting…" : state?.status === "running" ? "Experiment Running" : <><Play size={15} /> Start Experiment</>}</Button>
        </div>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-7">
          <NumberControl label="Population" value={config.population_size} min={2} max={20} onChange={(value) => setConfig((current) => ({ ...current, population_size: value, games_per_individual: Math.min(current.games_per_individual, Math.min(8, value * 2)), elite_count: Math.min(current.elite_count, value - 1), tournament_size: Math.min(current.tournament_size, value) }))} />
          <NumberControl label="Games / candidate" value={config.games_per_individual} min={4} max={Math.min(8, config.population_size * 2)} step={2} onChange={(value) => updateConfig("games_per_individual", Math.max(4, Math.min(Math.min(8, config.population_size * 2), Math.round(value / 2) * 2)))} />
          <NumberControl label="Clock / side (seconds)" value={config.time_control_ms / 1000} min={1} max={600} onChange={(value) => updateConfig("time_control_ms", value * 1000)} />
          <NumberControl label="Mutation rate" value={config.mutation_rate} min={0} max={1} step={0.05} onChange={(value) => updateConfig("mutation_rate", value)} />
          <NumberControl label="Mutation strength" value={config.mutation_strength} min={1} max={100} onChange={(value) => updateConfig("mutation_strength", value)} />
          <NumberControl label="Elite count" value={config.elite_count} min={0} max={Math.max(0, config.population_size - 1)} onChange={(value) => updateConfig("elite_count", value)} />
          <NumberControl label="Generations" value={config.generations} min={1} max={100} onChange={(value) => updateConfig("generations", value)} />
          <NumberControl label="Max plies" value={config.max_plies} min={1} max={200} onChange={(value) => updateConfig("max_plies", value)} />
          <NumberControl label="Tournament size" value={config.tournament_size} min={1} max={config.population_size} onChange={(value) => updateConfig("tournament_size", value)} />
          <NumberControl label="Random seed" value={config.seed ?? ""} min={0} max={2147483647} onChange={(value) => updateConfig("seed", value)} />
        </div>
        <div className="mt-4 flex flex-wrap items-center gap-2">
          <Button variant="copper" onClick={step} disabled={busy || !state?.generation || state.status === "complete" || state.status === "running"} className="gap-2"><StepForward size={15} /> Next Generation</Button>
          {operation === "run" ? (
            <Button variant="secondary" onClick={pause} className="gap-2"><Pause size={15} /> Pause</Button>
          ) : (
            <Button variant="secondary" onClick={runTen} disabled={busy || !state?.generation || state.status === "complete" || state.status === "running"} className="gap-2"><Activity size={15} /> Run 10 generations</Button>
          )}
          <Button variant="ghost" onClick={reset} disabled={operation === "reset" || operation === "loading"} className="gap-2"><RotateCcw size={15} /> Reset</Button>
          {operation === "loading" && <span className="ml-1 text-xs text-[#77776d]" aria-live="polite">Loading experiment state…</span>}
          {(operation === "start" || operation === "step" || operation === "run" || operation === "reset") && <span className="ml-1 text-xs text-[#77776d]" aria-live="polite">{operation === "reset" ? "Resetting experiment…" : "Evaluating self-play games…"}</span>}
        </div>
        <p className="mb-0 mt-3 text-[11px] leading-5 text-[#77776d]">Each side gets the same whole-game clock. Phase 2H allocates each move’s search budget, then iterative deepening searches until that budget expires; observed depth varies by position. Fitness remains wins minus losses, and games are capped by the ply limit.</p>
      </section>

      <section aria-labelledby="archive-heading" className="mb-7 border border-[#d5cebf] bg-[#fbf9f3] p-4">
        <div className="mb-3 flex flex-wrap items-start justify-between gap-3">
          <div>
            <p className="m-0 text-[9px] font-semibold uppercase tracking-[.15em] text-[#8c744b]">CHECKPOINTS</p>
            <h2 id="archive-heading" className="m-0 mt-1 text-sm font-semibold">Save, load, and continue</h2>
            <p className="mb-0 mt-1 text-[11px] text-[#77776d]">
              Experiment {state?.experiment_id ?? "not started"} · Generation {state?.generation ?? 0} / {state?.generation_limit ?? config.generations} · Last saved {savedExperiments.find((item) => item.experiment_id === state?.experiment_id)?.updated_at ?? "not saved"}
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <Button variant="secondary" onClick={saveExperiment} disabled={archiveBusy || !state?.experiment_id} className="text-xs">Save Experiment</Button>
            <Button variant="secondary" onClick={continueExperiment} disabled={busy || !state || state.status === "running" || state.status === "complete" || state.generation >= state.generation_limit} className="text-xs">Continue Experiment</Button>
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-2 border-t border-[#e2dccf] pt-3">
          <select aria-label="Saved experiment" value={selectedExperimentId} onChange={(event) => setSelectedExperimentId(event.target.value)} className="min-w-64 rounded-md border border-[#d5cebf] bg-white px-3 py-2 text-xs">
            <option value="">Select saved experiment</option>
            {savedExperiments.map((item) => <option key={item.experiment_id} value={item.experiment_id}>{item.experiment_id} · Gen {item.current_generation}/{item.generation_limit} · best {item.best_fitness === null ? "—" : signed(item.best_fitness)}</option>)}
          </select>
          <Button variant="outline" onClick={loadExperiment} disabled={archiveBusy || busy || !selectedExperimentId} className="text-xs">Load Experiment</Button>
          <Button variant="outline" onClick={saveTiedBest} disabled={archiveBusy || !state?.generation || !state.population.length} className="text-xs">Save Tied-Best Candidates</Button>
          <select aria-label="Saved candidate set" value={selectedCandidateSetId} onChange={(event) => selectCandidateSet(event.target.value)} className="min-w-64 rounded-md border border-[#d5cebf] bg-white px-3 py-2 text-xs">
            <option value="">Load Candidate Set</option>
            {candidateSets.map((item) => <option key={item.candidate_set_id} value={item.candidate_set_id}>{item.name} · {item.bank_ids.length} individuals</option>)}
          </select>
        </div>
      </section>

      <section aria-labelledby="game-report-heading" className="mb-7 border border-[#d5cebf] bg-[#fbf9f3] p-4">
        <div className="mb-3 flex flex-wrap items-baseline justify-between gap-2">
          <h2 id="game-report-heading" className="m-0 text-sm font-semibold">Game termination report · current evaluation</h2>
          <span className="text-[10px] text-[#77776d]">Completed games only</span>
        </div>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-5">
          <Metric label="Total games" value={gameReport.total_games} />
          <Metric label="Wins · candidate" value={gameReport.wins} />
          <Metric label="Draws" value={gameReport.draws} />
          <Metric label="Losses · candidate" value={gameReport.losses} />
          <Metric label="Avg / max plies" value={`${gameReport.average_plies.toFixed(1)} / ${gameReport.max_plies}`} />
        </div>
        <div className="mt-3 flex flex-wrap gap-x-5 gap-y-1 border-t border-[#e8e3d8] pt-3 text-[11px] text-[#626258]">
          <span className="font-semibold">Draw by termination:</span>
          {Object.entries(gameReport.draws_by_termination_reason).length
            ? Object.entries(gameReport.draws_by_termination_reason).map(([reason, count]) => <span key={reason}>{reason.replaceAll("_", " ")}: <strong>{count}</strong></span>)
            : <span>None recorded yet</span>}
        </div>
      </section>

      {(operation === "start" || operation === "step" || operation === "run" || liveTraces.length > 0) && <LiveSelfPlayMonitor
        traces={liveTraces}
        events={liveEvents}
        progress={progressEvent}
        active={operation === "start" || operation === "step" || operation === "run"}
        followLive={followLive}
        replay={replay}
        displayedTrace={displayedTrace}
        displayedFen={displayedFen}
        displayedPly={displayedPly}
        displayedMove={displayedMove}
        onPause={() => {
          setFollowLive(false);
          if (displayedTrace) setReplay({ candidate_id: displayedTrace.candidate_id, game_index: displayedTrace.game_index, ply: displayedTrace.moves.length });
        }}
        onFollow={() => { setFollowLive(true); setReplay(null); }}
        onSelectReplay={(trace) => { setFollowLive(false); setReplay({ candidate_id: trace.candidate_id, game_index: trace.game_index, ply: 0 }); }}
        onSelectPly={(ply) => { if (displayedTrace) { setFollowLive(false); setReplay({ candidate_id: displayedTrace.candidate_id, game_index: displayedTrace.game_index, ply }); } }}
        onPrevious={() => { if (displayedTrace) { setFollowLive(false); setReplay({ candidate_id: displayedTrace.candidate_id, game_index: displayedTrace.game_index, ply: Math.max(0, displayedPly - 1) }); } }}
        onNext={() => { if (displayedTrace) { const nextPly = Math.min(displayedTrace.moves.length, displayedPly + 1); if (nextPly === displayedTrace.moves.length && (operation === "start" || operation === "step" || operation === "run")) { setReplay(null); setFollowLive(true); } else { setReplay({ candidate_id: displayedTrace.candidate_id, game_index: displayedTrace.game_index, ply: nextPly }); } } }}
      />}

      <section className="mb-7 grid grid-cols-2 gap-3 lg:grid-cols-4" aria-label="Generation summary">
        <Metric label="Generation" value={snapshot ? `${snapshot.generation} / ${state?.generation_limit}` : "—"} />
        <Metric label="Best fitness" value={snapshot ? signed(snapshot.best_individual.fitness ?? 0) : "—"} detail={snapshot?.best_individual.id} />
        <Metric label="Average fitness" value={snapshot ? signedFloat(snapshot.average_fitness) : "—"} />
        <Metric label="Population" value={snapshot?.population.length ?? "—"} detail={snapshot ? `${snapshot.elite_individuals.length} elites preserved` : "Awaiting experiment"} />
      </section>
      <div className="-mt-4 mb-7 flex flex-wrap items-center justify-between gap-3">
        <p className="m-0 text-[10px] text-[#77776d]">
          {bestCandidate ? `Strongest in latest generation: ${bestCandidate.id} · fitness ${signed(bestCandidate.fitness ?? 0)}` : "Complete an evaluation to select its strongest candidate."}
        </p>
        <Button onClick={useBestCandidateInChess} disabled={!bestCandidate} className="gap-2">
          <Swords size={15} />
          {bestCandidate ? `Use ${bestCandidate.id} in Chess` : "Use in Chess"}
        </Button>
      </div>

      <div className="grid min-w-0 grid-cols-1 gap-6 xl:grid-cols-[minmax(0,1.05fr)_minmax(0,1fr)]">
        <section aria-labelledby="population-heading" className="min-w-0">
          <div className="flex flex-wrap items-start justify-between gap-2">
            <SectionHeading icon={<Activity size={16} />} eyebrow="CURRENT SNAPSHOT" title="Population" detail={snapshot ? `Generation ${snapshot.generation} · sorted by fitness` : "Individuals will appear after the experiment starts"} />
            <Button variant="outline" size="sm" onClick={saveSelectedIndividuals} disabled={archiveBusy || !selectedPopulationIds.length} className="text-[10px]">Save selected to Individual Bank</Button>
          </div>
          <div className="overflow-x-auto border-y border-[#d5cebf] bg-[#fbf9f3]">
            <table className="w-full min-w-[690px] border-collapse text-left text-xs">
              <thead className="bg-[#e8e4d9] text-[9px] uppercase tracking-[.12em] text-[#67675e]">
                <tr>
                  <th className="w-8 px-2 py-3"><span className="sr-only">Select</span></th><th className="px-3 py-3">Individual</th>
                  {geneKeys.map((gene) => <th key={gene} className="px-2 py-2" scope="col"><PieceMark gene={gene} /></th>)}
                  <th className="px-2 py-3">Fitness</th><th className="px-3 py-3">Record / status</th>
                </tr>
              </thead>
              <tbody>
                {sortedPopulation.map((individual) => {
                    const isBest = individual.fitness !== null && individual.fitness === bestFitnessInSnapshot;
                  const isElite = Boolean(individual.elite_from);
                  return (
                    <tr key={individual.id} className={`border-t border-[#e8e3d8] ${selectedIndividual?.id === individual.id ? "bg-[#eef0e6]" : "hover:bg-white"}`}>
                      <td className="px-2 py-2.5"><input aria-label={`Select ${individual.id} for Individual Bank`} type="checkbox" checked={selectedPopulationIds.includes(individual.id)} onChange={(event) => setSelectedPopulationIds((current) => event.target.checked ? [...current, individual.id] : current.filter((id) => id !== individual.id))} /></td>
                      <td className="px-3 py-2.5"><button type="button" onClick={() => setSelectedIndividualId(individual.id)} className="font-mono font-semibold text-[#343930] underline-offset-2 hover:underline">{individual.id}</button></td>
                      {geneKeys.map((gene) => <td key={gene} className="px-2 py-2.5 font-mono tabular-nums text-[#54564e]">{individual.weights[gene]}</td>)}
                      <td className="px-2 py-2.5 font-mono font-semibold tabular-nums">{signed(individual.fitness ?? 0)}</td>
                      <td className="px-3 py-2.5 text-[10px] text-[#747369]">{isBest ? <span className="mr-2 font-semibold text-[#627449]">{tiedBestCount > 1 ? "TIED BEST" : "BEST"}</span> : null}{isElite ? <span className="mr-2 font-semibold text-[#9a7040]">ELITE</span> : null}{individual.wins}W · {individual.draws}D · {individual.losses}L</td>
                    </tr>
                  );
                })}
                {!sortedPopulation.length && <tr><td colSpan={9} className="px-4 py-12 text-center text-sm text-[#858378]">Start an experiment to create and evaluate the first generation.</td></tr>}
              </tbody>
            </table>
          </div>
          {selectedIndividual && <div className="mt-3 border border-[#d5cebf] bg-[#fbf9f3] p-3"><Genome label="Selected individual chromosome" id={selectedIndividual.id} weights={selectedIndividual.weights} /></div>}
          {selectedIndividual && <div className="mt-3 flex flex-wrap items-center justify-between gap-2 text-[11px] text-[#6c6c62]"><span>Selected: <strong className="font-mono text-[#343930]">{selectedIndividual.id}</strong> · {selectedIndividual.wins} wins, {selectedIndividual.draws} draws, {selectedIndividual.losses} losses</span><span>Fitness = wins − losses</span></div>}
        </section>

        <EvolutionLineageVisualizer snapshot={visualizerSnapshot} previousSnapshot={previousSnapshot} selectedId={selectedChildId} onSelect={setSelectedChildId} />
      </div>

      <section aria-labelledby="individual-bank-heading" className="mt-7 border border-[#d5cebf] bg-[#fbf9f3] p-4">
        <div className="mb-3 flex flex-wrap items-start justify-between gap-3">
          <div>
            <p className="m-0 text-[9px] font-semibold uppercase tracking-[.15em] text-[#8c744b]">LONG-TERM GENOME STORAGE</p>
            <h2 id="individual-bank-heading" className="m-0 mt-1 text-sm font-semibold">Individual Bank</h2>
            <p className="mb-0 mt-1 text-[11px] text-[#77776d]">Saved chromosomes remain unchanged. Select saved genomes below to seed a new experiment. Checking a row selects it; use “Save selected to Individual Bank” above the population table to store it.</p>
          </div>
          <div className="flex flex-wrap gap-2">
            <Button variant="outline" onClick={() => void refreshArchives().catch((reason: unknown) => setError(messageFrom(reason)))} className="text-xs">Refresh Bank</Button>
            <Button onClick={startFromSelectedBank} disabled={busy || !selectedBankIds.length || selectedBankIds.length > config.population_size} className="text-xs">Create Experiment from Selected Individuals</Button>
          </div>
        </div>
        {archiveNotice && <p aria-live="polite" className="mb-3 text-xs font-medium text-[#52633e]">{archiveNotice}</p>}
        <div className="max-h-[28rem] overflow-auto border-y border-[#d5cebf]">
          <table className="w-full min-w-[900px] border-collapse text-left text-[11px]">
            <thead className="sticky top-0 bg-[#e8e4d9] text-[9px] uppercase tracking-[.12em] text-[#67675e]">
              <tr><th className="w-8 px-2 py-2"><span className="sr-only">Select</span></th><th className="px-2 py-2">Individual</th><th className="px-2 py-2">Source experiment / gen</th><th className="px-2 py-2">P / N / B / R / Q</th><th className="px-2 py-2">Fitness</th><th className="px-2 py-2">W / D / L · games</th><th className="px-2 py-2">Saved</th></tr>
            </thead>
            <tbody>
              {individualBank.map((item) => <tr key={item.bank_id} className="border-t border-[#e5dfd2]">
                <td className="px-2 py-2"><input aria-label={`Select bank individual ${item.individual_id} from ${item.source_experiment_id}`} type="checkbox" checked={selectedBankIds.includes(item.bank_id)} onChange={(event) => setSelectedBankIds((current) => event.target.checked ? [...current, item.bank_id] : current.filter((id) => id !== item.bank_id))} /></td>
                <td className="px-2 py-2 font-mono font-semibold">{item.individual_id}<small className="block font-sans font-normal text-[#858378]">{item.tags.join(", ") || item.notes}</small></td>
                <td className="px-2 py-2 font-mono">{item.source_experiment_id}<small className="block font-sans text-[#858378]">Generation {item.source_generation}</small></td>
                <td className="px-2 py-2 font-mono tabular-nums">{[item.chromosome.pawn, item.chromosome.knight, item.chromosome.bishop, item.chromosome.rook, item.chromosome.queen].join(" / ")}</td>
                <td className="px-2 py-2 font-mono">{item.fitness === null ? "—" : signed(item.fitness)}</td>
                <td className="px-2 py-2 font-mono">{item.wins} / {item.draws} / {item.losses} · {item.games_played}</td>
                <td className="px-2 py-2 text-[#77776d]">{new Date(item.created_at).toLocaleString()}</td>
              </tr>)}
              {!individualBank.length && <tr><td colSpan={7} className="px-3 py-8 text-center text-sm text-[#858378]">No saved individuals yet. Select population rows and save them, or save every tied-best individual.</td></tr>}
            </tbody>
          </table>
        </div>
        {selectedBankIds.length > config.population_size && <p role="alert" className="mb-0 mt-2 text-xs text-[#9b5144]">Selected bank individuals exceed the configured population size.</p>}
      </section>

      <div className="mt-7 grid grid-cols-1 gap-6 xl:grid-cols-[minmax(0,1.4fr)_minmax(320px,.8fr)]">
        <section aria-labelledby="fitness-heading" className="min-w-0">
          <SectionHeading icon={<Activity size={16} />} eyebrow="FITNESS OVER TIME" title="Generation history" detail="Best and average candidate fitness · higher is better" />
          <FitnessChart history={state?.history ?? []} />
          <div className="mt-3 flex flex-wrap gap-2" aria-label="Generation timeline">
            {(state?.history ?? []).map((step) => <button key={step.generation} type="button" aria-pressed={snapshot?.generation === step.generation} onClick={() => { setSelectedGeneration(step.generation); setSelectedIndividualId(null); setSelectedChildId(null); }} className={`border px-3 py-2 font-mono text-[10px] transition-colors ${snapshot?.generation === step.generation ? "border-[#48563c] bg-[#48563c] text-white" : "border-[#d5cebf] bg-[#fbf9f3] text-[#606057] hover:bg-white"}`}>GEN {String(step.generation).padStart(2, "0")}</button>)}
            {!state?.history.length && <p className="m-0 text-xs text-[#858378]">No generations recorded yet.</p>}
          </div>
        </section>

        <section aria-labelledby="learning-heading" className="border-l-2 border-[#a87643] bg-[#f7f3e9] px-5 py-4">
          <SectionHeading icon={<Dna size={16} />} eyebrow="EDUCATIONAL MODE" title="What happened?" detail={snapshot ? `Generation ${snapshot.generation} snapshot` : "A short explanation based on experiment events"} />
          {snapshot ? (
            <>
              <ul className="mt-4 space-y-2 pl-4 text-xs leading-5 text-[#55574f]">
                <li>Best fitness: <strong>{signed(snapshot.best_individual.fitness ?? 0)}</strong> ({snapshot.best_individual.id}).</li>
                <li>Average fitness: <strong>{signedFloat(snapshot.average_fitness)}</strong>.</li>
                <li>Elite individuals preserved: <strong>{snapshot.elite_individuals.length}</strong>.</li>
                <li>Parents selected: <strong>{snapshot.selected_parents.length}</strong> distinct individuals.</li>
                <li>Children created: <strong>{snapshot.children.length}</strong>.</li>
                <li>Recorded mutations: <strong>{snapshot.children.reduce((total, child) => total + child.mutations.length, 0)}</strong>.</li>
              </ul>
              {snapshot.generation === 1 ? <p className="mb-0 mt-4 border-t border-[#ded5c4] pt-3 text-xs leading-5 text-[#737166]">The first population is sampled around baseline material values. Each candidate always faces BASELINE plus random, unique opponents from the population, with both colors played.</p> : <p className="mb-0 mt-4 border-t border-[#ded5c4] pt-3 text-xs leading-5 text-[#737166]">The strongest individuals survive unchanged. Tournament selection chooses parents, uniform crossover selects each gene source independently, and mutation perturbs selected genes within their allowed bounds. Each new candidate faces BASELINE plus random, unique opponents sampled from the previous generation's five strongest individuals.</p>}
              <p className="mb-0 mt-3 border-t border-[#ded5c4] pt-3 text-xs leading-5 text-[#737166]">BASELINE is one reference opponent, not ground truth. Candidates are evaluated against multiple opponents and their results are aggregated. Fitness measures performance across this selected pool; it is not a universal measure of chess strength.</p>
            </>
          ) : <p className="mb-0 mt-4 text-xs leading-5 text-[#737166]">BASELINE is one reference opponent, not ground truth. Each candidate plays a deterministic opponent pool with both colors where practical. Wins add one fitness point, losses subtract one, and draws leave fitness unchanged. Fitness measures performance across this selected pool; it is not a universal measure of chess strength.</p>}
        </section>
      </div>

      <footer className="mt-9 flex flex-wrap justify-between gap-2 border-t border-[#d5cebf] pt-4 text-[9px] uppercase tracking-[.12em] text-[#89877d]">
        <span>Experimental weights only · production chess evaluation remains unchanged</span>
        <span>Pawn · Knight · Bishop · Rook · Queen</span>
      </footer>
    </main>
  );

}

function NumberControl({
  label,
  value,
  min,
  max,
  step = 1,
  onChange,
}: {
  label: string;
  value: number | "";
  min: number;
  max: number;
  step?: number;
  onChange: (value: number) => void;
}) {
  const [draftValue, setDraftValue] = useState(String(value));
  const [focused, setFocused] = useState(false);

  useEffect(() => {
    if (!focused) setDraftValue(String(value));
  }, [focused, value]);

  const commitDraft = () => {
    const parsed = draftValue === "" ? Number(value) : Number(draftValue);
    const next = Math.max(min, Math.min(max, Number.isFinite(parsed) ? parsed : min));
    setDraftValue(String(next));
    onChange(next);
    setFocused(false);
  };

  return (
    <label className="flex min-w-0 flex-col gap-1.5 text-[10px] font-medium text-[#6c6b61]">
      {label}
      <input
        type="number"
        value={draftValue}
        min={min}
        max={max}
        step={step}
        onFocus={() => setFocused(true)}
        onChange={(event) => {
          const next = event.target.value;
          setDraftValue(next);
          if (next === "") return;
          const parsed = Number(next);
          if (Number.isFinite(parsed)) onChange(Math.max(min, Math.min(max, parsed)));
        }}
        onBlur={commitDraft}
        className="h-9 min-w-0 rounded-md border border-[#d5cebf] bg-[#fbf9f3] px-2 font-mono text-xs text-[#343930] outline-none focus-visible:ring-2 focus-visible:ring-amber-700"
      />
    </label>
  );
}

type ProgressEvent = Extract<GeneticEvent, {
  type: "candidate_started" | "game_started" | "move_played" | "game_finished" | "candidate_game_finished" | "candidate_finished";
}>;

function LiveSelfPlayMonitor({
  traces,
  events,
  progress,
  active,
  followLive,
  replay,
  displayedTrace,
  displayedFen,
  displayedPly,
  displayedMove,
  onPause,
  onFollow,
  onSelectReplay,
  onSelectPly,
  onPrevious,
  onNext,
}: {
  traces: CandidateGameTrace[];
  events: GeneticEvent[];
  progress: GeneticEvent | null;
  active: boolean;
  followLive: boolean;
  replay: { candidate_id: string; game_index: number; ply: number } | null;
  displayedTrace: CandidateGameTrace | null;
  displayedFen: string;
  displayedPly: number;
  displayedMove: CandidateGameTrace["moves"][number] | null;
  onPause: () => void;
  onFollow: () => void;
  onSelectReplay: (trace: CandidateGameTrace) => void;
  onSelectPly: (ply: number) => void;
  onPrevious: () => void;
  onNext: () => void;
}) {
  const progressCandidate = [...events].reverse().find((event): event is ProgressEvent =>
    event.type === "candidate_started" || event.type === "game_started" || event.type === "move_played"
      || event.type === "game_finished" || event.type === "candidate_game_finished" || event.type === "candidate_finished",
  );
  const total = progressCandidate?.candidates_total ?? (progress?.type === "evaluation_started" ? progress.candidates_total : 0);
  const completed = progressCandidate?.completed_candidates ?? (progress?.type === "evaluation_started" ? progress.completed_candidates : 0);
  const percent = total ? Math.min(100, completed / total * 100) : 0;
  const activeGame = [...events].reverse().find((event) => event.type === "game_started" && event.candidate_id === progressCandidate?.candidate_id);
  const currentGameIndex = progressCandidate && "game_index" in progressCandidate ? progressCandidate.game_index : activeGame?.type === "game_started" ? activeGame.game_index : 0;
  const currentGamesTotal = activeGame?.type === "game_started" ? activeGame.games_total : 0;
  const candidateResult = [...events].reverse().find((event) => event.type === "candidate_finished" && event.candidate_id === progressCandidate?.candidate_id);
  const latestFitness = [...events].reverse().find((event) => event.type === "candidate_game_finished" && event.candidate_id === displayedTrace?.candidate_id);
  const completionFitness = [...events].reverse().find((event) => event.type === "candidate_finished" && event.candidate_id === displayedTrace?.candidate_id);
  const fitness = completionFitness?.type === "candidate_finished" ? completionFitness.fitness : latestFitness?.type === "candidate_game_finished" ? latestFitness.fitness : null;
  const candidateCount = events.filter((event) => event.type === "candidate_started").length;
  const generation = progressCandidate?.generation ?? (progress?.type === "evaluation_started" ? progress.generation : 0);
  const resultLabel = displayedTrace?.result?.toUpperCase() ?? (displayedTrace ? "IN PROGRESS" : "WAITING");
  const candidateGames = displayedTrace ? traces
    .filter((trace) => trace.candidate_id === displayedTrace.candidate_id)
    .sort((left, right) => left.game_index - right.game_index) : [];

  return (
    <section aria-labelledby="live-self-play-heading" className="mb-7 border border-[#d5cebf] bg-[#fbf9f3]">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#d5cebf] bg-[#f0ede4] px-4 py-3 sm:px-5">
        <div className="flex items-center gap-2.5">
          <Activity size={16} className="text-[#8c744b]" />
          <div>
            <h2 id="live-self-play-heading" className="m-0 text-xs font-semibold tracking-[.1em]">LIVE SELF-PLAY</h2>
            <p className="m-0 mt-0.5 text-[10px] text-[#77776d]">Candidate vs opponent pool · real moves from Alpha-Beta search</p>
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {followLive ? (
            <Button variant="secondary" size="sm" onClick={onPause} disabled={!displayedTrace} className="gap-1.5"><Pause size={13} /> Pause replay</Button>
          ) : (
            <Button variant="secondary" size="sm" onClick={onFollow} className="gap-1.5"><Activity size={13} /> Follow Live</Button>
          )}
          <Button variant="outline" size="sm" onClick={onPrevious} disabled={!displayedTrace || displayedPly <= 0}>Previous Move</Button>
          <Button variant="outline" size="sm" onClick={onNext} disabled={!displayedTrace || displayedPly >= displayedTrace.moves.length}>Next Move</Button>
        </div>
      </div>

      {active && (
        <div className="border-b border-[#d5cebf] px-4 py-3 sm:px-5">
          <div className="mb-2 flex flex-wrap justify-between gap-2 text-[10px] text-[#66665d]">
            <span>Evaluating population · Generation {generation || "—"}</span>
            <span>Candidate {progressCandidate?.candidate_index ?? completed} / {total || "—"} · Game {currentGameIndex || "—"} / {currentGamesTotal || "—"}</span>
          </div>
          <div className="h-2 overflow-hidden rounded-full bg-[#e1dccf]" role="progressbar" aria-label="Candidates completed" aria-valuemin={0} aria-valuemax={total || 1} aria-valuenow={completed}>
            <div className="h-full bg-[#74845c] transition-[width] duration-300 motion-reduce:transition-none" style={{ width: `${percent}%` }} />
          </div>
          <p className="mb-0 mt-1.5 text-[9px] text-[#858378]">{completed} / {total || "—"} candidates completed · {candidateCount} started</p>
        </div>
      )}

      <div className="grid gap-4 p-4 lg:grid-cols-[minmax(280px,.8fr)_minmax(320px,1fr)_minmax(220px,.7fr)] sm:p-5">
        <div className="space-y-3">
          <div className="flex flex-wrap items-end justify-between gap-2">
            <div><p className="m-0 text-[9px] font-semibold uppercase tracking-[.14em] text-[#8c744b]">Current candidate</p><p className="mb-0 mt-1 font-mono text-xl font-semibold">{displayedTrace?.candidate_id ?? progressCandidate?.candidate_id ?? "—"}</p></div>
            <p className="m-0 text-xs text-[#66665d]">Fitness <strong className="font-mono text-sm text-[#343930]">{fitness === null ? "pending" : signed(fitness)}</strong></p>
          </div>
          {displayedTrace ? (
            <>
              <div className="flex flex-wrap gap-2 text-[10px]">
                <span className="border border-[#d5cebf] bg-white px-2 py-1">Game {displayedTrace.game_index} / {displayedTrace.games_total}</span>
                <span className="border border-[#d5cebf] bg-white px-2 py-1">Candidate: {displayedTrace.candidate_color.toUpperCase()}</span>
                <span className="border border-[#d5cebf] bg-white px-2 py-1">{displayedTrace.candidate_id} vs {displayedTrace.opponent_id}</span>
                <span className={`border px-2 py-1 font-semibold ${displayedTrace.result === "win" ? "border-[#bec9ad] bg-[#eef1e8] text-[#52653d]" : displayedTrace.result === "loss" ? "border-[#dec3b8] bg-[#f6ebe6] text-[#8a5143]" : "border-[#d8d1bf] bg-[#f2efe7] text-[#75684c]"}`}>{resultLabel}</span>
              </div>
              <div className="border border-[#d5cebf] bg-white p-3">
                <p className="m-0 mb-2 text-[9px] font-semibold uppercase tracking-[.14em] text-[#747267]">Candidate chromosome</p>
                <MiniChromosome weights={replayCandidateWeights(displayedTrace)} />
                <p className="mb-0 mt-3 border-t border-[#e5dfd2] pt-2 text-[9px] leading-4 text-[#77776d]">
                  {displayedTrace.opponent_type} opponent weights: {geneKeys.map((gene) => `${geneLabels[gene]} ${displayedTrace.opponent_weights[gene]}`).join(" · ")}
                </p>
              </div>
              <div className="border border-[#d5cebf] bg-white p-3">
                <p className="m-0 mb-2 text-[9px] font-semibold uppercase tracking-[.14em] text-[#747267]">Fitness matchup breakdown</p>
                <div className="space-y-1">
                  {candidateGames.map((trace) => <div key={trace.game_index} className="grid grid-cols-[1fr_auto_auto_auto] gap-3 font-mono text-[9px]">
                    <span>vs {trace.opponent_id} · {trace.candidate_color}</span>
                    <span>{trace.result?.toUpperCase() ?? "LIVE"}</span>
                    <span>{trace.termination?.replaceAll("_", " ") ?? "pending"}</span>
                    <span className="text-right">{trace.fitness_delta > 0 ? "+" : ""}{trace.fitness_delta}</span>
                  </div>)}
                </div>
                <div className="mt-2 flex justify-between border-t border-[#e5dfd2] pt-2 text-[9px] font-semibold">
                  <span>Aggregate fitness</span>
                  <span className="font-mono">{fitness === null ? "pending" : signed(fitness)}</span>
                </div>
              </div>
              <div className="flex justify-between border-t border-[#d5cebf] pt-2 text-[10px] text-[#66665d]">
                <span>Candidate score from games</span>
                <strong className="font-mono">{displayedTrace.fitness_delta > 0 ? "+" : ""}{displayedTrace.fitness_delta}</strong>
              </div>
              <p className="m-0 text-[9px] text-[#858378]">Fitness = wins − losses. Each completed game contributes {"+1 / 0 / −1"}.</p>
            </>
          ) : <p className="m-0 text-xs leading-5 text-[#77776d]">Start an experiment to watch each candidate play the opponent pool.</p>}
        </div>

        <div>
          <ChessReplayBoard fen={displayedFen} lastUci={displayedMove?.uci ?? null} />
          <div className="mt-2 flex flex-wrap items-center justify-between gap-2 border-y border-[#d5cebf] py-2 text-[10px]">
            <span>Move {displayedPly} / {displayedTrace?.plies || displayedTrace?.moves.length || "—"}</span>
            <span>Last move: <strong className="font-mono">{displayedMove ? `${displayedMove.move_number}${displayedMove.color === "black" ? "..." : "."} ${displayedMove.san}` : "—"}</strong></span>
          </div>
          {displayedMove && <div className="mt-2 grid grid-cols-3 gap-2 text-[9px] text-[#77776d]">
            <span>Eval <strong className="block font-mono text-xs text-[#343930]">{displayedMove.evaluation > 0 ? "+" : ""}{displayedMove.evaluation.toFixed(2)}</strong></span>
            <span>Observed depth <strong className="block font-mono text-xs text-[#343930]">{displayedMove.depth}</strong></span>
            <span>Nodes <strong className="block font-mono text-xs text-[#343930]">{displayedMove.nodes.toLocaleString()}</strong></span>
          </div>}
          <div className="mt-2 max-h-24 overflow-y-auto border border-[#e1dccf] bg-white p-2" aria-label="Move history">
            {displayedTrace?.moves.length ? <ol className="m-0 flex flex-wrap gap-x-3 gap-y-1 pl-5 font-mono text-[10px]">{displayedTrace.moves.map((move, index) => <li key={`${move.ply}-${move.uci}`}><button type="button" onClick={() => onSelectPly(index + 1)} aria-current={displayedPly === index + 1 ? "step" : undefined} className="rounded px-1 hover:bg-[#eeeade] aria-[current=step]:bg-[#48563c] aria-[current=step]:text-white">{move.color === "white" ? `${move.move_number}. ` : `${move.move_number}... `}{move.san}</button></li>)}</ol> : <p className="m-0 text-center text-[10px] text-[#89877d]">Moves will appear here as the backend searches.</p>}
          </div>
        </div>

        <div className="space-y-3">
          <div>
            <p className="m-0 text-[9px] font-semibold uppercase tracking-[.14em] text-[#8c744b]">Game history · this evaluation</p>
            <div className="mt-2 max-h-64 space-y-1.5 overflow-y-auto pr-1">
              {traces.length ? traces.map((trace) => <button key={`${trace.candidate_id}-${trace.game_index}`} type="button" onClick={() => onSelectReplay(trace)} className={`flex w-full items-center justify-between gap-2 border px-2.5 py-2 text-left text-[10px] ${replay?.candidate_id === trace.candidate_id && replay.game_index === trace.game_index ? "border-[#48563c] bg-[#eef0e6]" : "border-[#ded7c7] bg-white hover:bg-[#f3f0e8]"}`}>
                <span><strong className="font-mono">{trace.candidate_id}</strong> · Game {trace.game_index}<small className="block text-[#77776d]">Candidate {trace.candidate_color} · {trace.plies} plies · {trace.termination?.replaceAll("_", " ") ?? "live"}</small></span>
                <span className="text-right font-semibold">{trace.result?.toUpperCase() ?? "LIVE"}<small className="block font-mono">{trace.fitness_delta > 0 ? "+" : ""}{trace.fitness_delta}</small></span>
              </button>) : <p className="m-0 text-[10px] text-[#89877d]">Completed games appear here and can be replayed.</p>}
            </div>
          </div>
          <div className="border-t border-[#d5cebf] pt-3">
            <p className="m-0 text-[9px] font-semibold uppercase tracking-[.14em] text-[#8c744b]">Fitness connection</p>
            <p className="mb-0 mt-2 text-[10px] leading-5 text-[#66665d]">Chromosome → self-play → game result → fitness</p>
            <p className="mb-0 mt-1 text-[10px] leading-5 text-[#77776d]">{candidateResult?.type === "candidate_finished" ? `${candidateResult.record.wins} wins − ${candidateResult.record.losses} losses = ${signed(candidateResult.fitness)} fitness.` : "Each game adds +1 for a win, 0 for a draw, and −1 for a loss."}</p>
          </div>
        </div>
      </div>
      {!active && !traces.length && <p className="mb-4 px-5 text-[10px] text-[#858378]">The live monitor stays idle until an evaluation actually runs; no sample moves are shown.</p>}
    </section>
  );
}

const pieceGlyphs: Record<string, string> = {
  P: "♙", N: "♘", B: "♗", R: "♖", Q: "♕", K: "♔",
  p: "♟", n: "♞", b: "♝", r: "♜", q: "♛", k: "♚",
};

function ChessReplayBoard({ fen, lastUci }: { fen: string; lastUci: string | null }) {
  const ranks = fen.split(" ")[0].split("/");
  const squares = ranks.flatMap((rank, row) => {
    const rankPieces: (string | null)[] = [];
    for (const token of rank) {
      if (/\d/.test(token)) rankPieces.push(...Array(Number(token)).fill(null));
      else rankPieces.push(token);
    }
    return rankPieces.slice(0, 8).map((piece, col) => ({ piece, row, col, square: `${"abcdefgh"[col]}${8 - row}` }));
  });
  const from = lastUci?.slice(0, 2);
  const to = lastUci?.slice(2, 4);
  return <div className="mx-auto grid aspect-square w-full max-w-[400px] grid-cols-8 overflow-hidden border-2 border-[#554b3e]" style={{ gridTemplateRows: "repeat(8, minmax(0, 1fr))" }} role="img" aria-label="Chessboard replay position">
    {squares.map(({ piece, row, col, square }) => <div key={square} className={`relative grid min-h-0 min-w-0 place-items-center text-[clamp(1.3rem,5vw,2.3rem)] leading-none ${((row + col) % 2) ? "bg-[#84906e]" : "bg-[#f0e8d5]"} ${square === from || square === to ? "after:absolute after:inset-0 after:bg-[#c69a4a]/30" : ""}`} aria-label={piece ? `${piece === piece.toUpperCase() ? "White" : "Black"} ${pieceNames[piece.toLowerCase()]}` : square}>
      {piece && <span className={`relative z-[1] ${piece === piece.toUpperCase() ? "text-white [text-shadow:0_1px_2px_#343930,0_0_2px_#343930]" : "text-[#343930] [text-shadow:0_1px_1px_#f8f4e9]"}`}>{pieceGlyphs[piece]}</span>}
      {col === 0 && <span className="absolute left-0.5 top-0.5 z-[2] text-[8px] font-semibold opacity-70">{8 - row}</span>}
      {row === 7 && <span className="absolute bottom-0 right-1 z-[2] text-[8px] font-semibold opacity-70">{"abcdefgh"[col]}</span>}
    </div>)}
  </div>;
}

export function EvolutionLineageVisualizer({
  snapshot,
  previousSnapshot,
  selectedId,
  onSelect,
}: {
  snapshot: EvolutionLineageSnapshot | null;
  previousSnapshot: EvolutionStep | undefined;
  selectedId: string | null;
  onSelect: (id: string) => void;
}) {
  const lineages = snapshot ? lineageEntriesForStep(snapshot) : [];
  const lineage = lineages.find((entry) => entry.id === selectedId)
    ?? lineages.find((entry) => entry.kind === "offspring")
    ?? lineages[0]
    ?? null;
  const parentPopulation = previousSnapshot?.population ?? [];
  const detail = lineage
    ? `${lineage.kind === "elite_clone" ? "Elite clone" : lineage.kind === "bank_seed" ? "Bank seed" : "Offspring"} ${lineage.id} · generation ${lineage.generation}`
    : "Advance a generation to inspect offspring and elite lineage";

  return (
    <section aria-labelledby="evolution-heading" className="min-w-0">
      <SectionHeading icon={<GitBranch size={16} />} eyebrow="GENETIC OPERATORS" title="Evolution visualizer" detail={detail} />
      {lineage ? (
        <div className="space-y-4 border-y border-[#d5cebf] bg-[#fbf9f3] p-4 sm:p-5">
          {lineage.kind === "offspring" ? (
            <>
              <p className="m-0 text-xs font-semibold">{lineage.id} · offspring</p>
              <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
                <ParentPanel label="Parent A" id={lineage.parent_a_id} individual={individualById(parentPopulation, lineage.parent_a_id) ?? undefined} />
                <ParentPanel label="Parent B" id={lineage.parent_b_id} individual={individualById(parentPopulation, lineage.parent_b_id) ?? undefined} />
              </div>
              <div className="flex flex-wrap items-center gap-2 text-[9px] font-semibold uppercase tracking-[.15em] text-[#8c744b]"><span className="h-px min-w-4 flex-1 bg-[#d7ccb7]" /><span>Crossover · pre-mutation</span><span className="h-px flex-1 bg-[#d7ccb7]" /></div>
              <GeneStrip weights={lineage.pre_mutation_weights} origins={lineage.gene_origins} />
              <div className="flex items-center gap-2 text-[9px] font-semibold uppercase tracking-[.15em] text-[#8c744b]"><span className="h-px flex-1 bg-[#d7ccb7]" /><span>Mutation</span><span className="h-px flex-1 bg-[#d7ccb7]" /></div>
              {lineage.mutations.length ? (
                <ul className="m-0 flex flex-wrap gap-2 p-0" aria-label="Mutation events">
                  {lineage.mutations.map((mutation, index) => <li key={`${mutation.gene}-${index}`} className="list-none border border-[#d7c8ad] bg-[#f3ede0] px-3 py-2 text-xs"><strong>{geneLabels[mutation.gene]}</strong> {mutation.old_value} → {mutation.new_value} <span className="font-mono font-semibold text-[#8a5f31]">({mutation.delta > 0 ? "+" : ""}{mutation.delta})</span></li>)}
                </ul>
              ) : <p className="m-0 text-xs text-[#78776d]">No genes mutated in this offspring.</p>}
              <p className="m-0 text-[11px] leading-5 text-[#68685f]">{crossoverExplanation(lineage.gene_origins)} {mutationExplanation(lineage.mutations)}</p>
            </>
          ) : lineage.kind === "elite_clone" ? (
            <>
              <p className="m-0 text-xs font-semibold">{lineage.id} · elite clone</p>
              <ParentPanel label="Copied from" id={lineage.parent_a_id} individual={individualById(parentPopulation, lineage.parent_a_id) ?? undefined} />
              <p className="m-0 text-[11px] leading-5 text-[#68685f]">This elite carries the parent chromosome forward unchanged; crossover and mutation were not applied.</p>
            </>
          ) : (
            <>
              <p className="m-0 text-xs font-semibold">{lineage.id} · Individual Bank seed</p>
              <p className="m-0 text-[11px] leading-5 text-[#68685f]">
                Preserved from {lineage.source_individual_id} in experiment {lineage.source_experiment_id}, generation {lineage.source_generation}. The saved chromosome is copied unchanged and evaluated in this experiment.
              </p>
              {lineage.individual.bank_source_lineage && <details className="text-[10px] text-[#68685f]"><summary className="cursor-pointer">View source lineage</summary><pre className="max-h-40 overflow-auto whitespace-pre-wrap">{JSON.stringify(lineage.individual.bank_source_lineage, null, 2)}</pre></details>}
            </>
          )}
          <div className="border-t border-[#e1dbce] pt-3"><Genome label={lineage.kind === "offspring" ? "Final chromosome after mutation" : lineage.kind === "elite_clone" ? "Final elite chromosome" : "Initial bank seed chromosome"} id={lineage.id} weights={lineage.post_mutation_weights} /></div>
          {lineages.length > 1 && <label className="flex items-center gap-2 text-xs text-[#64645a]">Inspect individual <select aria-label="Inspect individual lineage" value={lineage.id} onChange={(event) => onSelect(event.target.value)} className="rounded-md border border-[#d5cebf] bg-white px-2 py-1">{lineages.map((entry) => <option key={entry.id} value={entry.id}>{entry.id} · {entry.kind === "elite_clone" ? "elite clone" : entry.kind === "bank_seed" ? "bank seed" : "offspring"}</option>)}</select></label>}
        </div>
      ) : (
        <div className="grid min-h-52 place-items-center border-y border-[#d5cebf] bg-[#fbf9f3] px-6 text-center text-sm text-[#77776d]">{snapshot ? "This generation has no offspring or elite lineage to display." : "Start the experiment, then advance a generation to inspect its actual parents, gene inheritance, and mutations."}</div>
      )}
    </section>
  );
}

const pieceNames: Record<string, string> = { p: "pawn", n: "knight", b: "bishop", r: "rook", q: "queen", k: "king" };

function Metric({ label, value, detail }: { label: string; value: string | number; detail?: string }) {
  return <div className="border-l border-[#d5cebf] pl-4 first:border-0"><p className="m-0 text-[9px] font-semibold uppercase tracking-[.14em] text-[#89877d]">{label}</p><p className="mb-0 mt-1 font-mono text-xl font-semibold tabular-nums text-[#343930]">{value}</p>{detail && <p className="mb-0 mt-0.5 text-[10px] text-[#78776d]">{detail}</p>}</div>;
}

function SectionHeading({ icon, eyebrow, title, detail }: { icon: ReactNode; eyebrow: string; title: string; detail: string }) {
  return <div className="mb-3 flex items-start gap-2.5"><span className="mt-0.5 text-[#8c744b]">{icon}</span><div><p className="m-0 text-[9px] font-semibold uppercase tracking-[.15em] text-[#89877d]">{eyebrow}</p><h2 className="m-0 mt-0.5 font-display text-lg font-semibold">{title}</h2><p className="mb-0 mt-1 text-[11px] text-[#77776d]">{detail}</p></div></div>;
}

function ParentPanel({ label, id, individual }: { label: string; id: string; individual?: Individual }) {
  return <div className="border border-[#ddd5c6] bg-white/70 p-3"><p className="m-0 text-[9px] font-semibold uppercase tracking-[.14em] text-[#8c744b]">{label} · {id}</p>{individual ? <><MiniChromosome weights={individual.weights} /><p className="mb-0 mt-2 text-[10px] text-[#77776d]">Fitness {signed(individual.fitness ?? 0)} · {individual.wins}W / {individual.draws}D / {individual.losses}L</p></> : <p className="mb-0 mt-3 text-xs text-[#89877d]">Parent snapshot not available.</p>}</div>;
}

function Genome({ label, id, weights }: { label: string; id: string; weights: PieceWeights }) {
  return <div><div className="mb-2 flex items-center justify-between gap-2"><p className="m-0 text-[9px] font-semibold uppercase tracking-[.14em] text-[#747267]">{label}</p><span className="font-mono text-[10px] text-[#77776d]">{id}</span></div><div className="grid grid-cols-5 gap-1.5">{geneKeys.map((gene) => <div key={gene} className="border border-[#dcd5c7] bg-white px-2 py-2 text-center"><PieceMark gene={gene} /><span className="mt-1 block font-mono text-xs font-semibold tabular-nums">{weights[gene]}</span></div>)}</div></div>;
}

function MiniChromosome({ weights }: { weights: PieceWeights }) {
  return <div className="mt-3 grid grid-cols-5 gap-1">{geneKeys.map((gene) => <div key={gene} className="bg-[#f1eee6] px-1 py-1.5 text-center"><PieceMark gene={gene} compact /><span className="font-mono text-[10px]">{weights[gene]}</span></div>)}</div>;
}

function GeneStrip({ weights, origins }: { weights: PieceWeights; origins: EvolutionStep["children"][number]["gene_origins"] }) {
  const sourceByGene = new Map(origins.map((origin) => [origin.gene, origin.source_parent]));
  return <div className="grid grid-cols-5 gap-1.5">{geneKeys.map((gene) => { const source = sourceByGene.get(gene); return <div key={gene} className="border border-[#dcd5c7] bg-white px-1.5 py-2 text-center"><PieceMark gene={gene} /><span className="mt-1 block font-mono text-xs font-semibold">{weights[gene]}</span><span className="mt-1 block text-[8px] font-semibold text-[#68734e]">{source === "parent_a" ? "Parent A" : "Parent B"}</span></div>; })}</div>;
}

function PieceMark({ gene, compact = false }: { gene: keyof PieceWeights; compact?: boolean }) {
  return <span role="img" aria-label={geneLabels[gene]} title={geneLabels[gene]} className={`block font-serif leading-none text-[#48563c] ${compact ? "text-sm" : "text-lg"}`}>
    {geneSymbols[gene]}
  </span>;
}

function FitnessChart({ history }: { history: EvolutionStep[] }) {
  const width = 720;
  const height = 190;
  const padX = 38;
  const padY = 22;
  const values = history.flatMap((step) => [step.best_individual.fitness ?? 0, step.average_fitness]);
  const min = Math.min(0, ...values);
  const max = Math.max(1, ...values);
  const range = max - min || 1;
  const x = (index: number) => history.length <= 1 ? width / 2 : padX + index * (width - padX * 2) / (history.length - 1);
  const y = (value: number) => height - padY - (value - min) * (height - padY * 2) / range;
  const best = history.map((step, index) => `${index ? "L" : "M"}${x(index)},${y(step.best_individual.fitness ?? 0)}`).join(" ");
  const average = history.map((step, index) => `${index ? "L" : "M"}${x(index)},${y(step.average_fitness)}`).join(" ");
  const summary = history.length ? `${history.length} generations. Best fitness ranges from ${min} to ${max}.` : "No fitness data yet.";
  return <div className="border-y border-[#d5cebf] bg-[#fbf9f3] px-2 py-3" role="img" aria-label={`Fitness chart. ${summary}`}>
    <svg viewBox={`0 0 ${width} ${height}`} className="h-48 w-full" aria-hidden="true">
      {[0, 1, 2, 3].map((index) => { const yPos = padY + index * (height - padY * 2) / 3; return <g key={index}><line x1={padX} x2={width - padX} y1={yPos} y2={yPos} stroke="#e2ddd2" strokeWidth="1" /><text x="5" y={yPos + 3} fill="#858378" fontSize="8">{(max - index * range / 3).toFixed(0)}</text></g>; })}
      {history.length > 0 && <><path d={best} fill="none" stroke="#48563c" strokeWidth="2.5" />{history.length > 1 && <path d={average} fill="none" stroke="#a87643" strokeWidth="2" strokeDasharray="5 4" />}{history.map((step, index) => <circle key={step.generation} cx={x(index)} cy={y(step.best_individual.fitness ?? 0)} r="3.5" fill="#48563c" />)}</>}
      <text x={padX} y={height - 3} fill="#858378" fontSize="9">GEN 1</text><text x={width - padX} y={height - 3} textAnchor="end" fill="#858378" fontSize="9">GEN {history.at(-1)?.generation ?? 0}</text>
    </svg>
    <div className="flex flex-wrap gap-4 px-3 text-[10px] text-[#6c6b61]"><span className="inline-flex items-center gap-1.5"><i className="size-2 rounded-full bg-[#48563c]" />Best fitness</span><span className="inline-flex items-center gap-1.5"><i className="size-2 rounded-full bg-[#a87643]" />Average fitness</span><span className="sr-only">{summary}</span></div>
  </div>;
}

function crossoverExplanation(origins: EvolutionStep["children"][number]["gene_origins"]): string {
  const fromA = origins.filter((origin) => origin.source_parent === "parent_a").map((origin) => geneLabels[origin.gene].toLowerCase());
  const fromB = origins.filter((origin) => origin.source_parent === "parent_b").map((origin) => geneLabels[origin.gene].toLowerCase());
  return `Crossover inherited ${fromA.length ? fromA.join(", ") : "no genes"} from Parent A and ${fromB.length ? fromB.join(", ") : "no genes"} from Parent B.`;
}

function mutationExplanation(mutations: EvolutionStep["children"][number]["mutations"]): string {
  if (!mutations.length) return " No genes changed during mutation.";
  return ` Mutation changed ${mutations.map((mutation) => `${geneLabels[mutation.gene]} ${mutation.delta > 0 ? "+" : ""}${mutation.delta}`).join(", ")}.`;
}

function signed(value: number): string {
  return value > 0 ? `+${value}` : String(value);
}

function signedFloat(value: number): string {
  return value > 0 ? `+${value.toFixed(1)}` : value.toFixed(1);
}

function messageFrom(reason: unknown): string {
  return reason instanceof Error ? reason.message : "The laboratory request failed.";
}
