# Chess AI Service

A Python FastAPI chess service with an explicit **INPUT → PROCESSING → OUTPUT** flow:

```text
JSON request → FastAPI route → search controller → Minimax or Alpha-Beta → JSON response
```

The `minimax` endpoint uses a material evaluation and exhaustive minimax search. The `alpha-beta` endpoint uses the same evaluation and move order with alpha-beta pruning. Both use `python-chess` to parse FEN positions and generate legal moves. The legacy input spelling `alpha_beta` is also accepted and normalized to `alpha-beta` in the response. A terminal position has no move to return, so the endpoint responds with HTTP 422.

## Install and run

From this directory, create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
uvicorn app.main:app --reload
```

The service listens at `http://127.0.0.1:8000`.

## Health check

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Response:

```json
{"status":"ok"}
```

## AI move endpoint

`POST /api/ai/move`

Example request:

```json
{
  "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
  "algorithm": "alpha-beta",
  "depth": 2
}
```

`fen` must be non-empty, `algorithm` must be `minimax` or `alpha-beta` (the legacy input alias `alpha_beta` is accepted), and `depth` must be an integer from 1 through 10.

Example response (elapsed time varies by machine):

```json
{
  "move": "g1h3",
  "algorithm": "alpha-beta",
  "depth": 2,
  "score": 0,
  "nodes": 60,
  "time_ms": 8.4
}
```

`score` is material in centipawns, positive for White and negative for Black. `nodes` includes the root and every searched position. `time_ms` is measured with `time.perf_counter()`.

On the starting position at depth 2, pure Minimax visits 421 nodes and Alpha-Beta visits 60 nodes in the current deterministic move order. Both return the same move and score; elapsed time depends on the machine and runtime.

## Run tests

```powershell
python -m unittest discover -s tests -v
```

## Time-management dataset (Phase 2A)

Generate an offline JSONL dataset from the starting position and seeded engine games. Each sampled position is analyzed once for every configured synthetic clock value. The records contain position features and observed iterative Alpha-Beta search telemetry; they do not contain a recommended or target time.

```powershell
python -m app.learning.time_management.generate_dataset `
  --games 10 `
  --sampling-interval 4 `
  --game-depth 2 `
  --game-candidate-count 3 `
  --max-plies 80 `
  --analysis-budget-ms 1000 `
  --max-analysis-depth 6 `
  --remaining-times-ms 300000,120000,30000,10000,3000 `
  --seed 42 `
  --output data/time_management/dataset.jsonl
```

An optional local PGN can be added with `--pgn path/to/games.pgn`; no external dataset is downloaded. The command writes JSONL records to the selected output and a sibling metadata file such as `dataset.metadata.json`. Relative output paths are resolved from `Chess_AI/`.

## Time-management teacher targets (Phase 2B)

Phase 2B reads the raw Phase 2A JSONL without changing it and writes a separate `training_dataset.jsonl` plus `teacher_summary.json`:

```powershell
python -m app.learning.time_management.teacher `
  --input data/time_management/dataset.jsonl `
  --output data/time_management/training_dataset.jsonl `
  --summary data/time_management/teacher_summary.json
```

For each consecutive depth pair, the teacher normalizes score change as `abs(score_delta) / (score_scale + abs(score_delta))`, combines it with a binary move-change signal using configurable weights, and discounts the result by a bounded node-growth cost signal. Search nodes are used for the cost adjustment because they are less hardware-dependent than elapsed time. The first transition below `min_gain` is the stopping point; a low-gain first transition selects depth 2 as a confirmation depth. If all observed transitions remain meaningful, the deepest completed depth is selected. The target is that depth's cumulative observed time, capped at `remaining_time_ms - clock_safety_margin_ms`.

Confidence is a heuristic quality score from 0 to 1: more observed transitions and lower average information gain increase it, while a timed-out search applies a configurable penalty. A sample with fewer than two completed depths, or no usable clock after the reserve, is marked `insufficient_data` with no target. This teacher is an offline heuristic, not a statistical probability or a mathematically optimal time. No model is trained by this command.

For a larger offline validation, write to separate `validation_*` files so the Phase 2A raw dataset and existing labeled dataset remain untouched:

```powershell
python -m app.learning.time_management.generate_dataset `
  --games 10 --sampling-interval 4 --game-depth 3 `
  --game-candidate-count 3 --max-plies 40 `
  --analysis-budget-ms 1000 --max-analysis-depth 6 `
  --remaining-times-ms 300000,120000,30000,10000,3000 `
  --seed 202602 --output data/time_management/validation_dataset.jsonl

python -m app.learning.time_management.teacher `
  --input data/time_management/validation_dataset.jsonl `
  --output data/time_management/validation_training_dataset.jsonl `
  --summary data/time_management/validation_teacher_summary.json

python -m app.learning.time_management.validation `
  --raw data/time_management/validation_dataset.jsonl `
  --labeled data/time_management/validation_training_dataset.jsonl `
  --output data/time_management/validation_report.json
```

The validation report compares raw and labeled records by `sample_id`, confirms the raw fields are preserved, and reports target percentiles and clock-group statistics without assigning a subjective quality score.

## Time-management baseline models (Phase 2C)

Install the Python requirements, then run the grouped baseline experiment from `Chess_AI/`:

```powershell
python -m app.learning.time_management.train_models `
  --input data/time_management/validation_training_dataset.jsonl `
  --output-root data/time_management `
  --random-state 42 `
  --test-size 0.2
```

The feature vector uses current position/clock features plus the bounded Phase 2A probe summary. Search telemetry (`completed_depth`, nodes, elapsed time, timeout, and configured caps) is only available after running such a probe; the depth-by-depth profile and teacher-derived summaries are excluded. The model does not receive `teacher_time_ms`, `teacher_depth`, confidence, information gain, label status, or other teacher output. The target is `teacher_time_ms`.

Records are split by FEN with `GroupShuffleSplit`, so clock variants of a position cannot cross train/test. The experiment compares a train-median predictor, a small Gradient Boosting regressor, and a small Random Forest. Clock-group errors and tree feature importances are diagnostics; importance is not causal. Models are saved under a timestamped `data/time_management/models/phase2c_*` directory with the scikit-learn version and configuration. Existing model artifacts and root evaluation reports are not silently overwritten.

The current validation corpus has only 98 FEN groups, and its teacher targets did not vary by clock group. These models are an experimental predictive-signal baseline, not production-ready time management, proof of clock awareness, chess strength, or optimal timing. The command does not integrate a model into Chess runtime.

## Phase 2F — Runtime Shadow Evaluation

Shadow mode runs the existing Phase 2C Random Forest **after** the timed Alpha-Beta search has finished. The prediction is diagnostic only: it does not set or change the actual search budget, depth, move selection, timeout handling, or game result. The frontend sends the current side's remaining clock to the timed endpoint; legacy callers may omit it, in which case the requested search budget is used as the clock input.

Runtime inference uses the shared `model_features.py` feature order and the existing artifact `data/time_management/models/phase2c_20261001T122840Z_seed42/random_forest.joblib`. The Phase 2E safety reserve (300 ms) is applied only to `shadow_budget_ms`; the actual engine retains its existing budget and 50 ms search-deadline reserve. One JSONL record is appended after each completed timed search to `data/time_management/runtime_shadow.jsonl`. Records include FEN, remaining clock, actual search time/depth/nodes/timeout/move, model/version, raw prediction, capped shadow budget, usable clock, cap status, and prediction latency. A shadow artifact or logging error is logged and cannot fail the already completed search response.

Run a small sample of real timed engine searches (this plays the engine against itself and appends records):

```powershell
python -m app.learning.time_management.run_shadow_sample `
  --searches 5 `
  --time-budget-ms 1000 `
  --remaining-time-ms 30000 `
  --max-depth 6
```

Evaluate the JSONL log and write `data/time_management/runtime_shadow_report.json`:

```powershell
python -m app.learning.time_management.evaluate_shadow `
  --input data/time_management/runtime_shadow.jsonl `
  --output data/time_management/runtime_shadow_report.json
```

The report includes raw and capped prediction distributions, actual-time distributions, prediction-versus-actual MAE and above/below rates, clock-cap rate, clock-bucket summaries, and inference latency. Runtime does not run the teacher. To compare with existing labels where FEN and clock match, pass an optional labeled dataset:

```powershell
python -m app.learning.time_management.evaluate_shadow `
  --input data/time_management/runtime_shadow.jsonl `
  --teacher-dataset data/time_management/validation_training_dataset.jsonl `
  --output data/time_management/runtime_shadow_report.json
```

Inspect the raw per-search records in `runtime_shadow.jsonl` and aggregate metrics in `runtime_shadow_report.json`. Shadow mode is not a learned runtime time allocator and does not authorize the model to control engine searches.

## Phase 2F.1 — Shadow Data Collection

Phase 2F.1 gathers evidence only. The Phase 2C model remains shadow-only: it does not set the production search budget, and this phase does not run the Teacher, retrain the model, or change production search behavior. The collection uses reproducible positions from the Phase 2A position generator and runs real timed searches at the nine requested clock values. By default, 12 positions × 9 clocks produces 108 searches. The separate JSONL output preserves each FEN, clock, actual search duration/depth/nodes/timeout, raw prediction, capped shadow budget, usable clock, model ID, and inference latency.

Run collection and the offline pre-search probe experiment from `Chess_AI/`:

```powershell
python -m app.learning.time_management.phase2f1 `
  --positions 12 `
  --seed 20261001 `
  --search-budget-ms 250 `
  --max-depth 64 `
  --log data/time_management/phase2f1_runtime_shadow.jsonl `
  --report data/time_management/phase2f1_shadow_report.json
```

The same command runs probes capped at 25, 50, 100, 150, and 200 ms on four generated positions across all nine clock values. It records elapsed time, depth, nodes, post-probe clock, model output using the existing feature schema, a clock-capped estimated full-search budget after probe cost, and probe clock cost. This is an offline feasibility experiment; probe telemetry never enters a live game. Probe rows are saved in `phase2f1_probe_experiment.jsonl`. To rerun only the probes/report using an existing collection, add `--reuse-existing`; otherwise, re-run the command to replace the collection and report with a fresh seeded run. The report includes the sample and clock-bucket distributions, prediction and actual-runtime distributions, MAE/median absolute error/over- and underprediction/correlation where meaningful, cap and timeout rates, probe summaries, and a GO / NOT YET recommendation. Actual search duration is observed runtime behavior, not the Teacher's ideal target, so the runtime comparison is not direct prediction-quality evidence.

## Genetic Algorithm Laboratory

The isolated laboratory API evolves experimental pawn, knight, bishop, rook, and queen material weights. It uses the existing Alpha-Beta search with an injected evaluator; the default game endpoint still uses the existing material values, including the fixed king value of 20,000.

Available endpoints:

- `POST /api/lab/genetic/start/stream` starts generation 1 and streams evaluation events as Server-Sent Events.
- `POST /api/lab/genetic/step/stream` evaluates one next generation and streams evaluation events.
- `POST /api/lab/genetic/play-move` searches a FEN with a supplied evolved chromosome; it does not change the default `/api/ai/move` evaluator.

- `GET /api/lab/genetic/state` — inspect current status, population, and generation history.
- `POST /api/lab/genetic/start` — create and evaluate generation 1. Accepts population size, paired games per individual, search depth (default 2), mutation settings, elite/tournament sizes, ply limit, generation limit, and optional seed.
- `POST /api/lab/genetic/step` — evaluate exactly one new generation and return the updated snapshots and events.
- `POST /api/lab/genetic/reset` — clear the current experiment.

Each candidate is evaluated against a sampled opponent pool. BASELINE (100/320/330/500/900) is always included. Before fitness rankings exist, the other opponents are sampled without replacement from the current population; later generations sample from the previous generation's top five. The default is eight games per candidate: BASELINE plus three sampled opponents, each played once with the candidate as White and once as Black. The setting accepts four, six, or eight games. Opponent sampling is reproducible from the experiment seed. Every candidate receives the same number and types of matchups, and elites are re-evaluated alongside children. Fitness remains wins minus losses, with draws worth zero.

The state/history records fitness and W/D/L/game counts, detailed opponent IDs/types/colors/results for every game, selected parent pairs, gene-by-gene parent sources, crossover chromosomes, mutation deltas, elite copies, best individual, and average fitness. Training is never started on page load. The lab lives at `/laboratory/genetic`; its Run control advances up to ten generations through separate step requests so progress is visible and can be paused between generations.

Laboratory state is currently held in memory by one backend process. Restarting the process clears the experiment, and this first version is intended for a single active lab session rather than durable or multi-user storage.

The event stream reports each candidate and game, each real move with FEN, SAN/UCI, evaluation, nodes, and depth, game results, candidate fitness updates, and generation completion. Completed game traces remain in the active experiment state for replay. The monitor's Pause freezes its replay cursor while evaluation may continue; Follow Live resumes playback. Deployments that do not return a `text/event-stream` response produce a visible error instead of simulated progress.

## Frontend integration

The frontend sends the current position's FEN, chosen algorithm, and depth as JSON to `POST http://127.0.0.1:8000/api/ai/move`, then applies the returned UCI move through `chess.js`. Set `NEXT_PUBLIC_CHESS_AI_URL` to the service base URL when it is not running at `http://127.0.0.1:8000`. For cross-origin deployments, set `CHESS_AI_CORS_ORIGINS` to a comma-separated list of allowed frontend origins; local development allows `localhost:3000` and `127.0.0.1:3000` by default.
