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
