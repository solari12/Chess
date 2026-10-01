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

## Frontend integration

The frontend sends the current position's FEN, chosen algorithm, and depth as JSON to `POST http://127.0.0.1:8000/api/ai/move`, then applies the returned UCI move through `chess.js`. Set `NEXT_PUBLIC_CHESS_AI_URL` to the service base URL when it is not running at `http://127.0.0.1:8000`. For cross-origin deployments, set `CHESS_AI_CORS_ORIGINS` to a comma-separated list of allowed frontend origins; local development allows `localhost:3000` and `127.0.0.1:3000` by default.
