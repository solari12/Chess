# Chess AI Service

A Python FastAPI chess service with an explicit **INPUT → PROCESSING → OUTPUT** flow:

```text
JSON request → FastAPI route → search controller → Minimax → JSON response
```

The `minimax` endpoint uses a material evaluation and exhaustive minimax search without pruning. It uses `python-chess` to parse FEN positions and generate legal moves. `alpha_beta` remains unimplemented and returns HTTP 501. A terminal position has no move to return, so the endpoint responds with HTTP 422. The response schema is unchanged.

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
  "algorithm": "minimax",
  "depth": 2
}
```

`fen` must be non-empty, `algorithm` must be `minimax` or `alpha_beta`, and `depth` must be an integer from 1 through 10.

Example response (the selected move and measured statistics can vary):

```json
{
  "move": "b1c3",
  "algorithm": "minimax",
  "depth": 2,
  "score": 0,
  "nodes": 421,
  "time_ms": 8.4
}
```

`score` is material in centipawns, positive for White and negative for Black. `nodes` includes the root and every searched position. `time_ms` is measured with `time.perf_counter()`.

## Run tests

```powershell
python -m unittest discover -s tests -v
```

## Frontend integration

The frontend sends the current position's FEN, chosen algorithm, and depth as JSON to `POST http://127.0.0.1:8000/api/ai/move`, then applies the returned UCI move through `chess.js`. Set `NEXT_PUBLIC_CHESS_AI_URL` to the service base URL when it is not running at `http://127.0.0.1:8000`. For cross-origin deployments, set `CHESS_AI_CORS_ORIGINS` to a comma-separated list of allowed frontend origins; local development allows `localhost:3000` and `127.0.0.1:3000` by default.
