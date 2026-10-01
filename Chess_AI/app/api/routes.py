from fastapi import APIRouter, HTTPException

from app.engine.search import search, search_with_time_budget
from app.schemas.chess import AIRequest, AIResponse, TimedAIRequest, TimedAIResponse

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/api/ai/move", response_model=AIResponse)
def get_ai_move(request: AIRequest) -> AIResponse:
    """Run the requested search and return a UCI move with statistics."""
    try:
        return search(request)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.post("/api/ai/move/timed", response_model=TimedAIResponse)
def get_timed_ai_move(request: TimedAIRequest) -> TimedAIResponse:
    """Run time-budgeted iterative deepening without changing the fixed-depth API."""
    try:
        return search_with_time_budget(request)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
