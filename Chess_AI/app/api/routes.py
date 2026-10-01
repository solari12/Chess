from fastapi import APIRouter, HTTPException

from app.engine.search import search
from app.schemas.chess import AIRequest, AIResponse

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/api/ai/move", response_model=AIResponse)
def get_ai_move(request: AIRequest) -> AIResponse:
    """Run the requested search and return a UCI move with statistics."""
    try:
        return search(request)
    except NotImplementedError as error:
        raise HTTPException(status_code=501, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
