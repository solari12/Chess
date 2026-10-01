from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class AIRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fen: str = Field(min_length=1, description="Current board position in FEN format")
    # Accept alpha_beta as a compatibility alias for the existing frontend.
    algorithm: Literal["minimax", "alpha-beta", "alpha_beta"]
    depth: int = Field(strict=True, ge=1, le=10, description="Search depth (1–10)")


class AIResponse(BaseModel):
    move: str
    algorithm: Literal["minimax", "alpha-beta"]
    depth: int
    score: int | float
    nodes: int
    time_ms: float
