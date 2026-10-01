from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class AIRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fen: str = Field(min_length=1, description="Current board position in FEN format")
    # Accept alpha_beta as a compatibility alias for the existing frontend.
    algorithm: Literal["minimax", "alpha-beta", "alpha_beta"]
    depth: int = Field(strict=True, ge=1, le=10, description="Search depth (1–100)")


class AIResponse(BaseModel):
    move: str
    algorithm: Literal["minimax", "alpha-beta"]
    depth: int
    score: int | float
    nodes: int
    time_ms: float


class TimedAIRequest(BaseModel):
    """Request for the separate time-budgeted Alpha-Beta endpoint."""

    model_config = ConfigDict(extra="forbid")

    fen: str = Field(min_length=1, description="Current chess position in FEN format")
    algorithm: Literal["alpha-beta"]
    time_budget_ms: int = Field(strict=True, ge=1, le=60_000)
    remaining_time_ms: int | None = Field(
        default=None,
        strict=True,
        ge=0,
        le=259_200_000,
        description="Current player's remaining game clock; optional for legacy clients.",
    )
    max_depth: int = Field(default=64, strict=True, ge=1, le=64)


class TimedDepthStats(BaseModel):
    depth: int
    move: str
    score: int
    nodes: int
    time_ms: float = Field(description="Cumulative elapsed time from the start of this move search")


class TimedAIResponse(BaseModel):
    move: str | None
    algorithm: Literal["alpha-beta"]
    depth: int = Field(description="Compatibility alias for completed_depth")
    score: int
    completed_depth: int
    nodes: int
    time_ms: float
    timed_out: bool
    depths_completed: list[TimedDepthStats]
