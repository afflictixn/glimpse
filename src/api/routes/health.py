from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    counts: dict


@router.get("/health")
async def health_check(request: Request) -> HealthResponse:
    db = request.app.state.db
    counts = await db.get_counts()
    return HealthResponse(
        status="ok",
        uptime_seconds=round(db.uptime_seconds, 2),
        counts=counts,
    )
