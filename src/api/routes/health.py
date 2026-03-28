from __future__ import annotations

import logging

from fastapi import APIRouter, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    counts: dict


@router.get("/health")
async def health_check(request: Request) -> HealthResponse:
    db = request.app.state.db
    counts = await db.get_counts()
    logger.debug("Health check: uptime=%.1fs counts=%s", db.uptime_seconds, counts)
    return HealthResponse(
        status="ok",
        uptime_seconds=round(db.uptime_seconds, 2),
        counts=counts,
    )
