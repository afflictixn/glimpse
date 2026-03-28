from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()


class OCRSearchResult(BaseModel):
    frame_id: int
    text: str
    app_name: str | None
    window_name: str | None
    timestamp: str
    capture_trigger: str
    confidence: float | None
    snapshot_path: str | None


@router.get("/search")
async def search_ocr(
    request: Request,
    q: str = Query(..., description="FTS5 search query"),
    app_name: str | None = Query(None),
    limit: int = Query(20, ge=1, le=500),
    start_time: str | None = Query(None),
    end_time: str | None = Query(None),
) -> list[OCRSearchResult]:
    db = request.app.state.db
    logger.debug("OCR search q=%r app=%s limit=%d", q, app_name, limit)
    rows = await db.search(
        query=q,
        limit=limit,
        start_time=start_time,
        end_time=end_time,
        app_name=app_name,
    )
    logger.debug("OCR search returned %d results", len(rows))
    return [OCRSearchResult(**r) for r in rows]
