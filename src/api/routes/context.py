from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()


class ContextResult(BaseModel):
    context_id: int
    frame_id: int | None
    source: str
    content_type: str
    content: str
    metadata: dict | None = None
    created_at: str
    timestamp: str | None = None
    app_name: str | None = None
    window_name: str | None = None


class ContextInput(BaseModel):
    frame_id: int | None = None
    source: str
    content_type: str
    content: str
    metadata: dict | None = None


def _parse_metadata(raw) -> dict | None:
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None


@router.get("/context")
async def search_context(
    request: Request,
    q: str | None = Query(None, description="FTS5 search query"),
    source: str | None = Query(None),
    content_type: str | None = Query(None),
    limit: int = Query(20, ge=1, le=500),
    start_time: str | None = Query(None),
    end_time: str | None = Query(None),
) -> list[ContextResult]:
    db = request.app.state.db
    logger.debug("Context search q=%r source=%s type=%s limit=%d", q, source, content_type, limit)
    rows = await db.search_context(
        query=q,
        source=source,
        content_type=content_type,
        limit=limit,
        start_time=start_time,
        end_time=end_time,
    )
    return [
        ContextResult(
            context_id=r["context_id"],
            frame_id=r.get("frame_id"),
            source=r["source"],
            content_type=r["content_type"],
            content=r["content"],
            metadata=_parse_metadata(r.get("metadata")),
            created_at=r["created_at"],
            timestamp=r.get("timestamp"),
            app_name=r.get("app_name"),
            window_name=r.get("window_name"),
        )
        for r in rows
    ]


@router.get("/context/{frame_id}")
async def get_context_for_frame(request: Request, frame_id: int) -> list[ContextResult]:
    db = request.app.state.db
    rows = await db.get_context_for_frame(frame_id)
    return [
        ContextResult(
            context_id=r["id"],
            frame_id=r.get("frame_id"),
            source=r["source"],
            content_type=r["content_type"],
            content=r["content"],
            metadata=_parse_metadata(r.get("metadata")),
            created_at=r["created_at"],
            timestamp=r.get("timestamp"),
            app_name=r.get("app_name"),
            window_name=r.get("window_name"),
        )
        for r in rows
    ]


@router.post("/context")
async def push_context(request: Request, body: ContextInput) -> dict:
    from src.storage.models import AdditionalContext

    db = request.app.state.db

    frame_id = body.frame_id
    if frame_id is None:
        frame_id = await db.get_latest_frame_id()

    ctx = AdditionalContext(
        frame_id=frame_id,
        source=body.source,
        content_type=body.content_type,
        content=body.content,
        metadata=body.metadata or {},
    )
    context_id = await db.insert_context(frame_id, ctx)
    logger.debug("Inserted context %d (source=%s) for frame %d", context_id, body.source, frame_id)
    return {"context_id": context_id, "frame_id": frame_id}
