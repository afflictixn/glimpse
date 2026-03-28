from __future__ import annotations

import json

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel

router = APIRouter()


class EventResult(BaseModel):
    event_id: int
    frame_id: int
    agent_name: str
    app_type: str
    summary: str
    metadata: dict | None = None
    created_at: str
    timestamp: str = ""
    app_name: str | None = None
    window_name: str | None = None


def _parse_metadata(raw) -> dict | None:
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None


@router.get("/events")
async def search_events(
    request: Request,
    q: str | None = Query(None, description="FTS5 search query"),
    app_type: str | None = Query(None),
    agent_name: str | None = Query(None),
    limit: int = Query(20, ge=1, le=500),
    start_time: str | None = Query(None),
    end_time: str | None = Query(None),
) -> list[EventResult]:
    db = request.app.state.db
    rows = await db.search_events(
        query=q,
        app_type=app_type,
        agent_name=agent_name,
        limit=limit,
        start_time=start_time,
        end_time=end_time,
    )
    return [
        EventResult(
            event_id=r["event_id"],
            frame_id=r["frame_id"],
            agent_name=r["agent_name"],
            app_type=r["app_type"],
            summary=r["summary"],
            metadata=_parse_metadata(r.get("metadata")),
            created_at=r["created_at"],
            timestamp=r.get("timestamp", ""),
            app_name=r.get("app_name"),
            window_name=r.get("window_name"),
        )
        for r in rows
    ]


@router.get("/events/{frame_id}")
async def get_events_for_frame(request: Request, frame_id: int) -> list[EventResult]:
    db = request.app.state.db
    rows = await db.get_events_for_frame(frame_id)
    return [
        EventResult(
            event_id=r["id"],
            frame_id=r["frame_id"],
            agent_name=r["agent_name"],
            app_type=r["app_type"],
            summary=r["summary"],
            metadata=_parse_metadata(r.get("metadata")),
            created_at=r["created_at"],
            timestamp=r.get("timestamp", ""),
            app_name=r.get("app_name"),
            window_name=r.get("window_name"),
        )
        for r in rows
    ]
