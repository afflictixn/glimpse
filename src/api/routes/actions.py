from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()


class ActionResult(BaseModel):
    action_id: int
    event_id: int
    frame_id: int
    agent_name: str
    action_type: str
    action_description: str
    metadata: dict | None = None
    created_at: str
    event_summary: str = ""
    event_app_type: str = ""
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


@router.get("/actions")
async def search_actions(
    request: Request,
    q: str | None = Query(None, description="FTS5 search query"),
    action_type: str | None = Query(None),
    agent_name: str | None = Query(None),
    event_id: int | None = Query(None),
    limit: int = Query(20, ge=1, le=500),
    start_time: str | None = Query(None),
    end_time: str | None = Query(None),
) -> list[ActionResult]:
    db = request.app.state.db
    logger.debug("Action search q=%r type=%s agent=%s limit=%d", q, action_type, agent_name, limit)
    rows = await db.search_actions(
        query=q,
        action_type=action_type,
        agent_name=agent_name,
        event_id=event_id,
        limit=limit,
        start_time=start_time,
        end_time=end_time,
    )
    return [
        ActionResult(
            action_id=r["action_id"],
            event_id=r["event_id"],
            frame_id=r["frame_id"],
            agent_name=r["agent_name"],
            action_type=r["action_type"],
            action_description=r["action_description"],
            metadata=_parse_metadata(r.get("metadata")),
            created_at=r["created_at"],
            event_summary=r.get("event_summary", ""),
            event_app_type=r.get("event_app_type", ""),
            timestamp=r.get("timestamp", ""),
            app_name=r.get("app_name"),
            window_name=r.get("window_name"),
        )
        for r in rows
    ]


@router.get("/actions/{event_id}")
async def get_actions_for_event(request: Request, event_id: int) -> list[ActionResult]:
    db = request.app.state.db
    rows = await db.get_actions_for_event(event_id)
    return [
        ActionResult(
            action_id=r["id"],
            event_id=r["event_id"],
            frame_id=r["frame_id"],
            agent_name=r["agent_name"],
            action_type=r["action_type"],
            action_description=r["action_description"],
            metadata=_parse_metadata(r.get("metadata")),
            created_at=r["created_at"],
            event_summary=r.get("event_summary", ""),
            event_app_type=r.get("event_app_type", ""),
            timestamp=r.get("timestamp", ""),
            app_name=r.get("app_name"),
            window_name=r.get("window_name"),
        )
        for r in rows
    ]
