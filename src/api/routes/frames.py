from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()


class OCRSearchResult(BaseModel):
    frame_id: int
    text: str
    app_name: str | None = None
    window_name: str | None = None
    timestamp: str = ""
    capture_trigger: str = ""
    confidence: float | None = None
    snapshot_path: str | None = None


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


class FrameDetail(BaseModel):
    frame_id: int
    timestamp: str
    app_name: str | None = None
    window_name: str | None = None
    capture_trigger: str = ""
    snapshot_path: str | None = None
    ocr: OCRSearchResult | None = None
    events: list[EventResult] = []
    context: list[ContextResult] = []
    actions: list[ActionResult] = []


def _parse_metadata(raw) -> dict | None:
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None


@router.get("/frames/{frame_id}")
async def get_frame_image(request: Request, frame_id: int):
    db = request.app.state.db
    frame = await db.get_frame(frame_id)
    if frame is None:
        logger.debug("Frame %d not found", frame_id)
        raise HTTPException(status_code=404, detail="Frame not found")

    snapshot_path = frame.get("snapshot_path")
    if not snapshot_path or not Path(snapshot_path).exists():
        logger.warning("Snapshot file missing for frame %d: %s", frame_id, snapshot_path)
        raise HTTPException(status_code=404, detail="Snapshot file not found")

    return FileResponse(snapshot_path, media_type="image/jpeg")


@router.get("/frames/{frame_id}/metadata")
async def get_frame_metadata(request: Request, frame_id: int) -> FrameDetail:
    db = request.app.state.db
    frame = await db.get_frame(frame_id)
    if frame is None:
        logger.debug("Frame %d not found for metadata", frame_id)
        raise HTTPException(status_code=404, detail="Frame not found")

    ocr_data = frame.get("ocr")
    ocr = None
    if ocr_data:
        ocr = OCRSearchResult(
            frame_id=frame_id,
            text=ocr_data.get("text", ""),
            app_name=frame.get("app_name"),
            window_name=frame.get("window_name"),
            timestamp=frame.get("timestamp", ""),
            capture_trigger=frame.get("capture_trigger", ""),
            confidence=ocr_data.get("confidence"),
            snapshot_path=frame.get("snapshot_path"),
        )

    events_list = []
    for e in frame.get("events", []):
        events_list.append(EventResult(
            event_id=e["id"],
            frame_id=frame_id,
            agent_name=e["agent_name"],
            app_type=e["app_type"],
            summary=e["summary"],
            metadata=_parse_metadata(e.get("metadata")),
            created_at=e["created_at"],
            timestamp=frame.get("timestamp", ""),
            app_name=frame.get("app_name"),
            window_name=frame.get("window_name"),
        ))

    context_list = []
    for c in frame.get("context", []):
        context_list.append(ContextResult(
            context_id=c["id"],
            frame_id=c.get("frame_id"),
            source=c["source"],
            content_type=c["content_type"],
            content=c["content"],
            metadata=_parse_metadata(c.get("metadata")),
            created_at=c["created_at"],
            timestamp=frame.get("timestamp"),
            app_name=frame.get("app_name"),
            window_name=frame.get("window_name"),
        ))

    actions_list = []
    for a in frame.get("actions", []):
        actions_list.append(ActionResult(
            action_id=a["id"],
            event_id=a["event_id"],
            frame_id=a["frame_id"],
            agent_name=a["agent_name"],
            action_type=a["action_type"],
            action_description=a["action_description"],
            metadata=_parse_metadata(a.get("metadata")),
            created_at=a["created_at"],
            event_summary=a.get("event_summary", ""),
            event_app_type=a.get("event_app_type", ""),
            timestamp=frame.get("timestamp", ""),
            app_name=frame.get("app_name"),
            window_name=frame.get("window_name"),
        ))

    return FrameDetail(
        frame_id=frame_id,
        timestamp=frame.get("timestamp", ""),
        app_name=frame.get("app_name"),
        window_name=frame.get("window_name"),
        capture_trigger=frame.get("capture_trigger", ""),
        snapshot_path=frame.get("snapshot_path"),
        ocr=ocr,
        events=events_list,
        context=context_list,
        actions=actions_list,
    )
