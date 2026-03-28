"""FastAPI router for the general agent.

Mounted under /agent in the main API server.

Endpoints:
  POST /agent/push   — receive events/actions (internal, from pipeline)
  POST /agent/chat   — user sends a message, agent responds
  GET  /agent/status  — health check, queue depth, conversation state
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent", tags=["agent"])


class PushRequest(BaseModel):
    type: str  # "event" or "action"
    data: dict


class ChatRequest(BaseModel):
    message: str


@router.post("/push", status_code=202)
async def push(req: PushRequest, request: Request) -> dict:
    """Receive an event or action push. Returns 202 immediately."""
    if req.type not in ("event", "action"):
        return JSONResponse(
            status_code=400,
            content={"error": "type must be 'event' or 'action'"},
        )
    agent = request.app.state.general_agent
    await agent.push(req.type, req.data)
    return {"status": "queued"}


@router.post("/chat")
async def chat(req: ChatRequest, request: Request) -> dict:
    """User sends a message and gets a response."""
    agent = request.app.state.general_agent
    response = await agent.chat(req.message)
    return {"response": response}


@router.get("/status")
async def status(request: Request) -> dict:
    """Health check with queue depth and conversation state."""
    agent = request.app.state.general_agent
    return agent.status()
