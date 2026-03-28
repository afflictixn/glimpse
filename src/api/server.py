from __future__ import annotations

import json
import logging
import time

from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect

from src.api.routes import actions, context, events, frames, health, ingest, raw_sql, search
from src.general_agent import server as agent_router
from src.general_agent.ws_manager import ConnectionManager
from src.storage.database import DatabaseManager

logger = logging.getLogger(__name__)


def create_app(
    db: DatabaseManager,
    general_agent=None,
    snapshot_writer=None,
    process_agents=None,
    context_providers=None,
    intelligence_layer=None,
    ws_manager: ConnectionManager | None = None,
) -> FastAPI:
    app = FastAPI(title="Z Exp", version="0.1.0")
    app.state.db = db
    app.state.general_agent = general_agent
    app.state.snapshot_writer = snapshot_writer
    app.state.process_agents = process_agents or []
    app.state.context_providers = context_providers or []
    app.state.intelligence_layer = intelligence_layer
    app.state.ws_manager = ws_manager

    @app.middleware("http")
    async def log_requests(request: Request, call_next) -> Response:
        start = time.perf_counter()
        response: Response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000

        if elapsed_ms > 500:
            logger.warning(
                "%s %s -> %d (%.1fms SLOW)",
                request.method, request.url.path,
                response.status_code, elapsed_ms,
            )
        else:
            logger.debug(
                "%s %s -> %d (%.1fms)",
                request.method, request.url.path,
                response.status_code, elapsed_ms,
            )
        return response

    app.include_router(search.router)
    app.include_router(frames.router)
    app.include_router(events.router)
    app.include_router(context.router)
    app.include_router(actions.router)
    app.include_router(health.router)
    app.include_router(raw_sql.router)
    app.include_router(agent_router.router)
    app.include_router(ingest.router)

    @app.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket) -> None:
        if ws_manager is None:
            await websocket.close(code=1013)
            return
        await ws_manager.connect(websocket)
        try:
            while True:
                data = await websocket.receive_text()
                try:
                    msg = json.loads(data)
                except json.JSONDecodeError:
                    continue
                agent = app.state.general_agent
                if agent is not None:
                    try:
                        await agent._handle_overlay_message(msg)
                    except Exception:
                        logger.error("Overlay message handling failed", exc_info=True)
        except WebSocketDisconnect:
            pass
        except Exception:
            logger.debug("WS client error", exc_info=True)
        finally:
            await ws_manager.disconnect(websocket)

    return app
