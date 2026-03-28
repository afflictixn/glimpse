from __future__ import annotations

import logging
import time

from fastapi import FastAPI, Request, Response

from src.api.routes import actions, context, events, frames, health, raw_sql, search
from src.storage.database import DatabaseManager

logger = logging.getLogger(__name__)


def create_app(db: DatabaseManager) -> FastAPI:
    app = FastAPI(title="Glimpse", version="0.1.0")
    app.state.db = db

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

    return app
