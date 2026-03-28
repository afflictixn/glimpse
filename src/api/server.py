from __future__ import annotations

from fastapi import FastAPI

from src.api.routes import actions, context, events, frames, health, raw_sql, search
from src.storage.database import DatabaseManager


def create_app(db: DatabaseManager) -> FastAPI:
    app = FastAPI(title="Glimpse", version="0.1.0")
    app.state.db = db

    app.include_router(search.router)
    app.include_router(frames.router)
    app.include_router(events.router)
    app.include_router(context.router)
    app.include_router(actions.router)
    app.include_router(health.router)
    app.include_router(raw_sql.router)

    return app
