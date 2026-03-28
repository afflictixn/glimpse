from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()


class RawSQLRequest(BaseModel):
    sql: str
    limit: int = 100


@router.post("/raw_sql")
async def execute_raw_sql(request: Request, body: RawSQLRequest) -> list[dict]:
    db = request.app.state.db
    logger.debug("Raw SQL (limit=%d): %s", body.limit, body.sql)
    try:
        results = await db.execute_raw_sql(body.sql, body.limit)
    except ValueError as e:
        logger.warning("Raw SQL rejected: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Raw SQL failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    logger.debug("Raw SQL returned %d rows", len(results))
    return results
