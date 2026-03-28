from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter()


class RawSQLRequest(BaseModel):
    sql: str
    limit: int = 100


@router.post("/raw_sql")
async def execute_raw_sql(request: Request, body: RawSQLRequest) -> list[dict]:
    db = request.app.state.db
    try:
        results = await db.execute_raw_sql(body.sql, body.limit)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return results
