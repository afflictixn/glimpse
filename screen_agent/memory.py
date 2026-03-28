"""SQLite-backed context memory for analyses, proposals, and conversations."""

from __future__ import annotations

import json
import sqlite3
import time
from typing import Any, Optional

from screen_agent.config import Config

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS analyses (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL    NOT NULL,
    description TEXT    NOT NULL,
    score       INTEGER NOT NULL,
    category    TEXT    NOT NULL,
    summary     TEXT    NOT NULL,
    raw_json    TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS proposals (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            REAL    NOT NULL,
    text          TEXT    NOT NULL,
    action_type   TEXT    NOT NULL,
    confidence    REAL    NOT NULL,
    user_response TEXT,
    analysis_id   INTEGER REFERENCES analyses(id)
);

CREATE TABLE IF NOT EXISTS conversations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL    NOT NULL,
    role        TEXT    NOT NULL,
    content     TEXT    NOT NULL,
    proposal_id INTEGER REFERENCES proposals(id)
);
"""


class Memory:
    """Thin wrapper around a local SQLite database."""

    def __init__(self, cfg: Config) -> None:
        self._conn = sqlite3.connect(cfg.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)

    def save_analysis(
        self,
        description: str,
        score: int,
        category: str,
        summary: str,
        raw: dict[str, Any],
    ) -> int:
        cur = self._conn.execute(
            "INSERT INTO analyses (ts, description, score, category, summary, raw_json) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (time.time(), description, score, category, summary, json.dumps(raw)),
        )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def save_proposal(
        self,
        text: str,
        action_type: str,
        confidence: float,
        analysis_id: int,
    ) -> int:
        cur = self._conn.execute(
            "INSERT INTO proposals (ts, text, action_type, confidence, analysis_id) "
            "VALUES (?, ?, ?, ?, ?)",
            (time.time(), text, action_type, confidence, analysis_id),
        )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def record_user_response(self, proposal_id: int, response: str) -> None:
        self._conn.execute(
            "UPDATE proposals SET user_response = ? WHERE id = ?",
            (response, proposal_id),
        )
        self._conn.commit()

    def save_conversation_turn(
        self, role: str, content: str, proposal_id: Optional[int] = None,
    ) -> int:
        cur = self._conn.execute(
            "INSERT INTO conversations (ts, role, content, proposal_id) "
            "VALUES (?, ?, ?, ?)",
            (time.time(), role, content, proposal_id),
        )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def recent_analyses(self, n: int = 10) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM analyses ORDER BY ts DESC LIMIT ?", (n,),
        ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def recent_proposals(self, n: int = 10) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM proposals ORDER BY ts DESC LIMIT ?", (n,),
        ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def conversation_for_proposal(self, proposal_id: int) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM conversations WHERE proposal_id = ? ORDER BY ts",
            (proposal_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        self._conn.close()
