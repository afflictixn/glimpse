from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import re

import aiosqlite

from src.config import Settings
from src.storage.models import Action, AdditionalContext, Event, Frame, OCRResult

logger = logging.getLogger(__name__)

_FTS5_SPECIAL = re.compile(r'[^\w\s]', re.UNICODE)


def _sanitize_fts5_query(query: str) -> str:
    """Escape special characters so raw text is safe for FTS5 MATCH."""
    tokens = _FTS5_SPECIAL.sub(' ', query).split()
    if not tokens:
        return '""'
    return " ".join(f'"{t}"' for t in tokens)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS frames (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    snapshot_path TEXT,
    app_name TEXT,
    window_name TEXT,
    focused INTEGER DEFAULT 1,
    capture_trigger TEXT,
    content_hash TEXT
);

CREATE TABLE IF NOT EXISTS ocr_text (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    frame_id INTEGER NOT NULL REFERENCES frames(id) ON DELETE CASCADE,
    text TEXT,
    text_json TEXT,
    confidence REAL
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    frame_id INTEGER NOT NULL REFERENCES frames(id) ON DELETE CASCADE,
    agent_name TEXT NOT NULL,
    app_type TEXT NOT NULL,
    summary TEXT NOT NULL,
    metadata TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS context (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    frame_id INTEGER REFERENCES frames(id) ON DELETE CASCADE,
    source TEXT NOT NULL,
    content_type TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS actions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id INTEGER NOT NULL REFERENCES events(id) ON DELETE CASCADE,
    frame_id INTEGER NOT NULL REFERENCES frames(id) ON DELETE CASCADE,
    agent_name TEXT NOT NULL,
    action_type TEXT NOT NULL,
    action_description TEXT NOT NULL,
    metadata TEXT,
    created_at TEXT NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS ocr_fts USING fts5(
    text, app_name, window_name,
    content='',
    tokenize='unicode61'
);

CREATE VIRTUAL TABLE IF NOT EXISTS events_fts USING fts5(
    summary, app_type, agent_name,
    content='',
    tokenize='unicode61'
);

CREATE VIRTUAL TABLE IF NOT EXISTS context_fts USING fts5(
    content, source, content_type,
    content='',
    tokenize='unicode61'
);

CREATE VIRTUAL TABLE IF NOT EXISTS actions_fts USING fts5(
    action_description, action_type, agent_name,
    content='',
    tokenize='unicode61'
);

CREATE TABLE IF NOT EXISTS memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_type TEXT NOT NULL,
    entity_name TEXT NOT NULL,
    fact TEXT NOT NULL,
    source TEXT,
    confidence REAL DEFAULT 1.0,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    last_seen TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
    entity_name, fact, entity_type,
    content='memory',
    content_rowid='id',
    tokenize='unicode61'
);
"""


class DatabaseManager:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._db_path = settings.db_path
        self._write_lock = asyncio.Lock()
        self._db: aiosqlite.Connection | None = None
        self._started_at = time.monotonic()

    async def initialize(self) -> None:
        self._db = await aiosqlite.connect(str(self._db_path))
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA foreign_keys=ON")
        await self._db.executescript(_SCHEMA)
        await self._db.commit()
        logger.info("Database initialized at %s", self._db_path)

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    @property
    def _conn(self) -> aiosqlite.Connection:
        assert self._db is not None, "Database not initialized"
        return self._db

    # ── Write methods ──

    async def insert_frame(self, frame: Frame) -> int:
        async with self._write_lock:
            cursor = await self._conn.execute(
                "INSERT INTO frames (timestamp, snapshot_path, app_name, window_name, "
                "focused, capture_trigger, content_hash) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    frame.timestamp,
                    frame.snapshot_path,
                    frame.app_name,
                    frame.window_name,
                    1 if frame.focused else 0,
                    frame.capture_trigger,
                    frame.content_hash,
                ),
            )
            await self._conn.commit()
            return cursor.lastrowid  # type: ignore[return-value]

    async def insert_ocr(self, frame_id: int, ocr: OCRResult) -> int:
        async with self._write_lock:
            cursor = await self._conn.execute(
                "INSERT INTO ocr_text (frame_id, text, text_json, confidence) "
                "VALUES (?, ?, ?, ?)",
                (frame_id, ocr.text, ocr.text_json, ocr.confidence),
            )
            row_id = cursor.lastrowid
            frame = await self._conn.execute_fetchall(
                "SELECT app_name, window_name FROM frames WHERE id = ?", (frame_id,)
            )
            app_name = frame[0][0] if frame else None
            window_name = frame[0][1] if frame else None
            await self._conn.execute(
                "INSERT INTO ocr_fts (rowid, text, app_name, window_name) "
                "VALUES (?, ?, ?, ?)",
                (row_id, ocr.text, app_name, window_name),
            )
            await self._conn.commit()
            return row_id  # type: ignore[return-value]

    async def insert_event(self, frame_id: int, event: Event) -> int:
        now = datetime.now(timezone.utc).isoformat()
        meta_json = json.dumps(event.metadata) if event.metadata else None
        async with self._write_lock:
            cursor = await self._conn.execute(
                "INSERT INTO events (frame_id, agent_name, app_type, summary, "
                "metadata, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (frame_id, event.agent_name, event.app_type, event.summary, meta_json, now),
            )
            row_id = cursor.lastrowid
            await self._conn.execute(
                "INSERT INTO events_fts (rowid, summary, app_type, agent_name) "
                "VALUES (?, ?, ?, ?)",
                (row_id, event.summary, event.app_type, event.agent_name),
            )
            await self._conn.commit()
            event.id = row_id
            return row_id  # type: ignore[return-value]

    async def insert_context(self, frame_id: int | None, ctx: AdditionalContext) -> int:
        now = datetime.now(timezone.utc).isoformat()
        meta_json = json.dumps(ctx.metadata) if ctx.metadata else None
        async with self._write_lock:
            cursor = await self._conn.execute(
                "INSERT INTO context (frame_id, source, content_type, content, "
                "metadata, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (frame_id, ctx.source, ctx.content_type, ctx.content, meta_json, now),
            )
            row_id = cursor.lastrowid
            await self._conn.execute(
                "INSERT INTO context_fts (rowid, content, source, content_type) "
                "VALUES (?, ?, ?, ?)",
                (row_id, ctx.content, ctx.source, ctx.content_type),
            )
            await self._conn.commit()
            return row_id  # type: ignore[return-value]

    async def insert_action(self, action: Action) -> int:
        now = datetime.now(timezone.utc).isoformat()
        meta_json = json.dumps(action.metadata) if action.metadata else None
        async with self._write_lock:
            cursor = await self._conn.execute(
                "INSERT INTO actions (event_id, frame_id, agent_name, action_type, "
                "action_description, metadata, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    action.event_id,
                    action.frame_id,
                    action.agent_name,
                    action.action_type,
                    action.action_description,
                    meta_json,
                    now,
                ),
            )
            row_id = cursor.lastrowid
            await self._conn.execute(
                "INSERT INTO actions_fts (rowid, action_description, action_type, agent_name) "
                "VALUES (?, ?, ?, ?)",
                (row_id, action.action_description, action.action_type, action.agent_name),
            )
            await self._conn.commit()
            return row_id  # type: ignore[return-value]

    # ── Read methods ──

    async def search(
        self,
        query: str,
        limit: int = 20,
        start_time: str | None = None,
        end_time: str | None = None,
        app_name: str | None = None,
    ) -> list[dict]:
        sql = (
            "SELECT o.frame_id, o.text, f.app_name, f.window_name, f.timestamp, "
            "f.capture_trigger, o.confidence, f.snapshot_path "
            "FROM ocr_fts fts "
            "JOIN ocr_text o ON o.id = fts.rowid "
            "JOIN frames f ON f.id = o.frame_id "
            "WHERE ocr_fts MATCH ?"
        )
        params: list = [_sanitize_fts5_query(query)]
        if start_time:
            sql += " AND f.timestamp >= ?"
            params.append(start_time)
        if end_time:
            sql += " AND f.timestamp <= ?"
            params.append(end_time)
        if app_name:
            sql += " AND f.app_name = ?"
            params.append(app_name)
        sql += " ORDER BY rank LIMIT ?"
        params.append(limit)
        rows = await self._conn.execute_fetchall(sql, params)
        return [dict(r) for r in rows]

    async def search_events(
        self,
        query: str | None = None,
        app_type: str | None = None,
        agent_name: str | None = None,
        limit: int = 20,
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> list[dict]:
        if query:
            sql = (
                "SELECT e.id as event_id, e.frame_id, e.agent_name, e.app_type, "
                "e.summary, e.metadata, e.created_at, f.timestamp, f.app_name, f.window_name "
                "FROM events_fts fts "
                "JOIN events e ON e.id = fts.rowid "
                "JOIN frames f ON f.id = e.frame_id "
                "WHERE events_fts MATCH ?"
            )
            params: list = [_sanitize_fts5_query(query)]
        else:
            sql = (
                "SELECT e.id as event_id, e.frame_id, e.agent_name, e.app_type, "
                "e.summary, e.metadata, e.created_at, f.timestamp, f.app_name, f.window_name "
                "FROM events e "
                "JOIN frames f ON f.id = e.frame_id "
                "WHERE 1=1"
            )
            params = []
        if app_type:
            sql += " AND e.app_type = ?"
            params.append(app_type)
        if agent_name:
            sql += " AND e.agent_name = ?"
            params.append(agent_name)
        if start_time:
            sql += " AND e.created_at >= ?"
            params.append(start_time)
        if end_time:
            sql += " AND e.created_at <= ?"
            params.append(end_time)
        sql += " ORDER BY e.created_at DESC LIMIT ?"
        params.append(limit)
        rows = await self._conn.execute_fetchall(sql, params)
        return [dict(r) for r in rows]

    async def search_context(
        self,
        query: str | None = None,
        source: str | None = None,
        content_type: str | None = None,
        limit: int = 20,
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> list[dict]:
        if query:
            sql = (
                "SELECT c.id as context_id, c.frame_id, c.source, c.content_type, "
                "c.content, c.metadata, c.created_at, f.timestamp, f.app_name, f.window_name "
                "FROM context_fts fts "
                "JOIN context c ON c.id = fts.rowid "
                "LEFT JOIN frames f ON f.id = c.frame_id "
                "WHERE context_fts MATCH ?"
            )
            params: list = [_sanitize_fts5_query(query)]
        else:
            sql = (
                "SELECT c.id as context_id, c.frame_id, c.source, c.content_type, "
                "c.content, c.metadata, c.created_at, f.timestamp, f.app_name, f.window_name "
                "FROM context c "
                "LEFT JOIN frames f ON f.id = c.frame_id "
                "WHERE 1=1"
            )
            params = []
        if source:
            sql += " AND c.source = ?"
            params.append(source)
        if content_type:
            sql += " AND c.content_type = ?"
            params.append(content_type)
        if start_time:
            sql += " AND c.created_at >= ?"
            params.append(start_time)
        if end_time:
            sql += " AND c.created_at <= ?"
            params.append(end_time)
        sql += " ORDER BY c.created_at DESC LIMIT ?"
        params.append(limit)
        rows = await self._conn.execute_fetchall(sql, params)
        return [dict(r) for r in rows]

    async def search_actions(
        self,
        query: str | None = None,
        action_type: str | None = None,
        agent_name: str | None = None,
        event_id: int | None = None,
        limit: int = 20,
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> list[dict]:
        if query:
            sql = (
                "SELECT a.id as action_id, a.event_id, a.frame_id, a.agent_name, "
                "a.action_type, a.action_description, a.metadata, a.created_at, "
                "e.summary as event_summary, e.app_type as event_app_type, "
                "f.timestamp, f.app_name, f.window_name "
                "FROM actions_fts fts "
                "JOIN actions a ON a.id = fts.rowid "
                "JOIN events e ON e.id = a.event_id "
                "JOIN frames f ON f.id = a.frame_id "
                "WHERE actions_fts MATCH ?"
            )
            params: list = [_sanitize_fts5_query(query)]
        else:
            sql = (
                "SELECT a.id as action_id, a.event_id, a.frame_id, a.agent_name, "
                "a.action_type, a.action_description, a.metadata, a.created_at, "
                "e.summary as event_summary, e.app_type as event_app_type, "
                "f.timestamp, f.app_name, f.window_name "
                "FROM actions a "
                "JOIN events e ON e.id = a.event_id "
                "JOIN frames f ON f.id = a.frame_id "
                "WHERE 1=1"
            )
            params = []
        if action_type:
            sql += " AND a.action_type = ?"
            params.append(action_type)
        if agent_name:
            sql += " AND a.agent_name = ?"
            params.append(agent_name)
        if event_id is not None:
            sql += " AND a.event_id = ?"
            params.append(event_id)
        if start_time:
            sql += " AND a.created_at >= ?"
            params.append(start_time)
        if end_time:
            sql += " AND a.created_at <= ?"
            params.append(end_time)
        sql += " ORDER BY a.created_at DESC LIMIT ?"
        params.append(limit)
        rows = await self._conn.execute_fetchall(sql, params)
        return [dict(r) for r in rows]

    async def get_frame(self, frame_id: int) -> dict | None:
        rows = await self._conn.execute_fetchall(
            "SELECT * FROM frames WHERE id = ?", (frame_id,)
        )
        if not rows:
            return None
        frame = dict(rows[0])

        ocr_rows = await self._conn.execute_fetchall(
            "SELECT * FROM ocr_text WHERE frame_id = ?", (frame_id,)
        )
        frame["ocr"] = dict(ocr_rows[0]) if ocr_rows else None

        event_rows = await self._conn.execute_fetchall(
            "SELECT * FROM events WHERE frame_id = ?", (frame_id,)
        )
        frame["events"] = [dict(r) for r in event_rows]

        ctx_rows = await self._conn.execute_fetchall(
            "SELECT * FROM context WHERE frame_id = ?", (frame_id,)
        )
        frame["context"] = [dict(r) for r in ctx_rows]

        action_rows = await self._conn.execute_fetchall(
            "SELECT a.*, e.summary as event_summary, e.app_type as event_app_type "
            "FROM actions a JOIN events e ON e.id = a.event_id "
            "WHERE a.frame_id = ?",
            (frame_id,),
        )
        frame["actions"] = [dict(r) for r in action_rows]
        return frame

    async def get_events_for_frame(self, frame_id: int) -> list[dict]:
        rows = await self._conn.execute_fetchall(
            "SELECT e.*, f.timestamp, f.app_name, f.window_name "
            "FROM events e JOIN frames f ON f.id = e.frame_id "
            "WHERE e.frame_id = ?",
            (frame_id,),
        )
        return [dict(r) for r in rows]

    async def get_context_for_frame(self, frame_id: int) -> list[dict]:
        rows = await self._conn.execute_fetchall(
            "SELECT c.*, f.timestamp, f.app_name, f.window_name "
            "FROM context c LEFT JOIN frames f ON f.id = c.frame_id "
            "WHERE c.frame_id = ?",
            (frame_id,),
        )
        return [dict(r) for r in rows]

    async def get_actions_for_event(self, event_id: int) -> list[dict]:
        rows = await self._conn.execute_fetchall(
            "SELECT a.*, e.summary as event_summary, e.app_type as event_app_type, "
            "f.timestamp, f.app_name, f.window_name "
            "FROM actions a "
            "JOIN events e ON e.id = a.event_id "
            "JOIN frames f ON f.id = a.frame_id "
            "WHERE a.event_id = ?",
            (event_id,),
        )
        return [dict(r) for r in rows]

    async def get_actions_for_frame(self, frame_id: int) -> list[dict]:
        rows = await self._conn.execute_fetchall(
            "SELECT a.*, e.summary as event_summary, e.app_type as event_app_type, "
            "f.timestamp, f.app_name, f.window_name "
            "FROM actions a "
            "JOIN events e ON e.id = a.event_id "
            "JOIN frames f ON f.id = a.frame_id "
            "WHERE a.frame_id = ?",
            (frame_id,),
        )
        return [dict(r) for r in rows]

    async def execute_raw_sql(self, sql: str, limit: int = 100) -> list[dict]:
        stripped = sql.strip().upper()
        if not stripped.startswith("SELECT"):
            raise ValueError("Only SELECT queries are allowed")
        if "LIMIT" not in stripped:
            sql = sql.rstrip(";") + f" LIMIT {limit}"
        rows = await self._conn.execute_fetchall(sql)
        return [dict(r) for r in rows]

    # ── Memory methods ──

    async def insert_memory(
        self, entity_type: str, entity_name: str, fact: str,
        source: str | None = None, confidence: float = 1.0,
    ) -> int:
        now = datetime.now(timezone.utc).isoformat()
        async with self._write_lock:
            cursor = await self._conn.execute(
                "INSERT INTO memory (entity_type, entity_name, fact, source, "
                "confidence, created_at, last_seen) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (entity_type, entity_name, fact, source, confidence, now, now),
            )
            row_id = cursor.lastrowid
            await self._conn.execute(
                "INSERT INTO memory_fts (rowid, entity_name, fact, entity_type) "
                "VALUES (?, ?, ?, ?)",
                (row_id, entity_name, fact, entity_type),
            )
            await self._conn.commit()
            return row_id  # type: ignore[return-value]

    @staticmethod
    def _sanitize_fts_query(query: str) -> str:
        """Quote each token so FTS5 special chars (dots, dashes, etc.) are treated as literals."""
        tokens = query.split()
        return " ".join(f'"{t}"' for t in tokens if t)

    async def search_memory(
        self, query: str, entity_type: str | None = None, limit: int = 20,
    ) -> list[dict]:
        if query:
            fts_query = self._sanitize_fts_query(query)
            if not fts_query:
                query = ""
        if query:
            sql = (
                "SELECT m.id, m.entity_type, m.entity_name, m.fact, m.source, "
                "m.confidence, m.created_at, m.last_seen "
                "FROM memory_fts fts "
                "JOIN memory m ON m.id = fts.rowid "
                "WHERE memory_fts MATCH ?"
            )
            params: list = [fts_query]
        else:
            sql = (
                "SELECT id, entity_type, entity_name, fact, source, "
                "confidence, created_at, last_seen FROM memory WHERE 1=1"
            )
            params = []
        if entity_type:
            sql += " AND m.entity_type = ?" if query else " AND entity_type = ?"
            params.append(entity_type)
        sql += " ORDER BY last_seen DESC LIMIT ?"
        params.append(limit)
        rows = await self._conn.execute_fetchall(sql, params)
        return [dict(r) for r in rows]

    async def update_memory_last_seen(self, memory_id: int) -> None:
        now = datetime.now(timezone.utc).isoformat()
        async with self._write_lock:
            await self._conn.execute(
                "UPDATE memory SET last_seen = ? WHERE id = ?", (now, memory_id),
            )
            await self._conn.commit()

    async def delete_memory(self, memory_id: int) -> bool:
        async with self._write_lock:
            await self._conn.execute(
                "DELETE FROM memory_fts WHERE rowid = ?", (memory_id,),
            )
            cursor = await self._conn.execute(
                "DELETE FROM memory WHERE id = ?", (memory_id,),
            )
            await self._conn.commit()
            return cursor.rowcount > 0

    async def get_counts(self) -> dict:
        result = {}
        for table in ("frames", "ocr_text", "events", "context", "actions", "memory"):
            rows = await self._conn.execute_fetchall(f"SELECT COUNT(*) as c FROM {table}")
            result[table] = rows[0][0]
        return result

    async def get_latest_frame_id(self) -> int | None:
        rows = await self._conn.execute_fetchall(
            "SELECT id FROM frames ORDER BY id DESC LIMIT 1"
        )
        return rows[0][0] if rows else None

    # ── Retention cleanup ──

    async def cleanup(self) -> int:
        cutoff = (
            datetime.now(timezone.utc)
            - timedelta(days=self._settings.max_retention_days)
        ).isoformat()

        async with self._write_lock:
            snapshot_rows = await self._conn.execute_fetchall(
                "SELECT id, snapshot_path FROM frames WHERE timestamp < ?", (cutoff,)
            )
            frame_ids = [r[0] for r in snapshot_rows]
            snapshot_paths = [r[1] for r in snapshot_rows if r[1]]

            if not frame_ids:
                return 0

            placeholders = ",".join("?" * len(frame_ids))

            ocr_ids = await self._conn.execute_fetchall(
                f"SELECT id FROM ocr_text WHERE frame_id IN ({placeholders})", frame_ids
            )
            for r in ocr_ids:
                await self._conn.execute("DELETE FROM ocr_fts WHERE rowid = ?", (r[0],))

            event_rows = await self._conn.execute_fetchall(
                f"SELECT id FROM events WHERE frame_id IN ({placeholders})", frame_ids
            )
            event_ids = [r[0] for r in event_rows]
            for r in event_rows:
                await self._conn.execute("DELETE FROM events_fts WHERE rowid = ?", (r[0],))

            if event_ids:
                ev_placeholders = ",".join("?" * len(event_ids))
                action_ids = await self._conn.execute_fetchall(
                    f"SELECT id FROM actions WHERE event_id IN ({ev_placeholders})",
                    event_ids,
                )
                for r in action_ids:
                    await self._conn.execute(
                        "DELETE FROM actions_fts WHERE rowid = ?", (r[0],)
                    )

            ctx_ids = await self._conn.execute_fetchall(
                f"SELECT id FROM context WHERE frame_id IN ({placeholders})", frame_ids
            )
            for r in ctx_ids:
                await self._conn.execute("DELETE FROM context_fts WHERE rowid = ?", (r[0],))

            await self._conn.execute(
                f"DELETE FROM frames WHERE id IN ({placeholders})", frame_ids
            )
            await self._conn.commit()

            for p in snapshot_paths:
                path = Path(p)
                if path.exists():
                    path.unlink(missing_ok=True)

            logger.info("Cleaned up %d expired frames", len(frame_ids))
            return len(frame_ids)

    @property
    def uptime_seconds(self) -> float:
        return time.monotonic() - self._started_at
