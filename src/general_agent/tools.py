"""Tool registry for the general agent.

Each tool is an async callable that the agent can invoke mid-conversation
to gather information before deciding what to surface to the user.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Awaitable

from src.llm.types import ToolSpec
from src.storage.database import DatabaseManager

logger = logging.getLogger(__name__)


@dataclass
class RegisteredTool:
    """Pairs a portable ToolSpec with its implementation."""
    spec: ToolSpec
    fn: Callable[..., Awaitable[str]]


class ToolRegistry:
    """Registry of tools available to the general agent."""

    def __init__(self, db: DatabaseManager) -> None:
        self._db = db
        self._tools: dict[str, RegisteredTool] = {}
        self._register_builtins()

    def _register_builtins(self) -> None:
        self.register(RegisteredTool(
            spec=ToolSpec(
                name="db_query",
                description=(
                    "Query the Glimpse database. Supports full-text search across OCR text, "
                    "events, context, and actions. Use this to look up what the user was doing, "
                    "find past screen content, or check event/action history."
                ),
                parameters={
                    "table": "One of: ocr, events, context, actions",
                    "query": "FTS5 search query string",
                    "limit": "Max results (default 10)",
                },
            ),
            fn=self._db_query,
        ))
        self.register(RegisteredTool(
            spec=ToolSpec(
                name="db_raw_sql",
                description=(
                    "Execute a raw SELECT query against the Glimpse SQLite database. "
                    "Use for complex queries that the FTS search tools can't handle."
                ),
                parameters={
                    "sql": "SELECT query to execute",
                    "limit": "Max rows (default 20)",
                },
            ),
            fn=self._db_raw_sql,
        ))
        self.register(RegisteredTool(
            spec=ToolSpec(
                name="contact_lookup",
                description=(
                    "Look up context about a person from Glimpse event history — "
                    "recent interactions, mentions, social context gathered from screen captures."
                ),
                parameters={
                    "name": "Person's name to search for",
                },
            ),
            fn=self._contact_lookup,
        ))

    def register(self, tool: RegisteredTool) -> None:
        self._tools[tool.spec.name] = tool

    def get(self, name: str) -> RegisteredTool | None:
        return self._tools.get(name)

    def list_specs(self) -> list[ToolSpec]:
        """Return portable tool specs for the LLM provider."""
        return [t.spec for t in self._tools.values()]

    async def call(self, name: str, **kwargs: Any) -> str:
        tool = self._tools.get(name)
        if tool is None:
            return json.dumps({"error": f"Unknown tool: {name}"})
        try:
            return await tool.fn(**kwargs)
        except Exception as e:
            logger.error("Tool %s failed: %s", name, e, exc_info=True)
            return json.dumps({"error": str(e)})

    # ── Built-in tool implementations ──────────────────────────

    async def _db_query(self, table: str = "ocr", query: str = "", limit: str = "10") -> str:
        n = min(int(limit), 50)
        if table == "ocr":
            rows = await self._db.search(query, limit=n)
        elif table == "events":
            rows = await self._db.search_events(query, limit=n)
        elif table == "context":
            rows = await self._db.search_context(query, limit=n)
        elif table == "actions":
            rows = await self._db.search_actions(query, limit=n)
        else:
            return json.dumps({"error": f"Unknown table: {table}. Use: ocr, events, context, actions"})
        return json.dumps(rows, default=str)

    async def _db_raw_sql(self, sql: str = "", limit: str = "20") -> str:
        rows = await self._db.execute_raw_sql(sql, limit=min(int(limit), 100))
        return json.dumps(rows, default=str)

    async def _contact_lookup(self, name: str = "") -> str:
        ocr_hits = await self._db.search(name, limit=5)
        event_hits = await self._db.search_events(name, limit=5)
        return json.dumps({
            "ocr_mentions": ocr_hits,
            "event_mentions": event_hits,
        }, default=str)
