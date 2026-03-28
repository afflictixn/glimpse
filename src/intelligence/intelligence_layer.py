from __future__ import annotations

import asyncio
import logging

from src.intelligence.reasoning_agent import ReasoningAgent
from src.storage.database import DatabaseManager
from src.storage.models import Action, Event

logger = logging.getLogger(__name__)


class IntelligenceLayer:
    def __init__(
        self,
        agents: list[ReasoningAgent],
        db: DatabaseManager,
    ) -> None:
        self._agents = agents
        self._db = db
        self._event_queue: asyncio.Queue[Event] = asyncio.Queue()
        self._running = False

    async def submit(self, event: Event) -> None:
        await self._event_queue.put(event)

    async def run(self) -> None:
        self._running = True
        logger.info(
            "IntelligenceLayer started with %d reasoning agent(s)", len(self._agents)
        )
        while self._running:
            try:
                event = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            if not self._agents:
                continue

            results = await asyncio.gather(
                *(agent.reason(event, self._db) for agent in self._agents),
                return_exceptions=True,
            )
            for i, result in enumerate(results):
                if isinstance(result, BaseException):
                    logger.error(
                        "ReasoningAgent %s failed: %s",
                        self._agents[i].name,
                        result,
                        exc_info=result,
                    )
                elif isinstance(result, Action):
                    try:
                        await self._db.insert_action(result)
                    except Exception:
                        logger.error(
                            "Failed to insert action from %s",
                            self._agents[i].name,
                            exc_info=True,
                        )

    async def stop(self) -> None:
        self._running = False
