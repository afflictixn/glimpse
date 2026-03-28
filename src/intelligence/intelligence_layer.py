from __future__ import annotations

import asyncio
import logging

from typing import TYPE_CHECKING

from src.intelligence.reasoning_agent import ReasoningAgent
from src.storage.database import DatabaseManager
from src.storage.models import Action, Event

if TYPE_CHECKING:
    from src.general_agent.agent import GeneralAgent

logger = logging.getLogger(__name__)


class IntelligenceLayer:
    def __init__(
        self,
        agents: list[ReasoningAgent],
        db: DatabaseManager,
        general_agent: GeneralAgent | None = None,
    ) -> None:
        self._agents = agents
        self._db = db
        self._general_agent = general_agent
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
                        if self._general_agent is not None:
                            await self._general_agent.push("action", {
                                "event_id": result.event_id,
                                "frame_id": result.frame_id,
                                "agent_name": result.agent_name,
                                "action_type": result.action_type,
                                "action_description": result.action_description,
                                "metadata": result.metadata,
                            })
                    except Exception:
                        logger.error(
                            "Failed to insert action from %s",
                            self._agents[i].name,
                            exc_info=True,
                        )

    async def stop(self) -> None:
        self._running = False
