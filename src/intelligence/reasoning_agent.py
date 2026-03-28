from __future__ import annotations

from abc import ABC, abstractmethod

from src.storage.database import DatabaseManager
from src.storage.models import Action, Event


class ReasoningAgent(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    async def reason(self, event: Event, db: DatabaseManager) -> Action | None:
        ...
