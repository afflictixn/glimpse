from __future__ import annotations

from abc import ABC, abstractmethod

from src.storage.models import AdditionalContext


class ContextProvider(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    async def collect(
        self,
        app_name: str | None,
        window_name: str | None,
    ) -> list[AdditionalContext]:
        ...
