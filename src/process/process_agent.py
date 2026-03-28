from __future__ import annotations

from abc import ABC, abstractmethod

from PIL import Image

from src.storage.models import Event


class ProcessAgent(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    async def process(
        self,
        image: Image.Image,
        ocr_text: str,
        app_name: str | None,
        window_name: str | None,
    ) -> Event | None:
        ...


class NoOpAgent(ProcessAgent):
    @property
    def name(self) -> str:
        return "noop"

    async def process(
        self,
        image: Image.Image,
        ocr_text: str,
        app_name: str | None,
        window_name: str | None,
    ) -> Event | None:
        return None
