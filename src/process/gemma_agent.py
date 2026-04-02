from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import urllib.request

from PIL import Image

from src.process.process_agent import ProcessAgent
from src.process.vision_shared import (
    VISION_SYSTEM_PROMPT,
    build_vision_prompt,
    parse_vision_response,
)
from src.storage.models import AppType, Event

logger = logging.getLogger(__name__)

_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "app_type": {
            "type": "string",
            "enum": [v.value for v in AppType],
        },
        "summary": {"type": "string"},
        "metadata": {"type": "object"},
    },
    "required": ["app_type", "summary"],
}


class GemmaAgent(ProcessAgent):
    """Model-agnostic Ollama vision ProcessAgent (works with gemma3, qwen3-vl, etc.)."""

    def __init__(
        self,
        *,
        ollama_base_url: str = "http://localhost:11434",
        model: str = "gemma3:4b",
        include_ocr: bool = False,
        timeout_s: int = 30,
        max_image_width: int = 960,
    ) -> None:
        self._base_url = ollama_base_url.rstrip("/")
        self._model = model
        self._include_ocr = include_ocr
        self._timeout_s = timeout_s
        self._max_width = max_image_width

    @property
    def name(self) -> str:
        prefix = self._model.split(":")[0].replace("-", "_")
        return prefix

    async def process(
        self,
        image: Image.Image,
        ocr_text: str,
        app_name: str | None,
        window_name: str | None,
    ) -> Event | None:
        prompt = build_vision_prompt(
            ocr_text, app_name, window_name, include_ocr=self._include_ocr,
        )
        image_b64 = await asyncio.to_thread(self._encode_image, image)

        try:
            raw = await asyncio.to_thread(self._call_ollama, prompt, image_b64)
        except Exception:
            logger.error("Ollama request to %s failed", self._model, exc_info=True)
            return None

        return parse_vision_response(raw, agent_name=self.name)

    def _encode_image(self, image: Image.Image) -> str:
        rgb = image.convert("RGB") if image.mode != "RGB" else image
        if self._max_width and rgb.width > self._max_width:
            ratio = self._max_width / rgb.width
            rgb = rgb.resize(
                (self._max_width, int(rgb.height * ratio)),
                Image.Resampling.LANCZOS,
            )
        buf = io.BytesIO()
        rgb.save(buf, format="JPEG", quality=70)
        return base64.b64encode(buf.getvalue()).decode()

    def _call_ollama(self, prompt: str, image_b64: str) -> str:
        payload = json.dumps({
            "model": self._model,
            "prompt": prompt,
            "system": VISION_SYSTEM_PROMPT,
            "images": [image_b64],
            "stream": False,
            "format": _OUTPUT_SCHEMA,
            "options": {"temperature": 0.1},
        }).encode()

        req = urllib.request.Request(
            f"{self._base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:
            data = json.loads(resp.read())

        return data.get("response", "")
