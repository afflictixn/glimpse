from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import urllib.error
import urllib.request

from PIL import Image

from src.process.process_agent import ProcessAgent
from src.storage.models import AppType, Event

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a screen activity analyzer. Given a screenshot (and optionally OCR text), \
produce a JSON object describing the user's current activity. \
Keep summary to one sentence. Include 2-3 key observations in metadata."""

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
        prompt = self._build_prompt(ocr_text, app_name, window_name)
        image_b64 = await asyncio.to_thread(self._encode_image, image)

        try:
            raw = await asyncio.to_thread(self._call_ollama, prompt, image_b64)
        except Exception:
            logger.error("Ollama request to %s failed", self._model, exc_info=True)
            return None

        return self._parse_response(raw)

    def _build_prompt(
        self,
        ocr_text: str,
        app_name: str | None,
        window_name: str | None,
    ) -> str:
        parts = ["Analyze this screenshot."]
        if app_name:
            parts.append(f"Active app: {app_name}")
        if window_name:
            parts.append(f"Window: {window_name}")
        if self._include_ocr and ocr_text:
            parts.append(f"OCR text:\n{ocr_text[:2000]}")
        return "\n".join(parts)

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
            "system": _SYSTEM_PROMPT,
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

    def _parse_response(self, text: str) -> Event | None:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Failed to parse vision model JSON response: %.200s", text)
            return Event(
                agent_name=self.name,
                app_type=AppType.OTHER,
                summary=text[:500],
            )

        try:
            app_type = AppType(parsed.get("app_type", "other"))
        except ValueError:
            app_type = AppType.OTHER

        return Event(
            agent_name=self.name,
            app_type=app_type,
            summary=parsed.get("summary", ""),
            metadata=parsed.get("metadata", {}),
        )
