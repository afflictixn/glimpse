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

_APP_TYPES = ", ".join(f'"{v.value}"' for v in AppType)

_SYSTEM_PROMPT = f"""\
You are a screen activity analyzer. Given a screenshot (and optionally OCR text), \
produce a JSON object with these fields:
- "app_type": one of {_APP_TYPES}
- "summary": one-sentence description of what the user is doing
- "metadata": object with any additional observations (e.g. {{"url": "...", \
"language": "python", "topic": "..."}})

Respond ONLY with valid JSON, no markdown fences or extra text."""


class GemmaAgent(ProcessAgent):
    """ProcessAgent that sends screenshots to a local Gemma 3 12B via Ollama."""

    def __init__(
        self,
        *,
        ollama_base_url: str = "http://localhost:11434",
        model: str = "gemma3:12b",
        include_ocr: bool = False,
        timeout_s: int = 30,
    ) -> None:
        self._base_url = ollama_base_url.rstrip("/")
        self._model = model
        self._include_ocr = include_ocr
        self._timeout_s = timeout_s

    @property
    def name(self) -> str:
        return "gemma"

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

    @staticmethod
    def _encode_image(image: Image.Image) -> str:
        buf = io.BytesIO()
        rgb = image.convert("RGB") if image.mode != "RGB" else image
        rgb.save(buf, format="JPEG", quality=70)
        return base64.b64encode(buf.getvalue()).decode()

    def _call_ollama(self, prompt: str, image_b64: str) -> str:
        payload = json.dumps({
            "model": self._model,
            "prompt": prompt,
            "system": _SYSTEM_PROMPT,
            "images": [image_b64],
            "stream": False,
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

    @staticmethod
    def _coerce_app_type(raw: str) -> AppType:
        try:
            return AppType(raw.lower())
        except ValueError:
            return AppType.OTHER

    def _parse_response(self, text: str) -> Event | None:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [ln for ln in lines if not ln.strip().startswith("```")]
            cleaned = "\n".join(lines)

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("Failed to parse Gemma JSON response: %.200s", text)
            return Event(
                agent_name=self.name,
                app_type=AppType.OTHER,
                summary=text[:500],
            )

        return Event(
            agent_name=self.name,
            app_type=self._coerce_app_type(parsed.get("app_type", "other")),
            summary=parsed.get("summary", ""),
            metadata=parsed.get("metadata", {}),
        )
