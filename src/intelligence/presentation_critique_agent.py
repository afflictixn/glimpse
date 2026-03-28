"""PresentationCritiqueAgent — visual design critique for presentations via Ollama vision."""
from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image

from src.intelligence.reasoning_agent import ReasoningAgent
from src.storage.database import DatabaseManager
from src.storage.models import Action, Event

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a presentation design critic. Given a screenshot of a slide, provide \
brief, actionable feedback on visual design only. Focus on:
- Font choices and readability (size, weight, consistency)
- Color palette (contrast, harmony, accessibility)
- Overall layout and visual hierarchy

Rules:
- Maximum 3 bullet points total
- Each bullet must be one sentence
- Only flag real problems; if the slide looks fine, say so
- Be direct and specific, not generic

Respond ONLY with a JSON object:
{"critique": ["point 1", "point 2"], "verdict": "clean|needs_work"}

If the design is solid, respond with:
{"critique": [], "verdict": "clean"}"""


class PresentationCritiqueAgent(ReasoningAgent):
    """Critiques presentation slide design using Ollama vision model."""

    def __init__(
        self,
        *,
        ollama_base_url: str = "http://localhost:11434",
        model: str = "gemma3:4b",
        timeout_s: int = 30,
        max_image_width: int = 960,
    ) -> None:
        self._base_url = ollama_base_url.rstrip("/")
        self._model = model
        self._timeout_s = timeout_s
        self._max_width = max_image_width

    @property
    def name(self) -> str:
        return "presentation_critique"

    async def reason(self, event: Event, db: DatabaseManager) -> Action | None:
        if event.app_type != "presentation":
            return None

        if event.frame_id is None:
            logger.debug("Skipping presentation event with no frame_id")
            return None

        snapshot_path = await self._get_snapshot_path(event.frame_id, db)
        if not snapshot_path:
            logger.debug("No snapshot found for frame %s", event.frame_id)
            return None

        try:
            image = await asyncio.to_thread(Image.open, snapshot_path)
        except Exception:
            logger.error("Failed to load snapshot %s", snapshot_path, exc_info=True)
            return None

        image_b64 = await asyncio.to_thread(self._encode_image, image)

        try:
            raw = await asyncio.to_thread(self._call_ollama, image_b64)
        except Exception:
            logger.error("Ollama critique request failed", exc_info=True)
            return None

        return self._parse_response(raw, event)

    async def _get_snapshot_path(self, frame_id: int, db: DatabaseManager) -> str | None:
        try:
            frame = await db.get_frame(frame_id)
        except Exception:
            logger.debug("Failed to fetch frame %s", frame_id, exc_info=True)
            return None
        if not frame:
            return None
        path = frame.get("snapshot_path")
        if path and Path(path).is_file():
            return path
        return None

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

    def _call_ollama(self, image_b64: str) -> str:
        payload = json.dumps({
            "model": self._model,
            "prompt": "Critique this presentation slide's visual design.",
            "system": _SYSTEM_PROMPT,
            "images": [image_b64],
            "stream": False,
            "format": {
                "type": "object",
                "properties": {
                    "critique": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "verdict": {
                        "type": "string",
                        "enum": ["clean", "needs_work"],
                    },
                },
                "required": ["critique", "verdict"],
            },
            "options": {"temperature": 0.2},
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

    def _parse_response(self, text: str, event: Event) -> Action | None:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [ln for ln in lines if not ln.strip().startswith("```")]
            cleaned = "\n".join(lines)

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("Failed to parse presentation critique JSON: %.200s", text)
            return None

        critique = parsed.get("critique", [])
        verdict = parsed.get("verdict", "clean")

        if not critique or verdict == "clean":
            return None

        description = " | ".join(critique[:3])
        return Action(
            event_id=event.id,
            frame_id=event.frame_id,
            agent_name=self.name,
            action_type="presentation_critique",
            action_description=description,
            metadata={
                "critique": critique[:3],
                "verdict": verdict,
                "app_name": event.metadata.get("app_name", ""),
            },
        )
