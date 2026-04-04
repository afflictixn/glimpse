"""PresentationCritiqueAgent — visual design critique for presentations via LLM vision."""
from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import time
from pathlib import Path

from PIL import Image

from src.general_agent.ollama_client import sanitize_response
from src.intelligence.reasoning_agent import ReasoningAgent
from src.llm.client import LLMClient, create_llm_client
from src.llm.types import Message
from src.storage.database import DatabaseManager
from src.storage.models import Action, Event

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a presentation design critic. Given a description of a slide, provide \
brief, actionable feedback on visual design only. Focus exclusively on:
- Font: typeface choice, size, weight, consistency across the slide
- Color: palette harmony, contrast ratios, accessibility. When proposing a new color use a specific hex code.
- Style: layout balance, visual hierarchy, spacing, alignment

Rules:
- 2 most impactful bullet points total
- Each bullet must be one short sentence
- Only flag real problems; if the slide looks fine, say so
- Be specific (e.g. "body text is ~10pt, too small for projection") not generic
- Do NOT comment on content, grammar, or messaging — only visuals

Respond ONLY with a JSON object:
{"critique": ["point 1", "point 2"], "verdict": "clean|needs_work"}

If the design is solid, respond with:
{"critique": [], "verdict": "clean"}"""


class PresentationCritiqueAgent(ReasoningAgent):
    """Critiques presentation slide design using the configured LLM provider."""

    def __init__(
        self,
        *,
        llm: LLMClient | None = None,
        llm_provider: str = "ollama",
        model: str = "gemma3:4b",
        max_image_width: int = 1280,
        backoff_seconds: float = 30.0,
    ) -> None:
        self._llm = llm or create_llm_client(llm_provider, model)
        self._max_width = max_image_width
        self._backoff_seconds = backoff_seconds
        self._last_process_time: float = 0.0

    @property
    def name(self) -> str:
        return "presentation_critique"

    _PRESENTATION_APPS = frozenset({"Keynote"})

    async def reason(self, event: Event, db: DatabaseManager) -> Action | None:
        if event.frame_id is None:
            logger.debug("Skipping event with no frame_id")
            return None

        frame = await db.get_frame(event.frame_id)
        app_name = frame.get("app_name", "") if frame else ""
        if app_name not in self._PRESENTATION_APPS:
            return None

        elapsed = time.monotonic() - self._last_process_time
        if elapsed < self._backoff_seconds:
            logger.debug("Backoff: skipping presentation event (%.1fs remaining)", self._backoff_seconds - elapsed)
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

        self._last_process_time = time.monotonic()
        try:
            raw = await self._analyze(image)
        except Exception:
            logger.error("Presentation critique LLM request failed", exc_info=True)
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
        """Encode image to base64 for inclusion in prompt."""
        rgb = image.convert("RGB") if image.mode != "RGB" else image
        if self._max_width and rgb.width > self._max_width:
            ratio = self._max_width / rgb.width
            rgb = rgb.resize(
                (self._max_width, int(rgb.height * ratio)),
                Image.Resampling.LANCZOS,
            )
        buf = io.BytesIO()
        rgb.save(buf, format="JPEG", quality=80)
        return base64.b64encode(buf.getvalue()).decode()

    async def _analyze(self, image: Image.Image) -> str:
        """Send slide to LLM for critique."""
        image_b64 = await asyncio.to_thread(self._encode_image, image)

        messages = [
            Message(role="system", content=_SYSTEM_PROMPT),
            Message(
                role="user",
                content=f"[Image of presentation slide (base64): {image_b64[:100]}...]\n\n"
                        "Critique this presentation slide's visual design: font, font size, color, and style only.",
            ),
        ]

        response = await asyncio.wait_for(
            self._llm.complete(messages),
            timeout=15,
        )
        return response.content

    def _parse_response(self, text: str, event: Event) -> Action | None:
        cleaned = sanitize_response(text).strip()
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
