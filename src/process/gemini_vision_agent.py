"""GeminiVisionAgent — fast cloud-based screen activity analyzer via Gemini Flash."""
from __future__ import annotations

import asyncio
import io
import json
import logging

from google import genai
from google.genai import types as gtypes
from PIL import Image

from src.process.process_agent import ProcessAgent
from src.storage.models import AppType, Event

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a screen activity analyzer. Given a screenshot (and optionally OCR text), \
produce a JSON object describing the user's current activity. \
Keep summary to one sentence. Include 2-3 key observations in metadata."""

_RESPONSE_SCHEMA = gtypes.Schema(
    type="OBJECT",
    properties={
        "app_type": gtypes.Schema(
            type="STRING",
            enum=[v.value for v in AppType],
        ),
        "summary": gtypes.Schema(type="STRING"),
        "metadata": gtypes.Schema(type="OBJECT"),
    },
    required=["app_type", "summary"],
)


class GeminiVisionAgent(ProcessAgent):
    """Cloud vision ProcessAgent using the Gemini API for fast screen analysis."""

    def __init__(
        self,
        *,
        model: str = "gemini-2.0-flash",
        include_ocr: bool = False,
        max_image_width: int = 960,
        api_key: str | None = None,
    ) -> None:
        self._model = model
        self._include_ocr = include_ocr
        self._max_width = max_image_width
        self._client = genai.Client(api_key=api_key)

    @property
    def name(self) -> str:
        return "gemini_vision"

    async def process(
        self,
        image: Image.Image,
        ocr_text: str,
        app_name: str | None,
        window_name: str | None,
    ) -> Event | None:
        prompt = self._build_prompt(ocr_text, app_name, window_name)
        image_bytes = await asyncio.to_thread(self._encode_image, image)

        try:
            raw = await self._call_gemini(image_bytes, prompt)
        except Exception:
            logger.error("Gemini vision request failed", exc_info=True)
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

    def _encode_image(self, image: Image.Image) -> bytes:
        rgb = image.convert("RGB") if image.mode != "RGB" else image
        if self._max_width and rgb.width > self._max_width:
            ratio = self._max_width / rgb.width
            rgb = rgb.resize(
                (self._max_width, int(rgb.height * ratio)),
                Image.Resampling.LANCZOS,
            )
        buf = io.BytesIO()
        rgb.save(buf, format="JPEG", quality=70)
        return buf.getvalue()

    async def _call_gemini(self, image_bytes: bytes, prompt: str) -> str:
        image_part = gtypes.Part(
            inline_data=gtypes.Blob(mime_type="image/jpeg", data=image_bytes),
        )
        text_part = gtypes.Part(text=prompt)

        config = gtypes.GenerateContentConfig(
            system_instruction=_SYSTEM_PROMPT,
            temperature=0.2,
            response_mime_type="application/json",
            response_schema=_RESPONSE_SCHEMA,
        )

        response = await asyncio.wait_for(
            self._client.aio.models.generate_content(
                model=self._model,
                contents=[gtypes.Content(role="user", parts=[image_part, text_part])],
                config=config,
            ),
            timeout=10,
        )

        return response.text or ""

    def _parse_response(self, text: str) -> Event | None:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [ln for ln in lines if not ln.strip().startswith("```")]
            cleaned = "\n".join(lines)

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("Failed to parse Gemini vision JSON: %.200s", text)
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
