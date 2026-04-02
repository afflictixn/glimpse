"""GeminiVisionAgent — fast cloud-based screen activity analyzer via Gemini Flash."""
from __future__ import annotations

import asyncio
import io
import logging

from google import genai
from google.genai import types as gtypes
from PIL import Image

from src.process.process_agent import ProcessAgent
from src.process.vision_shared import (
    VISION_SYSTEM_PROMPT,
    LlmTokenCounter,
    build_vision_prompt,
    parse_vision_response,
)
from src.storage.models import AppType, Event

logger = logging.getLogger(__name__)

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
        token_counter: LlmTokenCounter | None = None,
    ) -> None:
        self._model = model
        self._include_ocr = include_ocr
        self._max_width = max_image_width
        self._client = genai.Client(api_key=api_key)
        self._token_counter = token_counter

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
        prompt = build_vision_prompt(
            ocr_text, app_name, window_name, include_ocr=self._include_ocr,
        )
        image_bytes = await asyncio.to_thread(self._encode_image, image)

        try:
            raw = await self._call_gemini(image_bytes, prompt)
        except Exception:
            logger.error("Gemini vision request failed", exc_info=True)
            return None

        return parse_vision_response(raw, agent_name=self.name)

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
            system_instruction=VISION_SYSTEM_PROMPT,
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

        if self._token_counter and response.usage_metadata:
            self._token_counter.record(
                self._model,
                input_tokens=response.usage_metadata.prompt_token_count or 0,
                output_tokens=response.usage_metadata.candidates_token_count or 0,
            )

        return response.text or ""
