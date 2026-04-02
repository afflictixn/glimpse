"""OpenAIVisionAgent — fast cloud vision ProcessAgent via OpenAI (gpt-5.4-nano)."""
from __future__ import annotations

import asyncio
import base64
import io
import logging

from openai import AsyncOpenAI
from PIL import Image

from src.process.process_agent import ProcessAgent
from src.process.vision_shared import (
    VISION_SYSTEM_PROMPT,
    LlmTokenCounter,
    ScreenActivity,
    build_vision_prompt,
    screen_activity_to_event,
)
from src.storage.models import AppType, Event

logger = logging.getLogger(__name__)

_OPENAI_SYSTEM_PROMPT = (
    VISION_SYSTEM_PROMPT
    + "\n\nRespond with a JSON object containing exactly these fields:\n"
    + '- "app_type": one of ' + ", ".join(f'"{v.value}"' for v in AppType) + "\n"
    + '- "summary": one-sentence description\n'
    + '- "metadata": object with string key/value observations'
)


class OpenAIVisionAgent(ProcessAgent):
    """Cloud vision ProcessAgent using the OpenAI API with detail=low for fast inference."""

    def __init__(
        self,
        *,
        model: str = "gpt-5.4-nano",
        include_ocr: bool = False,
        max_image_width: int = 960,
        image_detail: str = "low",
        api_key: str | None = None,
        timeout_s: float = 5.0,
        token_counter: LlmTokenCounter | None = None,
    ) -> None:
        self._model = model
        self._include_ocr = include_ocr
        self._max_width = max_image_width
        self._detail = image_detail
        self._timeout = timeout_s
        self._client = AsyncOpenAI(api_key=api_key)
        self._token_counter = token_counter

    @property
    def name(self) -> str:
        return "openai_vision"

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
            activity = await self._call_openai(image_b64, prompt)
        except Exception:
            logger.error("OpenAI vision request failed", exc_info=True)
            return None

        if activity is None:
            return None
        return screen_activity_to_event(activity, agent_name=self.name)

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

    async def _call_openai(
        self, image_b64: str, prompt: str,
    ) -> ScreenActivity | None:
        input_messages = [
            {"role": "system", "content": _OPENAI_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{image_b64}",
                        "detail": self._detail,
                    },
                    {"type": "input_text", "text": prompt},
                ],
            },
        ]

        response = await asyncio.wait_for(
            self._client.responses.create(
                model=self._model,
                input=input_messages,
                text={"format": {"type": "json_object"}},
                temperature=0.2,
            ),
            timeout=self._timeout,
        )

        if self._token_counter and response.usage:
            self._token_counter.record(
                self._model,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )

        raw = response.output_text
        if not raw:
            logger.warning("OpenAI returned empty response")
            return None

        try:
            return ScreenActivity.model_validate_json(raw)
        except Exception:
            logger.warning("Failed to validate OpenAI response as ScreenActivity: %.200s", raw)
            return None
