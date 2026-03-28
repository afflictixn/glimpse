"""Local vision analysis via Ollama — describe the screen and classify interest."""

from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

import requests
from PIL import Image

from screen_agent.capture import ScreenCapture
from screen_agent.config import Config

log = logging.getLogger(__name__)

DESCRIBE_SYSTEM = """\
You are an observant screen analyst. Given a screenshot of a user's screen, \
describe what the user is currently doing. Be specific: mention the \
application, visible text, URLs, product names, or page titles when possible. \
Keep your description to 2‑3 sentences."""

CLASSIFY_SYSTEM = """\
You are an interest classifier. Given a description of what a user is doing \
on their screen, decide whether this activity is *interesting or actionable* \
— meaning you could suggest a helpful follow-up action.

Reply with ONLY a JSON object (no markdown fences):
{
  "score": <1-5 integer>,
  "category": "<one of: CATEGORIES>",
  "summary": "<one-sentence summary of why this is or isn't interesting>"
}

Score guide:
1 = mundane (idle desktop, lock screen)
2 = mildly notable (general browsing, social media scrolling)
3 = somewhat actionable (reading an article, watching a tutorial)
4 = clearly actionable (viewing a recipe, shopping, researching a topic)
5 = highly actionable (comparing prices, filling a form, debugging code)
"""


@dataclass
class AnalysisResult:
    description: str
    score: int = 1
    category: str = "other"
    summary: str = ""
    raw_classify: dict[str, Any] = field(default_factory=dict)


class LocalAnalyzer:
    """Two-stage local analysis: describe then classify."""

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._api = f"{cfg.ollama_url}/api/chat"
        self._context_buffer: deque[AnalysisResult] = deque(
            maxlen=cfg.context_buffer_size,
        )

    @property
    def recent_context(self) -> list[AnalysisResult]:
        return list(self._context_buffer)

    def analyze(self, img: Image.Image) -> AnalysisResult:
        """Run describe → classify pipeline and return structured result."""
        b64 = ScreenCapture.image_to_base64(img)

        description = self._describe(b64)
        classification = self._classify(description)

        result = AnalysisResult(
            description=description,
            score=classification.get("score", 1),
            category=classification.get("category", "other"),
            summary=classification.get("summary", ""),
            raw_classify=classification,
        )
        self._context_buffer.append(result)
        return result

    def _describe(self, image_b64: str) -> str:
        body = {
            "model": self._cfg.vision_model,
            "messages": [
                {"role": "system", "content": DESCRIBE_SYSTEM},
                {
                    "role": "user",
                    "content": "Describe what the user is doing in this screenshot.",
                    "images": [image_b64],
                },
            ],
            "stream": False,
        }
        resp = requests.post(self._api, json=body, timeout=self._cfg.ollama_timeout)
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()

    def _classify(self, description: str) -> dict[str, Any]:
        cats = ", ".join(self._cfg.categories)
        system = CLASSIFY_SYSTEM.replace("CATEGORIES", cats)

        context_lines = []
        for prev in self._context_buffer:
            context_lines.append(f"- {prev.description} [score={prev.score}]")
        context_block = (
            "Recent activity:\n" + "\n".join(context_lines[-5:])
            if context_lines
            else ""
        )

        user_msg = f"{context_block}\n\nCurrent activity: {description}".strip()

        body = {
            "model": self._cfg.vision_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            "stream": False,
            "format": "json",
        }
        resp = requests.post(self._api, json=body, timeout=self._cfg.ollama_timeout)
        resp.raise_for_status()
        raw = resp.json()["message"]["content"].strip()
        return self._parse_json(raw)

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            log.warning("Failed to parse classifier JSON: %s", text[:200])
            return {"score": 1, "category": "other", "summary": text[:120]}
