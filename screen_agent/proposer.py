"""Generate actionable proposals based on screen analysis results."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import requests

from screen_agent.analyzer import AnalysisResult
from screen_agent.config import Config

log = logging.getLogger(__name__)

PROPOSE_SYSTEM = """\
You are a proactive personal assistant. The user is working on their computer \
and you have been watching what they do. Based on the current and recent \
activity, suggest ONE concise, helpful action you could take for them.

Rules:
- Be specific to what you see (don't be generic).
- Phrase it as a friendly question or offer.
- Keep it to 1‑2 sentences.
- Reply with ONLY a JSON object (no markdown fences):
{
  "proposal": "<your suggestion>",
  "action_type": "<one of: search, list, summarise, compare, save, remind, explain, other>",
  "confidence": <0.0-1.0 how confident you are this is useful>
}
"""


@dataclass
class Proposal:
    text: str
    action_type: str
    confidence: float
    source_analysis: AnalysisResult


class ActionProposer:
    """Uses the local model to craft a proposal from an analysis result."""

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._api = f"{cfg.ollama_url}/api/chat"

    def propose(
        self,
        current: AnalysisResult,
        recent: list[AnalysisResult],
    ) -> Proposal | None:
        if current.score < self._cfg.interest_threshold:
            return None

        history = "\n".join(
            f"- [{r.category}] {r.description}" for r in recent[-5:]
        )
        user_msg = (
            f"Recent user activity:\n{history}\n\n"
            f"Current activity (score {current.score}/5, "
            f"category: {current.category}):\n{current.description}"
        )

        body = {
            "model": self._cfg.vision_model,
            "messages": [
                {"role": "system", "content": PROPOSE_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            "stream": False,
            "format": "json",
        }

        try:
            resp = requests.post(
                self._api, json=body, timeout=self._cfg.ollama_timeout,
            )
            resp.raise_for_status()
            data = self._parse_json(resp.json()["message"]["content"])
        except Exception:
            log.exception("Proposal generation failed")
            return None

        return Proposal(
            text=data.get("proposal", ""),
            action_type=data.get("action_type", "other"),
            confidence=float(data.get("confidence", 0.5)),
            source_analysis=current,
        )

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            log.warning("Failed to parse proposal JSON: %s", text[:200])
            return {"proposal": text[:200], "action_type": "other", "confidence": 0.3}
