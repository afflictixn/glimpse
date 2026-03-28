"""CritiqueReasoningAgent — analyzes browser page content for inconsistencies and red flags."""
from __future__ import annotations

import json
import logging

from src.intelligence.reasoning_agent import ReasoningAgent
from src.llm.client import LLMClient
from src.llm.types import Message
from src.storage.database import DatabaseManager
from src.storage.models import Action, Event

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a consumer protection analyst. Given a web page's content, identify:
- Low reviews and small amount of reviews
- Specific bad reviews mentioning defects, unexpected info about the product, even if the overall reviews are good
- Inconsistencies in the product description

If you find actionable issues, respond with a JSON object:
{"findings": "concise user-facing summary", "severity": "info|warning|alert"}

If the page looks clean, respond with: {"findings": null}

Respond ONLY with valid JSON, no markdown fences or extra text."""


class CritiqueReasoningAgent(ReasoningAgent):
    """Analyzes allowlisted browser pages for inconsistencies and red flags."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    @property
    def name(self) -> str:
        return "critique"

    async def reason(self, event: Event, db: DatabaseManager) -> Action | None:
        if event.agent_name != "browser_content":
            logger.debug("Skipping event %s (agent=%s, not browser_content)", event.id, event.agent_name)
            return None

        page_text = event.metadata.get("text")
        if not page_text:
            logger.debug("Skipping browser_content event %s — no page text", event.id)
            return None

        url = event.metadata.get("url", "")
        title = event.metadata.get("title", "")
        label = event.metadata.get("allowlist_label", "")
        logger.info("Critiquing event %s: url=%s title=%.60s label=%s (%d chars)", event.id, url, title, label, len(page_text))

        visual_summary = ""
        if event.frame_id is not None:
            try:
                frame_events = await db.get_events_for_frame(event.frame_id)
                for fe in frame_events:
                    if fe.get("agent_name") == "gemma":
                        visual_summary = fe.get("summary", "")
                        break
            except Exception:
                logger.debug("Failed to fetch gemma event for frame %s", event.frame_id, exc_info=True)

        user_prompt = self._build_prompt(url, title, label, page_text, visual_summary)

        logger.debug("Sending critique prompt (%d chars) to LLM", len(user_prompt))
        try:
            response = await self._llm.complete(
                [
                    Message(role="system", content=_SYSTEM_PROMPT),
                    Message(role="user", content=user_prompt),
                ],
            )
        except Exception:
            logger.error("Critique LLM call failed for event %s", event.id, exc_info=True)
            return None

        logger.debug("LLM response for event %s: %.200s", event.id, response.content)
        action = self._parse_response(response.content, event)
        if action:
            logger.info("Critique found issues for event %s (severity=%s): %.120s", event.id, action.metadata.get("severity"), action.action_description)
        else:
            logger.debug("Critique found no issues for event %s", event.id)
        return action

    def _build_prompt(
        self,
        url: str,
        title: str,
        label: str,
        page_text: str,
        visual_summary: str,
    ) -> str:
        parts = [
            f"Page URL: {url}",
            f"Page title: {title}",
        ]
        if label:
            parts.append(f"Page type: {label}")
        if visual_summary:
            parts.append(f"Visual analysis: {visual_summary}")
        parts.append(f"\nPage content:\n{page_text[:6000]}")
        return "\n".join(parts)

    def _parse_response(self, text: str, event: Event) -> Action | None:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [ln for ln in lines if not ln.strip().startswith("```")]
            cleaned = "\n".join(lines)

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("Failed to parse critique JSON for event %s: %.200s", event.id, text)
            return None

        findings = parsed.get("findings")
        if not findings:
            return None

        severity = parsed.get("severity", "info")
        return Action(
            event_id=event.id,
            frame_id=event.frame_id,
            agent_name=self.name,
            action_type="critique",
            action_description=findings,
            metadata={
                "url": event.metadata.get("url", ""),
                "label": event.metadata.get("allowlist_label", ""),
                "severity": severity,
            },
        )
