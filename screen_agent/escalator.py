"""Escalation to a strong reasoning API (Gemini or OpenAI) for deeper analysis."""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Optional

from screen_agent.analyzer import AnalysisResult
from screen_agent.config import Config
from screen_agent.memory import Memory

log = logging.getLogger(__name__)

META_PROMPT = """\
You are a creative and resourceful personal assistant. The user has been \
working on their computer and a background agent has been watching their \
screen activity. You are given a summary of their recent activity plus the \
agent's original suggestion.

Your job:
1. Provide deeper insight, tips, or creative ideas related to what the user \
   is doing.
2. Be conversational and engaging — like a knowledgeable friend looking over \
   their shoulder.
3. If relevant, suggest next steps, resources, or things the user might not \
   have thought of.
4. Keep your reply concise (3‑5 sentences) unless the user asks for more \
   detail.
"""


# ---------------------------------------------------------------------------
# Provider abstraction
# ---------------------------------------------------------------------------

class _LLMProvider(ABC):
    """Thin wrapper so the Escalator doesn't care which API is behind it."""

    @property
    @abstractmethod
    def display_name(self) -> str: ...

    @abstractmethod
    def chat(self, history: list[dict[str, str]]) -> str:
        """Send a conversation and return the assistant reply."""

    @abstractmethod
    def is_available(self) -> bool: ...


class _GeminiProvider(_LLMProvider):
    def __init__(self, model: str) -> None:
        from google import genai
        from google.genai import types

        self._model = model
        self._types = types
        api_key = os.environ.get("GEMINI_API_KEY", "")
        self._client = genai.Client(api_key=api_key) if api_key else None

    @property
    def display_name(self) -> str:
        return "Gemini"

    def is_available(self) -> bool:
        return self._client is not None

    def chat(self, history: list[dict[str, str]]) -> str:
        assert self._client is not None
        types = self._types
        contents = []
        for msg in history:
            role = "model" if msg["role"] == "assistant" else "user"
            contents.append(
                types.Content(role=role, parts=[types.Part(text=msg["content"])]),
            )
        resp = self._client.models.generate_content(
            model=self._model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=META_PROMPT,
                temperature=0.8,
                max_output_tokens=1024,
            ),
        )
        return resp.text or ""


class _OpenAIProvider(_LLMProvider):
    def __init__(self, model: str) -> None:
        from openai import OpenAI

        self._model = model
        api_key = os.environ.get("OPENAI_API_KEY", "")
        self._client = OpenAI(api_key=api_key) if api_key else None

    @property
    def display_name(self) -> str:
        return "OpenAI"

    def is_available(self) -> bool:
        return self._client is not None

    def chat(self, history: list[dict[str, str]]) -> str:
        assert self._client is not None
        messages: list[dict[str, str]] = [
            {"role": "system", "content": META_PROMPT},
        ]
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore[arg-type]
            temperature=0.8,
            max_tokens=1024,
        )
        return resp.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Public Escalator
# ---------------------------------------------------------------------------

def _resolve_provider(cfg: Config) -> _LLMProvider:
    """Pick the right provider based on config + available API keys."""
    choice = cfg.escalation_provider

    if choice == "openai":
        return _OpenAIProvider(cfg.openai_model)
    if choice == "gemini":
        return _GeminiProvider(cfg.gemini_model)

    # auto: prefer whichever key is set (OpenAI first if both present)
    if os.environ.get("OPENAI_API_KEY"):
        return _OpenAIProvider(cfg.openai_model)
    if os.environ.get("GEMINI_API_KEY"):
        return _GeminiProvider(cfg.gemini_model)

    log.warning(
        "No escalation API key found. Set OPENAI_API_KEY or GEMINI_API_KEY."
    )
    return _GeminiProvider(cfg.gemini_model)  # will fail at call time


class Escalator:
    """Sends context bundles to a strong reasoning API and manages conversation."""

    def __init__(self, cfg: Config, memory: Memory) -> None:
        self._cfg = cfg
        self._memory = memory
        self._provider = _resolve_provider(cfg)
        self._active_history: list[dict[str, str]] = []
        self._active_proposal_id: Optional[int] = None

        log.info("Escalation provider: %s", self._provider.display_name)

    @property
    def provider_name(self) -> str:
        return self._provider.display_name

    def escalate(
        self,
        proposal_text: str,
        current: AnalysisResult,
        recent: list[AnalysisResult],
        proposal_id: Optional[int] = None,
    ) -> str:
        """Send context to the API and return the full response."""
        self._active_proposal_id = proposal_id
        context = self._build_context(proposal_text, current, recent)

        self._active_history = [{"role": "user", "content": context}]

        reply = self._provider.chat(self._active_history)
        self._active_history.append({"role": "assistant", "content": reply})

        if proposal_id is not None:
            self._memory.save_conversation_turn("user", context, proposal_id)
            self._memory.save_conversation_turn("assistant", reply, proposal_id)

        return reply

    def follow_up(self, user_message: str) -> str:
        """Continue the conversation thread with a follow-up."""
        self._active_history.append({"role": "user", "content": user_message})

        reply = self._provider.chat(self._active_history)
        self._active_history.append({"role": "assistant", "content": reply})

        if self._active_proposal_id is not None:
            self._memory.save_conversation_turn(
                "user", user_message, self._active_proposal_id,
            )
            self._memory.save_conversation_turn(
                "assistant", reply, self._active_proposal_id,
            )

        return reply

    @staticmethod
    def _build_context(
        proposal_text: str,
        current: AnalysisResult,
        recent: list[AnalysisResult],
    ) -> str:
        lines = ["## Recent screen activity"]
        for r in recent[-5:]:
            lines.append(f"- [{r.category}, score {r.score}] {r.description}")
        lines.append(f"\n## Current activity\n{current.description}")
        lines.append(f"\n## Agent's suggestion\n{proposal_text}")
        lines.append(
            "\nPlease provide deeper insight, tips, or creative follow-up ideas."
        )
        return "\n".join(lines)
