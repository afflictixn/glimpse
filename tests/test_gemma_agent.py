"""Integration test that hits a real local Ollama instance with GemmaAgent.

Run with:
    pytest tests/test_gemma_agent.py -v -s

Requires Ollama running locally with a gemma3 model pulled.
Skipped automatically if Ollama is not reachable.
"""

from __future__ import annotations

import urllib.error
import urllib.request

import pytest
import pytest_asyncio
from PIL import Image

from src.process.gemma_agent import GemmaAgent
from src.storage.models import AppType, Event
from src.config import Settings

_DEFAULTS = Settings()
OLLAMA_URL = _DEFAULTS.ollama_base_url
MODEL = _DEFAULTS.ollama_model


def _ollama_reachable() -> bool:
    try:
        urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=3)
        return True
    except (urllib.error.URLError, OSError):
        return False


requires_ollama = pytest.mark.skipif(
    not _ollama_reachable(),
    reason="Ollama not running at " + OLLAMA_URL,
)


def _make_test_image(width: int = 800, height: int = 600) -> Image.Image:
    """Synthetic desktop-like image: white background, dark rectangle as a 'window'."""
    img = Image.new("RGB", (width, height), (230, 230, 230))
    for x in range(100, 700):
        for y in range(80, 500):
            img.putpixel((x, y), (40, 40, 50))
    for x in range(120, 680):
        for y in range(120, 480):
            img.putpixel((x, y), (255, 255, 255))
    return img


@requires_ollama
class TestGemmaAgentLive:
    """Tests that actually call the local Ollama server."""

    @pytest.mark.asyncio
    async def test_process_returns_event(self):
        agent = GemmaAgent(
            ollama_base_url=OLLAMA_URL,
            model=MODEL,
            include_ocr=False,
            timeout_s=60,
        )

        image = _make_test_image()
        event = await agent.process(
            image=image,
            ocr_text="",
            app_name="TestApp",
            window_name="main.py",
        )

        assert event is not None, "Agent returned None — Ollama may have failed"
        assert isinstance(event, Event)
        assert event.agent_name == "gemma3"
        assert event.app_type != ""
        assert event.summary != ""
        print(f"\n  app_type : {event.app_type}")
        print(f"  summary  : {event.summary}")
        print(f"  metadata : {event.metadata}")

    @pytest.mark.asyncio
    async def test_process_with_ocr(self):
        agent = GemmaAgent(
            ollama_base_url=OLLAMA_URL,
            model=MODEL,
            include_ocr=True,
            timeout_s=60,
        )

        image = _make_test_image()
        event = await agent.process(
            image=image,
            ocr_text="def hello():\n    print('world')",
            app_name="Cursor",
            window_name="hello.py — Cursor",
        )

        assert event is not None
        assert isinstance(event, Event)
        assert event.agent_name == "gemma3"
        assert event.summary != ""
        print(f"\n  app_type : {event.app_type}")
        print(f"  summary  : {event.summary}")
        print(f"  metadata : {event.metadata}")

    @pytest.mark.asyncio
    async def test_process_no_context(self):
        """Minimal call — no app name, no window, no OCR."""
        agent = GemmaAgent(
            ollama_base_url=OLLAMA_URL,
            model=MODEL,
            timeout_s=60,
        )

        image = _make_test_image()
        event = await agent.process(
            image=image,
            ocr_text="",
            app_name=None,
            window_name=None,
        )

        assert event is not None
        assert isinstance(event, Event)
        print(f"\n  app_type : {event.app_type}")
        print(f"  summary  : {event.summary}")


class TestGemmaAgentUnit:
    """Fast unit tests that don't need Ollama running."""

    def test_build_prompt_no_ocr(self):
        agent = GemmaAgent(include_ocr=False)
        prompt = agent._build_prompt("some ocr text", "Safari", "Google")
        assert "Analyze this screenshot." in prompt
        assert "Active app: Safari" in prompt
        assert "Window: Google" in prompt
        assert "OCR text" not in prompt

    def test_build_prompt_with_ocr(self):
        agent = GemmaAgent(include_ocr=True)
        prompt = agent._build_prompt("hello world", "Code", None)
        assert "OCR text:" in prompt
        assert "hello world" in prompt

    def test_build_prompt_ocr_truncation(self):
        agent = GemmaAgent(include_ocr=True)
        long_text = "x" * 5000
        prompt = agent._build_prompt(long_text, None, None)
        assert len(prompt) < 2200

    def test_parse_valid_json(self):
        agent = GemmaAgent()
        event = agent._parse_response(
            '{"app_type": "ide", "summary": "Editing code", "metadata": {"language": "python"}}'
        )
        assert event is not None
        assert event.app_type is AppType.IDE
        assert event.summary == "Editing code"
        assert event.metadata == {"language": "python"}

    def test_parse_fenced_json_falls_back(self):
        """Structured output means we no longer strip markdown fences."""
        agent = GemmaAgent()
        event = agent._parse_response(
            '```json\n{"app_type": "browser", "summary": "Browsing web"}\n```'
        )
        assert event is not None
        assert event.app_type is AppType.OTHER
        assert "```" in event.summary

    def test_parse_invalid_json_fallback(self):
        agent = GemmaAgent()
        event = agent._parse_response("I cannot parse this image properly.")
        assert event is not None
        assert event.app_type is AppType.OTHER
        assert "cannot parse" in event.summary

    def test_parse_unknown_app_type_coerced_to_other(self):
        agent = GemmaAgent()
        event = agent._parse_response(
            '{"app_type": "spreadsheet", "summary": "Editing cells"}'
        )
        assert event is not None
        assert event.app_type is AppType.OTHER

    def test_encode_image_returns_base64(self):
        agent = GemmaAgent()
        img = Image.new("RGB", (100, 100), (255, 0, 0))
        b64 = agent._encode_image(img)
        assert isinstance(b64, str)
        assert len(b64) > 100

    def test_encode_rgba_image(self):
        agent = GemmaAgent()
        img = Image.new("RGBA", (100, 100), (255, 0, 0, 128))
        b64 = agent._encode_image(img)
        assert isinstance(b64, str)

    def test_encode_image_downscales(self):
        agent = GemmaAgent(max_image_width=200)
        img = Image.new("RGB", (400, 300), (255, 0, 0))
        b64 = agent._encode_image(img)
        assert isinstance(b64, str)
        assert len(b64) > 50

    @pytest.mark.asyncio
    async def test_unreachable_ollama_returns_none(self):
        agent = GemmaAgent(
            ollama_base_url="http://localhost:1",
            timeout_s=2,
        )
        img = Image.new("RGB", (100, 100))
        event = await agent.process(img, "", None, None)
        assert event is None
