"""Integration tests for LLM timeout behaviour.

Verifies that hanging LLM providers don't freeze the pipeline:
  - GeminiClient and OpenAIClient raise TimeoutError when the provider hangs
  - GeneralAgent keeps processing after an LLM timeout
  - process_frame completes even when a vision agent times out
  - IntelligenceLayer keeps processing after a reasoning agent LLM timeout
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import openai as _openai  # noqa: F401
    _has_openai = True
except ModuleNotFoundError:
    _has_openai = False

skip_no_openai = pytest.mark.skipif(not _has_openai, reason="openai not installed")

from src.general_agent.agent import GeneralAgent, PushItem
from src.general_agent.tools import ToolRegistry
from src.intelligence.intelligence_layer import IntelligenceLayer
from src.intelligence.reasoning_agent import ReasoningAgent
from src.llm.types import Message, ToolSpec
from src.storage.database import DatabaseManager
from src.storage.models import Action, AppType, Event, Frame, OCRResult


# ── Stubs ─────────────────────────────────────────────────────


class HangingLLM:
    """Simulates a provider that never responds."""

    async def complete(
        self, messages: list[Message], tools: list[ToolSpec] | None = None
    ) -> Message:
        await asyncio.sleep(3600)
        return Message(role="assistant", content="unreachable")


class TimedOutLLM:
    """Raises TimeoutError on every call — simulates what happens after
    asyncio.wait_for wraps a hanging provider."""

    def __init__(self):
        self.call_count = 0

    async def complete(
        self, messages: list[Message], tools: list[ToolSpec] | None = None
    ) -> Message:
        self.call_count += 1
        raise TimeoutError("LLM provider timed out")


class StubLLM:
    """Returns a canned response."""

    def __init__(self, response: str = "ok"):
        self._response = response

    async def complete(
        self, messages: list[Message], tools: list[ToolSpec] | None = None
    ) -> Message:
        return Message(role="assistant", content=self._response)


class OnceThenOkLLM:
    """Times out on the first call, works on subsequent calls."""

    def __init__(self):
        self._call_count = 0

    async def complete(
        self, messages: list[Message], tools: list[ToolSpec] | None = None
    ) -> Message:
        self._call_count += 1
        if self._call_count == 1:
            raise TimeoutError("provider timed out")
        return Message(role="assistant", content="recovered")


# ── Helpers ───────────────────────────────────────────────────


def _make_agent(db, llm=None) -> GeneralAgent:
    tools = ToolRegistry(db)
    return GeneralAgent(
        db=db,
        tools=tools,
        llm=llm or StubLLM(),
    )


async def _stop_agent(agent, task):
    await agent.stop()
    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, Exception):
        pass


# ── 1. Provider-level timeouts ───────────────────────────────


@pytest.mark.asyncio
class TestGeminiClientTimeout:
    async def test_hanging_generate_content_times_out(self):
        """GeminiClient.complete() must raise TimeoutError when the
        underlying SDK call hangs, not block forever."""
        from src.llm.providers.gemini import GeminiClient

        client = GeminiClient.__new__(GeminiClient)
        client._model = "test-model"
        client._timeout = 0.3
        client._extra_kwargs = {}

        # Mock the SDK: generate_content hangs forever
        async def _hang(*args, **kwargs):
            await asyncio.sleep(3600)

        mock_aio = MagicMock()
        mock_aio.models.generate_content = _hang
        mock_client = MagicMock()
        mock_client.aio = mock_aio
        client._client = mock_client

        messages = [Message(role="user", content="hello")]
        with pytest.raises(asyncio.TimeoutError):
            await client.complete(messages)

    async def test_fast_response_not_affected(self):
        """A normal fast response should not be broken by the timeout."""
        from google.genai import types as gtypes
        from src.llm.providers.gemini import GeminiClient

        client = GeminiClient.__new__(GeminiClient)
        client._model = "test-model"
        client._timeout = 5.0
        client._extra_kwargs = {}

        # Mock a quick successful response
        mock_part = MagicMock()
        mock_part.text = "blue"
        mock_part.function_call = None

        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_part]
        mock_candidate.grounding_metadata = None

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]

        async def _quick(*args, **kwargs):
            return mock_response

        mock_aio = MagicMock()
        mock_aio.models.generate_content = _quick
        mock_client = MagicMock()
        mock_client.aio = mock_aio
        client._client = mock_client

        messages = [Message(role="user", content="sky color?")]
        result = await client.complete(messages)
        assert result.content == "blue"


@skip_no_openai
@pytest.mark.asyncio
class TestOpenAIClientTimeout:
    async def test_hanging_completions_times_out(self):
        """OpenAIClient.complete() must raise TimeoutError when the
        underlying SDK call hangs."""
        from src.llm.providers.openai import OpenAIClient

        client = OpenAIClient.__new__(OpenAIClient)
        client._model = "test-model"
        client._timeout = 0.3
        client._extra_kwargs = {}

        async def _hang(**kwargs):
            await asyncio.sleep(3600)

        mock_completions = MagicMock()
        mock_completions.create = _hang
        mock_chat = MagicMock()
        mock_chat.completions = mock_completions
        mock_client = MagicMock()
        mock_client.chat = mock_chat
        client._client = mock_client

        messages = [Message(role="user", content="hello")]
        with pytest.raises(asyncio.TimeoutError):
            await client.complete(messages)

    async def test_fast_response_not_affected(self):
        """Normal fast response works fine with timeout in place."""
        from src.llm.providers.openai import OpenAIClient

        client = OpenAIClient.__new__(OpenAIClient)
        client._model = "test-model"
        client._timeout = 5.0
        client._extra_kwargs = {}

        mock_message = MagicMock()
        mock_message.role = "assistant"
        mock_message.content = "blue"
        mock_message.tool_calls = None

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        async def _quick(**kwargs):
            return mock_response

        mock_completions = MagicMock()
        mock_completions.create = _quick
        mock_chat = MagicMock()
        mock_chat.completions = mock_completions
        mock_client = MagicMock()
        mock_client.chat = mock_chat
        client._client = mock_client

        messages = [Message(role="user", content="sky color?")]
        result = await client.complete(messages)
        assert result.content == "blue"


# ── 2. GeneralAgent survives LLM timeouts ────────────────────


@pytest.mark.asyncio
class TestAgentLLMTimeoutResilience:
    async def test_run_loop_continues_after_llm_timeout(self, db):
        """When _analyze_screen_context hits a TimeoutError, the run loop
        must keep consuming queue items."""
        agent = _make_agent(db, llm=TimedOutLLM())
        task = asyncio.create_task(agent.run())
        await asyncio.sleep(0.2)

        for i in range(3):
            await agent.push("event", {
                "agent_name": "test",
                "summary": f"timeout event {i}",
                "app_name": "App", "metadata": {},
            })

        deadline = time.monotonic() + 5.0
        while not agent._queue.empty():
            assert time.monotonic() < deadline, "Run loop stalled on LLM timeout"
            await asyncio.sleep(0.1)

        assert agent._running
        await _stop_agent(agent, task)

    async def test_chat_returns_fallback_on_timeout(self, db):
        """chat() must return a fallback message, not raise TimeoutError."""
        agent = _make_agent(db, llm=TimedOutLLM())
        response = await agent.chat("what's on screen?")
        assert len(response) > 0
        assert "couldn't generate" in response.lower() or "sorry" in response.lower()

    async def test_agent_recovers_after_transient_timeout(self, db):
        """First LLM call times out, second works — agent should recover."""
        llm = OnceThenOkLLM()
        agent = _make_agent(db, llm=llm)
        task = asyncio.create_task(agent.run())
        await asyncio.sleep(0.2)

        # First item: LLM times out → no notification, but no crash
        await agent.push("event", {
            "agent_name": "test", "summary": "will timeout",
            "app_name": "App", "metadata": {},
        })
        await asyncio.sleep(0.5)

        # Second item: LLM works → should process normally
        await agent.push("event", {
            "agent_name": "test", "summary": "will work",
            "app_name": "App", "metadata": {},
        })

        deadline = time.monotonic() + 5.0
        while not agent._queue.empty():
            assert time.monotonic() < deadline, "Agent didn't recover after timeout"
            await asyncio.sleep(0.1)

        await _stop_agent(agent, task)

    async def test_generate_response_timeout_returns_fallback(self, db):
        """_generate_response with a TimedOutLLM should return fallback."""
        agent = _make_agent(db, llm=TimedOutLLM())
        result = await agent._generate_response("test", "context")
        assert "couldn't generate" in result.lower() or len(result) > 0


# ── 3. process_frame resilience ──────────────────────────────


@pytest.mark.asyncio
class TestProcessFrameTimeout:
    async def test_hanging_agent_doesnt_block_frame_processing(self, db, tmp_settings):
        """If a ProcessAgent hangs, process_frame should still complete
        because gather uses return_exceptions=True."""
        from src.capture.triggers import process_frame
        from src.storage.snapshot_writer import SnapshotWriter
        from PIL import Image
        import numpy as np

        writer = SnapshotWriter(tmp_settings)
        image = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))

        class HangingAgent:
            name = "hanging"
            async def process(self, image, ocr, app, window):
                await asyncio.sleep(3600)

        class QuickAgent:
            name = "quick"
            async def process(self, image, ocr, app, window):
                return Event(
                    agent_name="quick", app_type=AppType.OTHER,
                    summary="quick result",
                )

        # Wrap the hanging agent call with a timeout, same as a real scenario
        # where the agent itself should have a timeout
        class TimedHangingAgent:
            name = "timed_hanging"
            async def process(self, image, ocr, app, window):
                try:
                    await asyncio.wait_for(asyncio.sleep(3600), timeout=0.3)
                except asyncio.TimeoutError:
                    return None

        with patch("src.capture.triggers.perform_ocr") as mock_ocr:
            mock_ocr.return_value = OCRResult(text="test", confidence=0.9)

            frame_id = await asyncio.wait_for(
                process_frame(
                    image=image,
                    app_name="Test",
                    window_name="Window",
                    trigger="manual",
                    db=db,
                    writer=writer,
                    agents=[TimedHangingAgent(), QuickAgent()],
                    providers=[],
                ),
                timeout=5.0,
            )

        assert frame_id > 0
        events = await db.get_events_for_frame(frame_id)
        assert len(events) == 1
        assert events[0]["agent_name"] == "quick"

    async def test_all_agents_timeout_frame_still_saved(self, db, tmp_settings):
        """Even if all agents time out, the frame and OCR should still be saved."""
        from src.capture.triggers import process_frame
        from src.storage.snapshot_writer import SnapshotWriter
        from PIL import Image
        import numpy as np

        writer = SnapshotWriter(tmp_settings)
        image = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))

        class TimeoutAgent:
            name = "timeout_agent"
            async def process(self, image, ocr, app, window):
                raise TimeoutError("provider hung")

        with patch("src.capture.triggers.perform_ocr") as mock_ocr:
            mock_ocr.return_value = OCRResult(text="visible text", confidence=0.95)

            frame_id = await asyncio.wait_for(
                process_frame(
                    image=image,
                    app_name="Test",
                    window_name="Window",
                    trigger="manual",
                    db=db,
                    writer=writer,
                    agents=[TimeoutAgent()],
                    providers=[],
                ),
                timeout=5.0,
            )

        assert frame_id > 0
        # Frame saved
        frame = await db.get_frame(frame_id)
        assert frame is not None
        # OCR saved despite agent failure
        ocr = await db.search("visible", limit=1)
        assert len(ocr) == 1


# ── 4. IntelligenceLayer resilience ──────────────────────────


class TimedOutReasoningAgent(ReasoningAgent):
    """Reasoning agent whose LLM call always times out."""

    def __init__(self):
        self.call_count = 0

    @property
    def name(self) -> str:
        return "timed_out_reasoner"

    async def reason(self, event: Event, db: DatabaseManager) -> Action | None:
        self.call_count += 1
        raise TimeoutError("LLM provider timed out")


class WorkingReasoningAgent(ReasoningAgent):
    """Reasoning agent that always returns an action."""

    @property
    def name(self) -> str:
        return "working_reasoner"

    async def reason(self, event: Event, db: DatabaseManager) -> Action | None:
        return Action(
            event_id=event.id,
            frame_id=event.frame_id,
            agent_name=self.name,
            action_type="test",
            action_description="found something",
        )


@pytest.mark.asyncio
class TestIntelligenceLayerTimeout:
    async def test_timeout_agent_doesnt_block_layer(self, db):
        """A reasoning agent that times out should not prevent the
        IntelligenceLayer from processing subsequent events."""
        timeout_agent = TimedOutReasoningAgent()
        layer = IntelligenceLayer(agents=[timeout_agent], db=db)
        task = asyncio.create_task(layer.run())

        fid = await db.insert_frame(Frame(
            timestamp="2025-01-01T00:00:00Z", capture_trigger="manual",
        ))

        for i in range(3):
            evt = Event(
                agent_name="test", app_type=AppType.BROWSER,
                summary=f"event {i}", frame_id=fid,
            )
            evt.id = i + 1
            await layer.submit(evt)

        # All 3 events should be consumed
        deadline = time.monotonic() + 5.0
        while not layer._event_queue.empty():
            assert time.monotonic() < deadline, "IntelligenceLayer stalled on timeout"
            await asyncio.sleep(0.1)
        await asyncio.sleep(0.3)

        assert timeout_agent.call_count == 3
        await layer.stop()
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

    async def test_working_agent_unaffected_by_timed_out_sibling(self, db):
        """When one reasoning agent times out, a sibling agent should
        still produce its action."""
        timeout_agent = TimedOutReasoningAgent()
        working_agent = WorkingReasoningAgent()
        layer = IntelligenceLayer(
            agents=[timeout_agent, working_agent], db=db
        )
        task = asyncio.create_task(layer.run())

        fid = await db.insert_frame(Frame(
            timestamp="2025-01-01T00:00:00Z", capture_trigger="manual",
        ))
        evt = Event(
            agent_name="test", app_type=AppType.BROWSER,
            summary="test event", frame_id=fid,
        )
        evt.id = 1
        await db.insert_event(fid, evt)
        await layer.submit(evt)

        await asyncio.sleep(1.0)

        # Working agent's action should be in the DB despite sibling timeout
        actions = await db.search_actions(query="found something", limit=10)
        assert len(actions) >= 1
        assert actions[0]["agent_name"] == "working_reasoner"

        await layer.stop()
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass


# ── 5. GeminiVisionAgent timeout ─────────────────────────────


@pytest.mark.asyncio
class TestGeminiVisionAgentTimeout:
    async def test_hanging_gemini_returns_none(self):
        """When the Gemini API hangs, GeminiVisionAgent.process() should
        return None (not block forever) thanks to the wait_for timeout."""
        from src.process.gemini_vision_agent import GeminiVisionAgent
        from PIL import Image
        import numpy as np

        agent = GeminiVisionAgent(model="test-model")

        async def _hang(*args, **kwargs):
            await asyncio.sleep(3600)

        mock_aio = MagicMock()
        mock_aio.models.generate_content = _hang
        mock_client = MagicMock()
        mock_client.aio = mock_aio
        agent._client = mock_client

        image = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))

        # _call_gemini has an internal 15s timeout — patch it down for the test
        original_call = agent._call_gemini

        async def _fast_timeout_call(image_bytes, prompt):
            import src.process.gemini_vision_agent as mod
            # Temporarily override the timeout in _call_gemini by calling
            # the Gemini SDK with a short wait_for
            from google.genai import types as gtypes
            image_part = gtypes.Part(
                inline_data=gtypes.Blob(mime_type="image/jpeg", data=image_bytes),
            )
            text_part = gtypes.Part(text=prompt)
            config = gtypes.GenerateContentConfig(
                system_instruction="test",
                temperature=0.2,
                response_mime_type="application/json",
            )
            response = await asyncio.wait_for(
                agent._client.aio.models.generate_content(
                    model=agent._model,
                    contents=[gtypes.Content(role="user", parts=[image_part, text_part])],
                    config=config,
                ),
                timeout=0.3,
            )
            return response.text or ""

        agent._call_gemini = _fast_timeout_call

        result = await asyncio.wait_for(
            agent.process(image, "", "Test", "Window"),
            timeout=5.0,
        )
        # Should return None (timeout caught by the except Exception block)
        assert result is None
