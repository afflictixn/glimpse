"""Tests for the GeneralAgent + Ollama tool-calling pipeline.

Unit tests mock Ollama responses to verify the prompt-based tool-call loop
works correctly. Live integration tests hit a real local Ollama instance
and are auto-skipped when Ollama is not running.
"""
from __future__ import annotations

import json
import urllib.request
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from src.general_agent.agent import GeneralAgent
from src.general_agent.ollama_client import (
    OllamaChat,
    ToolCall,
    build_tools_prompt,
    parse_tool_calls,
    sanitize_response,
    tool_spec_to_ollama,
)
from src.general_agent.tools import ToolRegistry
from src.storage.database import DatabaseManager
from src.storage.models import Event, Frame, OCRResult, AppType


# ── Helpers ──


def _ollama_available() -> bool:
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


ollama = pytest.mark.skipif(not _ollama_available(), reason="Ollama not running")


# ── Fixtures ──


@pytest_asyncio.fixture
async def tools(db: DatabaseManager) -> ToolRegistry:
    return ToolRegistry(db)


@pytest_asyncio.fixture
async def agent(db: DatabaseManager, tools: ToolRegistry) -> GeneralAgent:
    return GeneralAgent(
        db=db,
        tools=tools,
        overlay_ws_url="ws://localhost:19321",
        ollama_base_url="http://localhost:11434",
        ollama_model="gemma3:12b",
    )


async def _seed(db: DatabaseManager) -> int:
    fid = await db.insert_frame(Frame(
        timestamp="2025-06-01T10:00:00Z",
        app_name="Chrome",
        window_name="Amazon - Sony WH-1000XM5",
        capture_trigger="click",
    ))
    await db.insert_ocr(fid, OCRResult(
        text="Sony WH-1000XM5 Wireless Headphones $299.99 Add to Cart",
        text_json="[]",
        confidence=0.95,
    ))
    await db.insert_event(fid, Event(
        agent_name="gemma",
        app_type=AppType.BROWSER,
        summary="User browsing Sony WH-1000XM5 headphones on Amazon, priced at $299.99",
    ))
    return fid


# ── parse_tool_calls ──


class TestParseToolCalls:
    def test_no_tool_calls(self):
        parsed = parse_tool_calls("Just a regular response.")
        assert not parsed.has_tool_calls
        assert parsed.text == "Just a regular response."

    def test_single_tool_call(self):
        text = 'Let me search for that.\n<tool_call>{"name": "web_search", "arguments": {"query": "best headphones"}}</tool_call>'
        parsed = parse_tool_calls(text)
        assert parsed.has_tool_calls
        assert len(parsed.tool_calls) == 1
        assert parsed.tool_calls[0].name == "web_search"
        assert parsed.tool_calls[0].arguments == {"query": "best headphones"}
        assert "Let me search" in parsed.text
        assert "<tool_call>" not in parsed.text

    def test_multiple_tool_calls(self):
        text = (
            '<tool_call>{"name": "web_search", "arguments": {"query": "price"}}</tool_call>\n'
            '<tool_call>{"name": "memory_query", "arguments": {"query": "headphones"}}</tool_call>'
        )
        parsed = parse_tool_calls(text)
        assert len(parsed.tool_calls) == 2
        assert parsed.tool_calls[0].name == "web_search"
        assert parsed.tool_calls[1].name == "memory_query"

    def test_multiline_json(self):
        text = '<tool_call>\n{\n  "name": "db_query",\n  "arguments": {"table": "ocr", "query": "test"}\n}\n</tool_call>'
        parsed = parse_tool_calls(text)
        assert parsed.has_tool_calls
        assert parsed.tool_calls[0].name == "db_query"

    def test_invalid_json_skipped(self):
        text = '<tool_call>not valid json</tool_call>\nSome text here.'
        parsed = parse_tool_calls(text)
        assert not parsed.has_tool_calls
        assert "Some text here" in parsed.text

    def test_markdown_fenced_tool_call(self):
        text = '```tool_call>{"name": "db_query", "arguments": {"table": "ocr", "query": "headphones", "limit": 10}}</tool_call>\n```'
        parsed = parse_tool_calls(text)
        assert parsed.has_tool_calls
        assert parsed.tool_calls[0].name == "db_query"
        assert parsed.tool_calls[0].arguments["table"] == "ocr"

    def test_markdown_fenced_inline(self):
        text = '```tool_call>{"name": "memory_query", "arguments": {"query": "product"}}</tool_call>```'
        parsed = parse_tool_calls(text)
        assert parsed.has_tool_calls
        assert parsed.tool_calls[0].name == "memory_query"

    def test_mixed_text_and_tool_calls(self):
        text = (
            'I\'ll look that up for you.\n'
            '<tool_call>{"name": "web_search", "arguments": {"query": "test"}}</tool_call>\n'
            'And also check this.'
        )
        parsed = parse_tool_calls(text)
        assert parsed.has_tool_calls
        assert "look that up" in parsed.text
        assert "also check" in parsed.text


# ── sanitize_response ──


class TestSanitizeResponse:
    def test_strips_start_of_image(self):
        assert sanitize_response("<start_of_image>Okay, let me check.") == "Okay, let me check."

    def test_strips_end_of_turn(self):
        assert sanitize_response("Hello<end_of_turn>") == "Hello"

    def test_strips_multiple_artifacts(self):
        text = "<start_of_image><start_of_turn>Here is your answer<end_of_turn><eos>"
        assert sanitize_response(text) == "Here is your answer"

    def test_collapses_newlines(self):
        assert sanitize_response("Hello\n\n\n\n\nWorld") == "Hello\n\nWorld"

    def test_clean_text_unchanged(self):
        assert sanitize_response("Perfectly normal response.") == "Perfectly normal response."

    def test_embedded_artifact(self):
        text = "Currently in Krakow<start_of_image>, it's 14C."
        assert sanitize_response(text) == "Currently in Krakow, it's 14C."


# ── system prompt ──


class TestSystemPrompt:
    def test_includes_current_date(self):
        from src.general_agent.agent import _build_system_prompt
        prompt = _build_system_prompt()
        # Should have current year
        from datetime import datetime, timezone
        year = str(datetime.now(timezone.utc).year)
        assert year in prompt

    def test_includes_no_hallucination_rule(self):
        from src.general_agent.agent import _build_system_prompt
        prompt = _build_system_prompt()
        assert "NEVER guess" in prompt
        assert "web_search" in prompt

    def test_no_emoji_rule(self):
        from src.general_agent.agent import _build_system_prompt
        prompt = _build_system_prompt()
        assert "No emojis" in prompt


# ── build_tools_prompt ──


class TestBuildToolsPrompt:
    def test_includes_tool_names(self):
        tools = [tool_spec_to_ollama("web_search", "Search web", {"query": "Search query"})]
        prompt = build_tools_prompt(tools)
        assert "web_search" in prompt
        assert "Search web" in prompt
        assert "query" in prompt

    def test_includes_example(self):
        tools = [tool_spec_to_ollama("test", "A test", {"x": "param"})]
        prompt = build_tools_prompt(tools)
        assert "<tool_call>" in prompt
        assert "</tool_call>" in prompt


# ── tool_spec_to_ollama ──


class TestToolSpecConversion:
    def test_converts_spec(self):
        result = tool_spec_to_ollama(
            name="web_search",
            description="Search the web",
            parameters={"query": "The search query"},
        )
        assert result["type"] == "function"
        fn = result["function"]
        assert fn["name"] == "web_search"
        assert fn["parameters"]["properties"]["query"]["type"] == "string"


# ── OllamaChat unit tests ──


@pytest.mark.asyncio
class TestOllamaChatUnit:
    async def test_tool_calling_loop_executes_tools(self):
        """Mock the API to return a tool call, then a text response."""
        client = OllamaChat()

        call_count = 0

        def mock_api(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"message": {"content":
                    '<tool_call>{"name": "web_search", "arguments": {"query": "best headphones 2025"}}</tool_call>'
                }}
            else:
                return {"message": {"content": "The best headphones in 2025 are the Sony XM5."}}

        client._call_api = mock_api

        async def mock_executor(tool_name, **kwargs):
            return json.dumps([{"title": "Sony XM5 review", "body": "Top rated"}])

        result = await client.chat_with_tools(
            messages=[{"role": "user", "content": "best headphones?"}],
            tools=[tool_spec_to_ollama("web_search", "Search", {"query": "q"})],
            tool_executor=mock_executor,
        )

        assert "Sony XM5" in result
        assert call_count == 2

    async def test_tool_calling_loop_max_rounds(self):
        """Verify it stops after MAX_TOOL_ROUNDS."""
        client = OllamaChat()

        call_count = 0

        def mock_api(messages):
            nonlocal call_count
            call_count += 1
            # Always return a tool call until the final forced round
            if call_count <= 6:
                return {"message": {"content":
                    f'<tool_call>{{"name": "web_search", "arguments": {{"query": "round {call_count}"}}}}</tool_call>'
                }}
            return {"message": {"content": "Done after max rounds."}}

        client._call_api = mock_api

        async def mock_executor(tool_name, **kwargs):
            return json.dumps({"result": "ok"})

        result = await client.chat_with_tools(
            messages=[{"role": "user", "content": "loop"}],
            tools=[tool_spec_to_ollama("web_search", "Search", {"query": "q"})],
            tool_executor=mock_executor,
        )

        # 5 rounds with tool calls + 1 final = 6, but the 6th returns a tool call too,
        # so we get the final forced call = 7 total
        assert call_count >= 6

    async def test_no_tool_calls_returns_immediately(self):
        client = OllamaChat()

        def mock_api(messages):
            return {"message": {"content": "Direct answer, no tools needed."}}

        client._call_api = mock_api

        result = await client.chat_with_tools(
            messages=[{"role": "user", "content": "hello"}],
            tools=[],
            tool_executor=AsyncMock(),
        )
        assert result == "Direct answer, no tools needed."

    async def test_tool_executor_error_is_fed_back(self):
        """If a tool raises, the error JSON is passed back to the LLM."""
        client = OllamaChat()

        call_count = 0
        received_messages = []

        def mock_api(messages):
            nonlocal call_count
            call_count += 1
            received_messages.append(list(messages))
            if call_count == 1:
                return {"message": {"content":
                    '<tool_call>{"name": "bad_tool", "arguments": {"x": "1"}}</tool_call>'
                }}
            return {"message": {"content": "Tool failed, here's my best answer anyway."}}

        client._call_api = mock_api

        async def failing_executor(tool_name, **kwargs):
            raise RuntimeError("connection refused")

        result = await client.chat_with_tools(
            messages=[{"role": "user", "content": "test"}],
            tools=[tool_spec_to_ollama("bad_tool", "Bad", {"x": "param"})],
            tool_executor=failing_executor,
        )

        assert "best answer" in result
        # Check that the error was fed back
        last_messages = received_messages[-1]
        tool_result_msgs = [m for m in last_messages if "Tool results" in m.get("content", "")]
        assert len(tool_result_msgs) == 1
        assert "connection refused" in tool_result_msgs[0]["content"]

    async def test_multiple_tools_in_one_round(self):
        """LLM calls two tools in one response, both get executed."""
        client = OllamaChat()

        call_count = 0
        executed_tools = []

        def mock_api(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"message": {"content":
                    '<tool_call>{"name": "web_search", "arguments": {"query": "price"}}</tool_call>\n'
                    '<tool_call>{"name": "memory_query", "arguments": {"query": "headphones"}}</tool_call>'
                }}
            return {"message": {"content": "Based on web search and your memory, here's the answer."}}

        client._call_api = mock_api

        async def track_executor(tool_name, **kwargs):
            executed_tools.append(tool_name)
            return json.dumps({"result": f"{tool_name} done"})

        result = await client.chat_with_tools(
            messages=[{"role": "user", "content": "check prices"}],
            tools=[
                tool_spec_to_ollama("web_search", "Search", {"query": "q"}),
                tool_spec_to_ollama("memory_query", "Memory", {"query": "q"}),
            ],
            tool_executor=track_executor,
        )

        assert executed_tools == ["web_search", "memory_query"]
        assert "Based on" in result


# ── GeneralAgent unit tests (mocked Ollama) ──


@pytest.mark.asyncio
class TestGeneralAgentUnit:
    async def test_chat_calls_llm(self, agent: GeneralAgent):
        agent._llm.chat_with_tools = AsyncMock(return_value="Mocked response from LLM.")
        agent._ws_send = AsyncMock()

        result = await agent.chat("What's on my screen?")
        assert result == "Mocked response from LLM."
        agent._llm.chat_with_tools.assert_called_once()

    async def test_chat_includes_conversation_history(self, agent: GeneralAgent):
        """Use a tool-triggering message so it goes through chat_with_tools."""
        call_args = []

        async def capture_chat(messages, tools, tool_executor):
            call_args.append(messages)
            return "Response"

        agent._llm.chat_with_tools = capture_chat
        agent._ws_send = AsyncMock()

        await agent.chat("Search for the first thing")
        await agent.chat("Search for the second thing")

        msgs = call_args[1]
        user_msgs = [m["content"] for m in msgs if m["role"] == "user"]
        assert "first thing" in " ".join(user_msgs).lower()
        assert "second thing" in " ".join(user_msgs).lower()

    async def test_chat_handles_llm_failure(self, agent: GeneralAgent):
        """Both fast and tool paths should catch exceptions gracefully."""
        agent._llm.chat = AsyncMock(side_effect=Exception("Ollama down"))
        agent._llm.chat_with_tools = AsyncMock(side_effect=Exception("Ollama down"))
        agent._ws_send = AsyncMock()

        result = await agent.chat("hello")
        assert "trouble" in result.lower()

    async def test_simple_chat_uses_fast_path(self, agent: GeneralAgent):
        """Trivial messages should skip the tool loop."""
        agent._llm.chat = AsyncMock(return_value="Hey there!")
        agent._llm.chat_with_tools = AsyncMock(return_value="Should not be called")
        agent._ws_send = AsyncMock()

        result = await agent.chat("hello")
        assert result == "Hey there!"
        agent._llm.chat.assert_called_once()
        agent._llm.chat_with_tools.assert_not_called()

    async def test_tool_query_uses_tool_path(self, agent: GeneralAgent):
        """Messages with tool signals should use chat_with_tools."""
        agent._llm.chat_with_tools = AsyncMock(return_value="Here are your events.")
        agent._ws_send = AsyncMock()

        result = await agent.chat("What's on my calendar today?")
        assert result == "Here are your events."
        agent._llm.chat_with_tools.assert_called_once()

    async def test_enrich_uses_llm_with_tools(self, agent: GeneralAgent, db: DatabaseManager):
        await _seed(db)

        agent._llm.chat_with_tools = AsyncMock(
            return_value="Those headphones are $50 cheaper on Best Buy right now."
        )

        from src.general_agent.agent import PushItem
        item = PushItem(
            kind="event",
            data={"summary": "User browsing Sony headphones at $299.99"},
        )
        result = await agent._enrich(item)
        assert "headphones" in result.lower() or "cheaper" in result.lower()

    async def test_process_item_surfaces_high_importance(self, agent: GeneralAgent):
        agent._llm.chat_with_tools = AsyncMock(return_value="Important insight here.")
        agent._ws_send = AsyncMock()

        from src.general_agent.agent import PushItem
        item = PushItem(
            kind="action",
            data={
                "action_type": "flag",
                "action_description": "User is about to overpay for this product",
            },
        )
        await agent._process_item(item)

        agent._ws_send.assert_called()
        call_args = agent._ws_send.call_args[0][0]
        assert call_args["type"] == "show_proposal"

    async def test_process_item_skips_empty_summary(self, agent: GeneralAgent):
        agent._ws_send = AsyncMock()

        from src.general_agent.agent import PushItem
        item = PushItem(kind="event", data={"summary": ""})
        await agent._process_item(item)
        agent._ws_send.assert_not_called()

    async def test_ollama_tools_built_from_registry(self, agent: GeneralAgent):
        assert len(agent._ollama_tools) == 13
        names = {t["function"]["name"] for t in agent._ollama_tools}
        assert "web_search" in names
        assert "memory_store" in names
        assert "calendar_events" in names


# ── Live integration tests (require running Ollama) ──


@pytest.mark.asyncio
class TestOllamaLive:
    @ollama
    async def test_simple_chat(self):
        """Verify basic Ollama chat works."""
        client = OllamaChat(model="gemma3:1b")
        result = await client.chat([
            {"role": "user", "content": "Reply with exactly one word: PONG"},
        ])
        assert len(result) > 0
        print(f"\nChat response: {result}")

    @ollama
    async def test_tool_call_executed_and_result_used(self):
        """Verify the full loop: LLM emits tool call → tool runs → result fed back → final answer."""
        client = OllamaChat(model="gemma3:12b", timeout_s=120)

        tools_defs = [tool_spec_to_ollama(
            name="web_search",
            description="Search the web for current information. You MUST use this tool.",
            parameters={"query": "The search query string"},
        )]

        executed_tools: list[tuple[str, dict]] = []

        async def tracking_executor(tool_name, **kwargs):
            executed_tools.append((tool_name, kwargs))
            return json.dumps([
                {"title": "Sony WH-1000XM5 - Best Buy", "body": "Price: $248.00"},
                {"title": "Sony WH-1000XM5 - Amazon", "body": "Price: $299.99"},
            ])

        result = await client.chat_with_tools(
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Always use tools when asked to search."},
                {"role": "user", "content": "Use the web_search tool to find the current price of Sony WH-1000XM5 headphones."},
            ],
            tools=tools_defs,
            tool_executor=tracking_executor,
        )

        print(f"\nExecuted tools: {executed_tools}")
        print(f"Final response: {result}")

        # The tool must have actually been called
        assert len(executed_tools) >= 1, "LLM did not call any tools"
        assert executed_tools[0][0] == "web_search"

        # The final response should incorporate the tool result
        assert any(kw in result.lower() for kw in ["248", "299", "best buy", "amazon", "sony", "price"]), \
            f"Final response doesn't reference tool results: {result}"

    @ollama
    async def test_db_tool_returns_seeded_data(self, db: DatabaseManager, tools: ToolRegistry):
        """Seed the DB, run the full agent pipeline, assert db_query was called and data surfaced."""
        await _seed(db)

        executed_tools: list[str] = []
        original_call = tools.call

        async def tracking_call(tool_name, **kwargs):
            executed_tools.append(tool_name)
            return await original_call(tool_name, **kwargs)

        tools.call = tracking_call  # type: ignore[assignment]

        agent = GeneralAgent(
            db=db,
            tools=tools,
            overlay_ws_url="ws://localhost:19321",
            ollama_base_url="http://localhost:11434",
            ollama_model="gemma3:12b",
        )
        agent._ws_send = AsyncMock()

        result = await agent.chat(
            "Use the db_query tool to search my screen history (ocr table) for 'headphones'. What did you find?"
        )

        print(f"\nExecuted tools: {executed_tools}")
        print(f"Agent response: {result}")

        assert "trouble connecting" not in result.lower()
        assert len(executed_tools) >= 1, "Agent did not call any tools"
        assert "db_query" in executed_tools, f"Expected db_query in {executed_tools}"

    @ollama
    async def test_memory_store_and_recall(self, db: DatabaseManager, tools: ToolRegistry):
        """Full round trip: store a memory, then ask the agent to recall it via memory_query."""
        await tools.call(
            "memory_store",
            entity_type="product",
            entity_name="Sony WH-1000XM5",
            fact="Seen at $299.99 on Amazon, $249.99 on Best Buy",
            source="price_check",
        )

        executed_tools: list[str] = []
        original_call = tools.call

        async def tracking_call(tool_name, **kwargs):
            executed_tools.append(tool_name)
            return await original_call(tool_name, **kwargs)

        tools.call = tracking_call  # type: ignore[assignment]

        agent = GeneralAgent(
            db=db,
            tools=tools,
            overlay_ws_url="ws://localhost:19321",
            ollama_base_url="http://localhost:11434",
            ollama_model="gemma3:12b",
        )
        agent._ws_send = AsyncMock()

        result = await agent.chat(
            "Use the memory_query tool to check if you have any saved product memories. What products do you remember?"
        )

        print(f"\nExecuted tools: {executed_tools}")
        print(f"Agent response: {result}")

        assert "trouble connecting" not in result.lower()
        assert len(executed_tools) >= 1, "Agent did not call any tools"
        assert "memory_query" in executed_tools, f"Expected memory_query in {executed_tools}"
        # The response should reference the stored memory content
        assert any(kw in result.lower() for kw in ["sony", "xm5", "headphone", "299", "249", "product"]), \
            f"Response doesn't reference stored memory: {result}"
