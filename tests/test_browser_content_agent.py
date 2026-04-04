"""Tests for BrowserContentAgent — all osascript calls are mocked."""
from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from src.process.browser_content_agent import BrowserContentAgent
from src.storage.models import AppType


def _img() -> Image.Image:
    return Image.new("RGB", (100, 100))


def _make_allowlist(tmp_path: Path, entries: list[dict] | None = None) -> Path:
    if entries is None:
        entries = [
            {"pattern": "amazon.com/", "selector": "#dp-container", "label": "Amazon product"},
            {"pattern": "linkedin.com/in/", "selector": "main", "label": "LinkedIn profile"},
        ]
    path = tmp_path / "allowlist.json"
    path.write_text(json.dumps(entries))
    return path


def _js_result(url: str = "https://example.com", title: str = "Example", **extra) -> str:
    data = {"url": url, "title": title, "meta": {"og:title": title}, **extra}
    return json.dumps(data)


def _mock_osascript(stdout: str, returncode: int = 0):
    """Return a mock for asyncio.create_subprocess_exec that simulates osascript."""
    mock_proc = AsyncMock()
    mock_proc.communicate = AsyncMock(
        return_value=(stdout.encode(), b"")
    )
    mock_proc.returncode = returncode
    mock_proc.kill = MagicMock()

    async def _create(*args, **kwargs):
        return mock_proc

    return _create, mock_proc


# ── Non-browser apps ──

class TestBrowserGating:
    @pytest.mark.asyncio
    async def test_non_browser_returns_none(self, tmp_path):
        agent = BrowserContentAgent(allowlist_path=_make_allowlist(tmp_path))
        result = await agent.process(_img(), "", "Xcode", "main.swift")
        assert result is None

    @pytest.mark.asyncio
    async def test_none_app_returns_none(self, tmp_path):
        agent = BrowserContentAgent(allowlist_path=_make_allowlist(tmp_path))
        result = await agent.process(_img(), "", None, None)
        assert result is None

    @pytest.mark.asyncio
    async def test_terminal_returns_none(self, tmp_path):
        agent = BrowserContentAgent(allowlist_path=_make_allowlist(tmp_path))
        result = await agent.process(_img(), "", "Terminal", "bash")
        assert result is None


# ── Successful extraction ──

class TestChromeExtraction:
    @pytest.mark.asyncio
    async def test_chrome_meta_only(self, tmp_path):
        agent = BrowserContentAgent(allowlist_path=_make_allowlist(tmp_path))
        js_out = _js_result("https://example.com/page", "Example Page")
        create_fn, _ = _mock_osascript(js_out)

        with patch("src.process.browser_content_agent.asyncio.create_subprocess_exec", side_effect=create_fn):
            result = await agent.process(_img(), "", "Google Chrome", "Example Page")

        assert result is not None
        assert result.app_type == AppType.BROWSER
        assert result.agent_name == "browser_content"
        assert result.metadata["url"] == "https://example.com/page"
        assert result.metadata["title"] == "Example Page"
        assert result.metadata["browser"] == "Google Chrome"
        assert "text" not in result.metadata
        assert "Viewing:" in result.summary

    @pytest.mark.asyncio
    async def test_chrome_allowlisted_with_text(self, tmp_path):
        agent = BrowserContentAgent(allowlist_path=_make_allowlist(tmp_path))

        meta_out = _js_result("https://www.amazon.com/product/123", "Widget")
        full_out = _js_result(
            "https://www.amazon.com/product/123", "Widget",
            text="Great product. 4.5 stars. Buy now.",
        )

        call_count = 0

        async def _create(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            mock = AsyncMock()
            if call_count == 1:
                mock.communicate = AsyncMock(return_value=(meta_out.encode(), b""))
            else:
                mock.communicate = AsyncMock(return_value=(full_out.encode(), b""))
            mock.returncode = 0
            mock.kill = MagicMock()
            return mock

        with patch("src.process.browser_content_agent.asyncio.create_subprocess_exec", side_effect=_create):
            result = await agent.process(_img(), "", "Google Chrome", "Widget")

        assert result is not None
        assert result.metadata["text"] == "Great product. 4.5 stars. Buy now."
        assert result.metadata["allowlist_label"] == "Amazon product"
        assert "[Amazon product]" in result.summary


class TestSafariExtraction:
    @pytest.mark.asyncio
    async def test_safari_meta_only(self, tmp_path):
        agent = BrowserContentAgent(allowlist_path=_make_allowlist(tmp_path))
        js_out = _js_result("https://example.com", "Safari Test")
        create_fn, _ = _mock_osascript(js_out)

        with patch("src.process.browser_content_agent.asyncio.create_subprocess_exec", side_effect=create_fn):
            result = await agent.process(_img(), "", "Safari", "Safari Test")

        assert result is not None
        assert result.app_type == AppType.BROWSER
        assert result.metadata["browser"] == "Safari"


# ── Fallback ──

class TestFallback:
    @pytest.mark.asyncio
    async def test_js_failure_falls_back_to_url_title(self, tmp_path):
        agent = BrowserContentAgent(allowlist_path=_make_allowlist(tmp_path))

        call_count = 0

        async def _create(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            mock = AsyncMock()
            if call_count == 1:
                mock.communicate = AsyncMock(return_value=(b"", b"error"))
                mock.returncode = 1
            else:
                mock.communicate = AsyncMock(
                    return_value=(b"https://example.com|||Fallback Title", b"")
                )
                mock.returncode = 0
            mock.kill = MagicMock()
            return mock

        with patch("src.process.browser_content_agent.asyncio.create_subprocess_exec", side_effect=_create):
            result = await agent.process(_img(), "", "Google Chrome", "window")

        assert result is not None
        assert result.metadata["url"] == "https://example.com"
        assert result.metadata["title"] == "Fallback Title"
        assert "text" not in result.metadata


# ── Timeout ──

class TestTimeout:
    @pytest.mark.asyncio
    async def test_timeout_returns_none(self, tmp_path):
        agent = BrowserContentAgent(allowlist_path=_make_allowlist(tmp_path))

        async def _create(*args, **kwargs):
            mock = AsyncMock()

            async def _slow_communicate():
                await asyncio.sleep(10)
                return (b"", b"")

            mock.communicate = _slow_communicate
            mock.returncode = 1
            mock.kill = MagicMock()
            return mock

        with patch("src.process.browser_content_agent.asyncio.create_subprocess_exec", side_effect=_create):
            with patch("src.process.browser_content_agent.OSASCRIPT_TIMEOUT_S", 0.1):
                result = await agent.process(_img(), "", "Google Chrome", "window")

        assert result is None


# ── URL dedup ──

class TestUrlDedup:
    @pytest.mark.asyncio
    async def test_same_url_returns_none_on_second_call(self, tmp_path):
        agent = BrowserContentAgent(allowlist_path=_make_allowlist(tmp_path))
        js_out = _js_result("https://example.com/page1", "Page 1")
        create_fn, _ = _mock_osascript(js_out)

        with patch("src.process.browser_content_agent.asyncio.create_subprocess_exec", side_effect=create_fn):
            r1 = await agent.process(_img(), "", "Google Chrome", "Page 1")
            r2 = await agent.process(_img(), "", "Google Chrome", "Page 1")

        assert r1 is not None
        assert r2 is None

    @pytest.mark.asyncio
    async def test_different_url_produces_new_event(self, tmp_path):
        agent = BrowserContentAgent(allowlist_path=_make_allowlist(tmp_path))

        call_count = 0

        async def _create(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            mock = AsyncMock()
            # Each process() call makes 2 osascript calls (meta + body).
            # Calls 1-2 → page1, calls 3-4 → page2.
            if call_count <= 2:
                out = _js_result("https://example.com/page1", "Page 1")
            else:
                out = _js_result("https://example.com/page2", "Page 2")
            mock.communicate = AsyncMock(return_value=(out.encode(), b""))
            mock.returncode = 0
            mock.kill = MagicMock()
            return mock

        with patch("src.process.browser_content_agent.asyncio.create_subprocess_exec", side_effect=_create):
            r1 = await agent.process(_img(), "", "Google Chrome", "Page 1")
            r2 = await agent.process(_img(), "", "Google Chrome", "Page 2")

        assert r1 is not None
        assert r2 is not None
        assert r1.metadata["url"] != r2.metadata["url"]


# ── Allowlist ──

class TestAllowlist:
    @pytest.mark.asyncio
    async def test_non_allowlisted_gets_generic_body_text(self, tmp_path):
        """Non-allowlisted pages now extract body text with 4000 char limit."""
        agent = BrowserContentAgent(allowlist_path=_make_allowlist(tmp_path))

        meta_out = _js_result("https://unknown-site.com/page", "Unknown")
        body_out = _js_result(
            "https://unknown-site.com/page", "Unknown",
            text="Hello world, this is page content.",
        )

        call_count = 0

        async def _create(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            mock = AsyncMock()
            out = meta_out if call_count == 1 else body_out
            mock.communicate = AsyncMock(return_value=(out.encode(), b""))
            mock.returncode = 0
            mock.kill = MagicMock()
            return mock

        with patch("src.process.browser_content_agent.asyncio.create_subprocess_exec", side_effect=_create):
            result = await agent.process(_img(), "", "Google Chrome", "Unknown")

        assert result is not None
        assert result.metadata["text"] == "Hello world, this is page content."
        assert result.metadata["allowlist_label"] == ""

    @pytest.mark.asyncio
    async def test_bad_selector_produces_event_with_empty_text(self, tmp_path):
        entries = [{"pattern": "example.com", "selector": "#nonexistent", "label": "Test"}]
        agent = BrowserContentAgent(allowlist_path=_make_allowlist(tmp_path, entries))

        meta_out = _js_result("https://example.com/page", "Test Page")
        full_out = _js_result("https://example.com/page", "Test Page", text="")

        call_count = 0

        async def _create(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            mock = AsyncMock()
            out = meta_out if call_count == 1 else full_out
            mock.communicate = AsyncMock(return_value=(out.encode(), b""))
            mock.returncode = 0
            mock.kill = MagicMock()
            return mock

        with patch("src.process.browser_content_agent.asyncio.create_subprocess_exec", side_effect=_create):
            result = await agent.process(_img(), "", "Google Chrome", "Test Page")

        assert result is not None
        assert result.metadata.get("allowlist_label") == "Test"

    @pytest.mark.asyncio
    async def test_missing_allowlist_file_uses_empty_list(self, tmp_path):
        agent = BrowserContentAgent(allowlist_path=tmp_path / "does_not_exist.json")
        assert agent._allowlist == []


# ── Text truncation ──

class TestTruncation:
    @pytest.mark.asyncio
    async def test_text_truncated_to_max_length(self, tmp_path):
        entries = [{"pattern": "example.com", "selector": "body", "label": "Test"}]
        agent = BrowserContentAgent(allowlist_path=_make_allowlist(tmp_path, entries))

        long_text = "x" * 10000
        meta_out = _js_result("https://example.com", "Test")
        full_out = _js_result("https://example.com", "Test", text=long_text)

        call_count = 0

        async def _create(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            mock = AsyncMock()
            out = meta_out if call_count == 1 else full_out
            mock.communicate = AsyncMock(return_value=(out.encode(), b""))
            mock.returncode = 0
            mock.kill = MagicMock()
            return mock

        with patch("src.process.browser_content_agent.asyncio.create_subprocess_exec", side_effect=_create):
            result = await agent.process(_img(), "", "Google Chrome", "Test")

        assert result is not None
        assert len(result.metadata["text"]) <= 8000

    @pytest.mark.asyncio
    async def test_non_allowlisted_text_truncated_to_generic_limit(self, tmp_path):
        """Non-allowlisted pages truncate body text to 4000 chars."""
        agent = BrowserContentAgent(allowlist_path=_make_allowlist(tmp_path))

        long_text = "y" * 6000
        meta_out = _js_result("https://unknown.com", "Unknown")
        full_out = _js_result("https://unknown.com", "Unknown", text=long_text)

        call_count = 0

        async def _create(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            mock = AsyncMock()
            out = meta_out if call_count == 1 else full_out
            mock.communicate = AsyncMock(return_value=(out.encode(), b""))
            mock.returncode = 0
            mock.kill = MagicMock()
            return mock

        with patch("src.process.browser_content_agent.asyncio.create_subprocess_exec", side_effect=_create):
            result = await agent.process(_img(), "", "Google Chrome", "Unknown")

        assert result is not None
        assert len(result.metadata["text"]) <= 4000
