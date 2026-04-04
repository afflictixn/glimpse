"""Tests for BrowserContentAgent — all osascript calls are mocked."""
from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest
from PIL import Image

from src.process.browser_content_agent import (
    BrowserContentAgent,
    _chromium_jxa_execute,
    _chromium_jxa_fallback,
    _safari_applescript_execute,
    _safari_applescript_fallback,
    _EXTRACT_META_ONLY_PAGE_JS,
)
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


# ── JXA script builders ──

class TestJxaScriptBuilders:
    def test_chromium_jxa_execute_contains_app_name(self):
        script = _chromium_jxa_execute("Google Chrome", "location.href")
        assert 'Application("Google Chrome")' in script
        assert ".activeTab.execute(" in script
        assert "location.href" in script

    def test_chromium_jxa_execute_escapes_quotes(self):
        script = _chromium_jxa_execute("Arc", 'JSON.stringify({"a":"b"})')
        assert "Arc" in script
        assert '\\"a\\"' in script

    def test_chromium_jxa_execute_checks_window_count(self):
        script = _chromium_jxa_execute("Google Chrome", "1+1")
        assert "windows.length === 0" in script
        assert "NO_WINDOWS" in script

    def test_chromium_jxa_fallback_returns_url_title(self):
        script = _chromium_jxa_fallback("Google Chrome")
        assert 'Application("Google Chrome")' in script
        assert "tab.url()" in script
        assert "tab.title()" in script
        assert "|||" in script

    def test_safari_applescript_execute_uses_do_javascript(self):
        script = _safari_applescript_execute("location.href")
        assert 'tell application "Safari"' in script
        assert "do JavaScript" in script

    def test_safari_applescript_fallback_uses_current_tab(self):
        script = _safari_applescript_fallback()
        assert "current tab of front window" in script
        assert "|||" in script

    def test_chromium_jxa_escapes_newlines(self):
        """Multiline page JS must not produce literal newlines in the JXA string."""
        script = _chromium_jxa_execute("Google Chrome", _EXTRACT_META_ONLY_PAGE_JS)
        assert "\n" not in script
        assert "\r" not in script
        assert "\\n" in script

    def test_safari_applescript_escapes_newlines(self):
        script = _safari_applescript_execute(_EXTRACT_META_ONLY_PAGE_JS)
        assert "\n" not in script
        assert "\r" not in script


# ── osascript invocation flags ──

class TestOsascriptFlags:
    """Verify that Chromium uses JXA (-l JavaScript) and Safari uses plain AppleScript."""

    @pytest.mark.asyncio
    async def test_chrome_passes_jxa_flag(self, tmp_path):
        agent = BrowserContentAgent(allowlist_path=_make_allowlist(tmp_path))
        js_out = _js_result("https://example.com/p", "Test")
        create_fn, mock_proc = _mock_osascript(js_out)

        captured_calls: list[tuple] = []
        original_create = create_fn

        async def _capture(*args, **kwargs):
            captured_calls.append(args)
            return await original_create(*args, **kwargs)

        with patch("src.process.browser_content_agent.asyncio.create_subprocess_exec", side_effect=_capture):
            await agent.process(_img(), "", "Google Chrome", "Test")

        assert len(captured_calls) >= 1
        first_call_args = captured_calls[0]
        assert first_call_args[0] == "osascript"
        assert first_call_args[1] == "-l"
        assert first_call_args[2] == "JavaScript"

    @pytest.mark.asyncio
    async def test_safari_no_jxa_flag(self, tmp_path):
        agent = BrowserContentAgent(allowlist_path=_make_allowlist(tmp_path))
        js_out = _js_result("https://example.com/s", "Safari Test")
        create_fn, _ = _mock_osascript(js_out)

        captured_calls: list[tuple] = []

        async def _capture(*args, **kwargs):
            captured_calls.append(args)
            result = await create_fn(*args, **kwargs)
            return result

        with patch("src.process.browser_content_agent.asyncio.create_subprocess_exec", side_effect=_capture):
            await agent.process(_img(), "", "Safari", "Safari Test")

        assert len(captured_calls) >= 1
        first_call_args = captured_calls[0]
        assert first_call_args[0] == "osascript"
        assert "-l" not in first_call_args
        assert "JavaScript" not in first_call_args

    @pytest.mark.asyncio
    async def test_arc_passes_jxa_flag(self, tmp_path):
        """Arc is a Chromium browser and should use JXA."""
        agent = BrowserContentAgent(allowlist_path=_make_allowlist(tmp_path))
        js_out = _js_result("https://example.com/a", "Arc Test")
        create_fn, _ = _mock_osascript(js_out)

        captured_calls: list[tuple] = []

        async def _capture(*args, **kwargs):
            captured_calls.append(args)
            return await create_fn(*args, **kwargs)

        with patch("src.process.browser_content_agent.asyncio.create_subprocess_exec", side_effect=_capture):
            await agent.process(_img(), "", "Arc", "Arc Test")

        first_call_args = captured_calls[0]
        assert first_call_args[1:3] == ("-l", "JavaScript")

    @pytest.mark.asyncio
    async def test_chrome_fallback_also_uses_jxa(self, tmp_path):
        """When JS execution fails, the Chromium fallback should also use JXA."""
        agent = BrowserContentAgent(allowlist_path=_make_allowlist(tmp_path))

        captured_calls: list[tuple] = []
        call_count = 0

        async def _create(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            captured_calls.append(args)
            mock = AsyncMock()
            if call_count == 1:
                mock.communicate = AsyncMock(return_value=(b"", b"error"))
                mock.returncode = 1
            else:
                mock.communicate = AsyncMock(
                    return_value=(b"https://example.com|||Fallback", b"")
                )
                mock.returncode = 0
            mock.kill = MagicMock()
            return mock

        with patch("src.process.browser_content_agent.asyncio.create_subprocess_exec", side_effect=_create):
            await agent.process(_img(), "", "Google Chrome", "window")

        for call_args in captured_calls:
            assert call_args[0] == "osascript"
            assert call_args[1] == "-l"
            assert call_args[2] == "JavaScript"


# ── Live integration tests (skipped unless Chrome is running with windows) ──

def _chrome_has_windows() -> bool:
    """Quick check if Chrome has accessible windows via JXA."""
    import subprocess
    try:
        result = subprocess.run(
            ["osascript", "-l", "JavaScript", "-e",
             'Application("Google Chrome").windows.length'],
            capture_output=True, text=True, timeout=3,
        )
        return result.returncode == 0 and result.stdout.strip() not in ("0", "")
    except Exception:
        return False


skip_no_chrome = pytest.mark.skipif(
    not _chrome_has_windows(),
    reason="Google Chrome not running or has no windows",
)


@skip_no_chrome
class TestLiveChromeIntegration:
    """Integration tests that run against a live Chrome instance."""

    @pytest.mark.asyncio
    async def test_meta_extraction_returns_url(self, tmp_path):
        agent = BrowserContentAgent(allowlist_path=_make_allowlist(tmp_path))
        result = await agent.process(_img(), "", "Google Chrome", "live test")
        assert result is not None
        assert result.metadata["url"].startswith("http")
        assert result.metadata["title"]
        assert result.metadata["browser"] == "Google Chrome"

    @pytest.mark.asyncio
    async def test_body_text_extracted(self, tmp_path):
        agent = BrowserContentAgent(allowlist_path=_make_allowlist(tmp_path))
        result = await agent.process(_img(), "", "Google Chrome", "live body test")
        assert result is not None
        # Non-allowlisted pages should still have text (generic body extraction)
        if "text" in result.metadata:
            assert len(result.metadata["text"]) > 0

    @pytest.mark.asyncio
    async def test_dedup_same_page(self, tmp_path):
        agent = BrowserContentAgent(allowlist_path=_make_allowlist(tmp_path))
        r1 = await agent.process(_img(), "", "Google Chrome", "dedup 1")
        r2 = await agent.process(_img(), "", "Google Chrome", "dedup 2")
        assert r1 is not None
        assert r2 is None  # same URL, should be deduped
