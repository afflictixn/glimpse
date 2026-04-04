"""ProcessAgent that extracts structured content from the active browser tab via AppleScript."""
from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from PIL import Image

from src.process.process_agent import ProcessAgent
from src.storage.models import AppType, Event

logger = logging.getLogger(__name__)

MAX_TEXT_LENGTH = 8000
GENERIC_TEXT_LENGTH = 4000
OSASCRIPT_TIMEOUT_S = 3

CHROMIUM_BROWSERS = frozenset({
    "Google Chrome",
    "Arc",
    "Brave Browser",
    "Microsoft Edge",
    "Chromium",
    "Vivaldi",
    "Opera",
})

SAFARI_BROWSERS = frozenset({"Safari"})

ALL_BROWSERS = CHROMIUM_BROWSERS | SAFARI_BROWSERS

_EXTRACT_WITH_SELECTOR_JS = """(() => {{
    const meta = {{}};
    document.querySelectorAll('meta[property], meta[name]').forEach(m => {{
        const key = m.getAttribute('property') || m.getAttribute('name');
        if (key) meta[key] = m.content;
    }});
    const root = document.querySelector('{selector}');
    return JSON.stringify({{
        url: location.href,
        title: document.title,
        meta: meta,
        text: root ? root.innerText.substring(0, {max_len}) : ''
    }});
}})()"""

_EXTRACT_META_ONLY_JS = """(() => {
    const meta = {};
    document.querySelectorAll('meta[property], meta[name]').forEach(m => {
        const key = m.getAttribute('property') || m.getAttribute('name');
        if (key) meta[key] = m.content;
    });
    return JSON.stringify({
        url: location.href,
        title: document.title,
        meta: meta
    });
})()"""


def _chromium_js_script(app: str, js: str) -> str:
    escaped = js.replace("\\", "\\\\").replace('"', '\\"')
    return (
        f'tell application "{app}" to execute front window\'s '
        f'active tab javascript "{escaped}"'
    )


def _safari_js_script(js: str) -> str:
    escaped = js.replace("\\", "\\\\").replace('"', '\\"')
    return (
        f'tell application "Safari" to do JavaScript '
        f'"{escaped}" in current tab of front window'
    )


def _chromium_fallback_script(app: str) -> str:
    return (
        f'tell application "{app}"\n'
        f"  set tabUrl to URL of active tab of front window\n"
        f"  set tabTitle to title of active tab of front window\n"
        f'  return tabUrl & "|||" & tabTitle\n'
        f"end tell"
    )


def _safari_fallback_script() -> str:
    return (
        'tell application "Safari"\n'
        "  set tabUrl to URL of current tab of front window\n"
        "  set tabTitle to name of current tab of front window\n"
        '  return tabUrl & "|||" & tabTitle\n'
        "end tell"
    )


class BrowserContentAgent(ProcessAgent):
    """Extracts structured page content from the user's active browser tab."""

    def __init__(self, allowlist_path: str | Path | None = None) -> None:
        if allowlist_path is None:
            allowlist_path = Path(__file__).resolve().parents[2] / "config" / "browser_allowlist.json"
        self._allowlist_path = Path(allowlist_path)
        self._allowlist: list[dict[str, str]] = self._load_allowlist()
        self._last_url: str | None = None

    @property
    def name(self) -> str:
        return "browser_content"

    def _load_allowlist(self) -> list[dict[str, str]]:
        if not self._allowlist_path.exists():
            logger.warning("Browser allowlist not found at %s", self._allowlist_path)
            return []
        try:
            with open(self._allowlist_path) as f:
                data = json.load(f)
            logger.info("Loaded %d browser allowlist entries from %s", len(data), self._allowlist_path)
            return data
        except Exception:
            logger.error("Failed to load browser allowlist", exc_info=True)
            return []

    def _match_allowlist(self, url: str) -> dict[str, str] | None:
        for entry in self._allowlist:
            if entry.get("pattern") and entry["pattern"] in url:
                return entry
        return None

    async def process(
        self,
        image: Image.Image,
        ocr_text: str,
        app_name: str | None,
        window_name: str | None,
    ) -> Event | None:
        if not app_name or app_name not in ALL_BROWSERS:
            return None

        t_start = time.monotonic()
        is_chromium = app_name in CHROMIUM_BROWSERS

        match = None
        if is_chromium:
            js = _EXTRACT_META_ONLY_JS
        else:
            js = _EXTRACT_META_ONLY_JS

        # First, try a quick URL-only check for dedup before heavy extraction.
        # We use the meta-only JS which is lightweight and always gives us the URL.
        raw = await self._run_applescript(
            _chromium_js_script(app_name, js) if is_chromium else _safari_js_script(js),
        )
        t_meta = time.monotonic()

        if raw is None:
            logger.debug("DOM meta extraction failed for %s, trying fallback", app_name)
            return await self._fallback_extraction(app_name, is_chromium)

        parsed = self._parse_js_result(raw)
        if parsed is None:
            return await self._fallback_extraction(app_name, is_chromium)

        url = parsed.get("url", "")

        # URL-based dedup
        if url and url == self._last_url:
            logger.debug("DOM dedup: same URL, skipping (%s)", url[:80])
            return None
        self._last_url = url

        # Check allowlist for targeted extraction; fall back to generic body text
        match = self._match_allowlist(url)
        selector = match.get("selector", "body") if match else "body"
        max_len = MAX_TEXT_LENGTH if match else GENERIC_TEXT_LENGTH

        full_js = _EXTRACT_WITH_SELECTOR_JS.format(
            selector=selector.replace("'", "\\'"),
            max_len=max_len,
        )
        full_raw = await self._run_applescript(
            _chromium_js_script(app_name, full_js)
            if is_chromium
            else _safari_js_script(full_js),
        )
        t_body = time.monotonic()

        if full_raw is not None:
            full_parsed = self._parse_js_result(full_raw)
            if full_parsed is not None:
                parsed = full_parsed

        text_len = len(parsed.get("text", ""))
        label = match.get("label", "") if match else ""
        logger.debug(
            "DOM extraction: meta=%.0fms body=%.0fms total=%.0fms | "
            "url=%s selector=%s label=%s text=%d chars",
            (t_meta - t_start) * 1000,
            (t_body - t_meta) * 1000,
            (t_body - t_start) * 1000,
            url[:60], selector, label or "(generic)", text_len,
        )

        return self._build_event(parsed, app_name, match)

    async def _fallback_extraction(
        self, app_name: str, is_chromium: bool
    ) -> Event | None:
        """Fall back to simple URL + title extraction without JS execution."""
        if is_chromium:
            script = _chromium_fallback_script(app_name)
        else:
            script = _safari_fallback_script()

        raw = await self._run_applescript(script)
        if raw is None or "|||" not in raw:
            return None

        url, title = raw.split("|||", 1)
        url = url.strip()
        title = title.strip()

        if url == self._last_url:
            return None
        self._last_url = url

        domain = urlparse(url).netloc if url else ""
        return Event(
            agent_name=self.name,
            app_type=AppType.BROWSER,
            summary=f"Viewing: {title} ({domain})",
            metadata={"url": url, "title": title, "meta": {}, "browser": app_name},
        )

    def _build_event(
        self,
        parsed: dict[str, Any],
        app_name: str,
        match: dict[str, str] | None,
    ) -> Event:
        url = parsed.get("url", "")
        title = parsed.get("title", "")
        domain = urlparse(url).netloc if url else ""

        metadata: dict[str, Any] = {
            "url": url,
            "title": title,
            "meta": parsed.get("meta", {}),
            "browser": app_name,
        }

        text = parsed.get("text", "")
        if match:
            label = match.get("label", "")
            if text:
                metadata["text"] = text[:MAX_TEXT_LENGTH]
            metadata["allowlist_label"] = label
            summary = f"Viewing [{label}]: {title} ({domain})"
        else:
            if text:
                metadata["text"] = text[:GENERIC_TEXT_LENGTH]
            metadata["allowlist_label"] = ""
            summary = f"Viewing: {title} ({domain})"

        return Event(
            agent_name=self.name,
            app_type=AppType.BROWSER,
            summary=summary,
            metadata=metadata,
        )

    async def _run_applescript(self, script: str) -> str | None:
        try:
            proc = await asyncio.create_subprocess_exec(
                "osascript", "-e", script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=OSASCRIPT_TIMEOUT_S
            )
            if proc.returncode != 0:
                logger.debug(
                    "osascript failed (rc=%d): %s",
                    proc.returncode,
                    stderr.decode(errors="replace")[:200],
                )
                return None
            return stdout.decode(errors="replace").strip()
        except asyncio.TimeoutError:
            logger.debug("osascript timed out after %ds", OSASCRIPT_TIMEOUT_S)
            try:
                proc.kill()
            except Exception:
                pass
            return None
        except Exception:
            logger.debug("osascript execution error", exc_info=True)
            return None

    @staticmethod
    def _parse_js_result(raw: str) -> dict[str, Any] | None:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.debug("Failed to parse JS result as JSON: %.200s", raw)
            return None
