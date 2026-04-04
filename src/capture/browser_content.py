"""Extracts structured content from the active browser tab.

Chromium browsers use JXA (JavaScript for Automation) because Chrome's traditional
AppleScript bridge breaks on recent macOS — it reports 0 windows even when windows
are open. JXA uses a different code path (Application().windows[0]) and works reliably.

Safari keeps traditional AppleScript which still works.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from src.storage.models import AppType, Event

logger = logging.getLogger(__name__)

MAX_TEXT_LENGTH = 8000
GENERIC_TEXT_LENGTH = 4000
OSASCRIPT_TIMEOUT_S = 5

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

# ── Page-injected JS (runs inside the browser tab) ────────────────

_EXTRACT_META_ONLY_PAGE_JS = r"""(function(){
    var meta = {};
    document.querySelectorAll('meta[property], meta[name]').forEach(function(m){
        var key = m.getAttribute('property') || m.getAttribute('name');
        if (key) meta[key] = m.content;
    });
    return JSON.stringify({
        url: location.href,
        title: document.title,
        meta: meta
    });
})()"""

_EXTRACT_WITH_SELECTOR_PAGE_JS = r"""(function(){{
    var meta = {{}};
    document.querySelectorAll('meta[property], meta[name]').forEach(function(m){{
        var key = m.getAttribute('property') || m.getAttribute('name');
        if (key) meta[key] = m.content;
    }});
    var root = document.querySelector('{selector}');
    return JSON.stringify({{
        url: location.href,
        title: document.title,
        meta: meta,
        text: root ? root.innerText.substring(0, {max_len}) : ''
    }});
}})()"""


# ── JXA wrappers for Chromium (osascript -l JavaScript) ──────────

def _chromium_jxa_execute(app: str, page_js: str) -> str:
    """Build a JXA script that executes *page_js* inside the active tab."""
    escaped = (
        page_js
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
    )
    return (
        f'var app = Application("{app}");'
        f'if (app.windows.length === 0) throw "NO_WINDOWS";'
        f'app.windows[0].activeTab.execute({{javascript: "{escaped}"}});'
    )


def _chromium_jxa_fallback(app: str) -> str:
    """Build a JXA script that returns URL|||title without JS execution."""
    return (
        f'var app = Application("{app}");'
        f'if (app.windows.length === 0) throw "NO_WINDOWS";'
        f'var tab = app.windows[0].activeTab;'
        f'tab.url() + "|||" + tab.title();'
    )


# ── AppleScript wrappers for Safari ──────────────────────────────

def _safari_applescript_execute(page_js: str) -> str:
    escaped = (
        page_js
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
    )
    return (
        f'tell application "Safari" to do JavaScript '
        f'"{escaped}" in current tab of front window'
    )


def _safari_applescript_fallback() -> str:
    return (
        'tell application "Safari"\n'
        "  set tabUrl to URL of current tab of front window\n"
        "  set tabTitle to name of current tab of front window\n"
        '  return tabUrl & "|||" & tabTitle\n'
        "end tell"
    )


class BrowserContentAgent:
    """Extracts structured page content from the user's active browser tab."""

    name = "browser_content"

    def __init__(self, allowlist_path: str | Path | None = None) -> None:
        if allowlist_path is None:
            allowlist_path = Path(__file__).resolve().parents[2] / "config" / "browser_allowlist.json"
        self._allowlist_path = Path(allowlist_path)
        self._allowlist: list[dict[str, str]] = self._load_allowlist()
        self._last_url: str | None = None

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

    async def extract(
        self,
        app_name: str | None,
        window_name: str | None = None,
    ) -> Event | None:
        """Extract structured content from the active browser tab.

        Returns None if the app is not a browser or extraction fails.
        """
        if not app_name or app_name not in ALL_BROWSERS:
            return None

        t_start = time.monotonic()
        is_chromium = app_name in CHROMIUM_BROWSERS

        if is_chromium:
            script = _chromium_jxa_execute(app_name, _EXTRACT_META_ONLY_PAGE_JS)
            raw = await self._run_osascript(script, jxa=True)
        else:
            script = _safari_applescript_execute(_EXTRACT_META_ONLY_PAGE_JS)
            raw = await self._run_osascript(script, jxa=False)
        t_meta = time.monotonic()

        if raw is None:
            logger.debug("DOM meta extraction failed for %s, trying fallback", app_name)
            return await self._fallback_extraction(app_name, is_chromium)

        parsed = self._parse_js_result(raw)
        if parsed is None:
            return await self._fallback_extraction(app_name, is_chromium)

        url = parsed.get("url", "")

        if url and url == self._last_url:
            logger.debug("DOM dedup: same URL, skipping (%s)", url[:80])
            return None
        self._last_url = url

        match = self._match_allowlist(url)
        selector = match.get("selector", "body") if match else "body"
        max_len = MAX_TEXT_LENGTH if match else GENERIC_TEXT_LENGTH

        page_js = _EXTRACT_WITH_SELECTOR_PAGE_JS.format(
            selector=selector.replace("'", "\\'"),
            max_len=max_len,
        )
        if is_chromium:
            body_script = _chromium_jxa_execute(app_name, page_js)
            full_raw = await self._run_osascript(body_script, jxa=True)
        else:
            body_script = _safari_applescript_execute(page_js)
            full_raw = await self._run_osascript(body_script, jxa=False)
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
            script = _chromium_jxa_fallback(app_name)
            raw = await self._run_osascript(script, jxa=True)
        else:
            script = _safari_applescript_fallback()
            raw = await self._run_osascript(script, jxa=False)

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

    async def _run_osascript(self, script: str, *, jxa: bool = False) -> str | None:
        cmd = ["osascript"]
        if jxa:
            cmd += ["-l", "JavaScript"]
        cmd += ["-e", script]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=OSASCRIPT_TIMEOUT_S
            )
            if proc.returncode != 0:
                logger.debug(
                    "osascript failed (rc=%d, jxa=%s): %s",
                    proc.returncode, jxa,
                    stderr.decode(errors="replace")[:200],
                )
                return None
            return stdout.decode(errors="replace").strip()
        except asyncio.TimeoutError:
            logger.debug("osascript timed out after %ds (jxa=%s)", OSASCRIPT_TIMEOUT_S, jxa)
            try:
                proc.kill()
            except Exception:
                pass
            return None
        except Exception:
            logger.debug("osascript execution error (jxa=%s)", jxa, exc_info=True)
            return None

    @staticmethod
    def _parse_js_result(raw: str) -> dict[str, Any] | None:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.debug("Failed to parse JS result as JSON: %.200s", raw)
            return None
