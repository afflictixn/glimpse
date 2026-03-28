"""Tool registry for the general agent.

Each tool is an async callable that the agent can invoke mid-conversation
to gather information before deciding what to surface to the user.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Awaitable

from src.llm.types import ToolSpec
from src.storage.database import DatabaseManager

logger = logging.getLogger(__name__)


@dataclass
class RegisteredTool:
    """Pairs a portable ToolSpec with its implementation."""
    spec: ToolSpec
    fn: Callable[..., Awaitable[str]]


class ToolRegistry:
    """Registry of tools available to the general agent."""

    def __init__(self, db: DatabaseManager) -> None:
        self._db = db
        self._tools: dict[str, RegisteredTool] = {}
        self._register_builtins()

    _REDDIT_HEADERS = {"User-Agent": "ZExp/1.0"}

    def _register_builtins(self) -> None:
        self.register(RegisteredTool(
            spec=ToolSpec(
                name="db_query",
                description=(
                    "Query the Z Exp database. Supports full-text search across OCR text, "
                    "events, context, and actions. Use this to look up what the user was doing, "
                    "find past screen content, or check event/action history."
                ),
                parameters={
                    "table": "One of: ocr, events, context, actions",
                    "query": "FTS5 search query string",
                    "limit": "Max results (default 10)",
                },
            ),
            fn=self._db_query,
        ))
        self.register(RegisteredTool(
            spec=ToolSpec(
                name="db_raw_sql",
                description=(
                    "Execute a read-only SELECT query against the Z Exp SQLite database. "
                    "Only SELECT statements are allowed. "
                    "Schema: "
                    "frames(id, timestamp, snapshot_path, app_name, window_name, focused, capture_trigger, content_hash) | "
                    "ocr_text(id, frame_id, text, text_json, confidence) | "
                    "events(id, frame_id, agent_name, app_type, summary, metadata, created_at) | "
                    "context(id, frame_id, source, content_type, content, metadata, created_at) | "
                    "actions(id, event_id, frame_id, agent_name, action_type, action_description, metadata, created_at) | "
                    "memory(id, entity_type, entity_name, fact, source, confidence, created_at, last_seen). "
                    "Note: only 'frames' has 'timestamp'; all other tables use 'created_at'."
                ),
                parameters={
                    "sql": "SELECT query to execute (only SELECT allowed)",
                    "limit": "Max rows (default 20)",
                },
            ),
            fn=self._db_raw_sql,
        ))
        self.register(RegisteredTool(
            spec=ToolSpec(
                name="fetch_url",
                description=(
                    "Fetch and extract text content from a URL. Use to read page content "
                    "the user is viewing — terms of service, product listings, articles, etc."
                ),
                parameters={"url": "The URL to fetch"},
            ),
            fn=self._fetch_url,
        ))
        self.register(RegisteredTool(
            spec=ToolSpec(
                name="contact_lookup",
                description=(
                    "Look up context about a person from Z Exp event history — "
                    "recent interactions, mentions, social context gathered from screen captures."
                ),
                parameters={"name": "Person's name to search for"},
            ),
            fn=self._contact_lookup,
        ))
        self.register(RegisteredTool(
            spec=ToolSpec(
                name="contacts_search",
                description=(
                    "Search macOS Contacts for a person. Returns name, emails, phones, "
                    "and birthday if available. Requires Contacts permission on macOS."
                ),
                parameters={"name": "Person's name to search for"},
            ),
            fn=self._contacts_search,
        ))
        self.register(RegisteredTool(
            spec=ToolSpec(
                name="calendar_events",
                description=(
                    "Fetch upcoming calendar events from macOS Calendar. "
                    "Use to check for scheduling conflicts, upcoming meetings, etc."
                ),
                parameters={"days_ahead": "Number of days to look ahead (default 7)"},
            ),
            fn=self._calendar_events,
        ))
        self.register(RegisteredTool(
            spec=ToolSpec(
                name="memory_store",
                description=(
                    "Store a fact about an entity (person, product, price, pattern, preference) "
                    "for long-term recall. Use to remember user habits, prices seen, people met, etc."
                ),
                parameters={
                    "entity_type": "One of: person, product, price, pattern, preference",
                    "entity_name": "Name of the entity (e.g. person name, product name)",
                    "fact": "The fact to remember",
                    "source": "Where this fact came from (optional)",
                },
            ),
            fn=self._memory_store,
        ))
        self.register(RegisteredTool(
            spec=ToolSpec(
                name="memory_query",
                description=(
                    "Search stored memories about entities. Use to recall user habits, "
                    "past prices, people context, product history, etc."
                ),
                parameters={
                    "query": "Search query for memories",
                    "entity_type": "Filter by type: person, product, price, pattern, preference (optional)",
                    "limit": "Max results (default 10)",
                },
            ),
            fn=self._memory_query,
        ))
        self.register(RegisteredTool(
            spec=ToolSpec(
                name="social_profile",
                description=(
                    "Look up a social media profile by handle. Returns display name, bio, "
                    "follower/following counts, and post count. Currently uses Bluesky — "
                    "pass handles like 'jay.bsky.social'."
                ),
                parameters={"handle": "Social media handle (e.g. 'jay.bsky.social')"},
            ),
            fn=self._social_profile,
        ))
        self.register(RegisteredTool(
            spec=ToolSpec(
                name="social_feed",
                description=(
                    "Get recent posts from a social media profile. Returns their latest posts "
                    "with text, timestamps, and engagement counts."
                ),
                parameters={
                    "handle": "Social media handle (e.g. 'jay.bsky.social')",
                    "limit": "Max posts to return (default 10)",
                },
            ),
            fn=self._social_feed,
        ))
        self.register(RegisteredTool(
            spec=ToolSpec(
                name="social_search",
                description=(
                    "Search social media posts by keyword. Returns matching posts with author, "
                    "text, timestamps, and engagement. Use for trending topics, fact checking, "
                    "or finding what people are saying about something."
                ),
                parameters={
                    "query": "Search query string",
                    "limit": "Max posts to return (default 10)",
                    "sort": "Sort by: 'top' or 'latest' (default 'latest')",
                },
            ),
            fn=self._social_search,
        ))

        self.register(RegisteredTool(
            spec=ToolSpec(
                name="imessage_recent",
                description=(
                    "Get recent iMessage/SMS conversations with message previews. "
                    "Returns the latest messages grouped by contact or group chat. "
                    "Requires Full Disk Access for the Python process."
                ),
                parameters={
                    "limit": "Max conversations to return (default 10)",
                },
            ),
            fn=self._imessage_recent,
        ))
        self.register(RegisteredTool(
            spec=ToolSpec(
                name="imessage_search",
                description=(
                    "Search iMessage/SMS history by keyword. Returns matching messages "
                    "with sender, timestamp, and conversation context. "
                    "Use to find past conversations about a topic or with a person."
                ),
                parameters={
                    "query": "Search query string",
                    "limit": "Max messages to return (default 20)",
                },
            ),
            fn=self._imessage_search,
        ))
        self.register(RegisteredTool(
            spec=ToolSpec(
                name="imessage_conversation",
                description=(
                    "Get messages from a specific iMessage conversation by contact name, "
                    "phone number, or email. Returns recent messages in chronological order."
                ),
                parameters={
                    "contact": "Contact name, phone number, or email to look up",
                    "limit": "Max messages to return (default 30)",
                },
            ),
            fn=self._imessage_conversation,
        ))
        self.register(RegisteredTool(
            spec=ToolSpec(
                name="imessage_send",
                description=(
                    "Send an iMessage or SMS to a phone number or email. "
                    "Uses macOS Shortcuts/AppleScript. The user will see the message "
                    "appear in their Messages app."
                ),
                parameters={
                    "to": "Phone number or email address to send to",
                    "message": "Message text to send",
                },
            ),
            fn=self._imessage_send,
        ))

        self.register(RegisteredTool(
            spec=ToolSpec(
                name="mail_recent",
                description=(
                    "Get recent emails from Apple Mail. Returns subject, sender, "
                    "date received, read status, and a preview of each email."
                ),
                parameters={
                    "limit": "Max emails to return (default 10)",
                },
            ),
            fn=self._mail_recent,
        ))
        self.register(RegisteredTool(
            spec=ToolSpec(
                name="mail_search",
                description=(
                    "Search Apple Mail by keyword. Searches subject lines, sender "
                    "addresses/names, and email body previews."
                ),
                parameters={
                    "query": "Search query string",
                    "limit": "Max emails to return (default 20)",
                },
            ),
            fn=self._mail_search,
        ))

        self.register(RegisteredTool(
            spec=ToolSpec(
                name="reddit_subreddit",
                description=(
                    "Get top/hot/new posts from a subreddit. No auth needed. "
                    "Use to check what's trending in a community or find discussions."
                ),
                parameters={
                    "subreddit": "Subreddit name without r/ (e.g. 'python', 'technology')",
                    "sort": "Sort by: hot, new, top, rising (default 'hot')",
                    "limit": "Max posts to return (default 10)",
                },
            ),
            fn=self._reddit_subreddit,
        ))
        self.register(RegisteredTool(
            spec=ToolSpec(
                name="reddit_comments",
                description=(
                    "Get top comments from a Reddit post. Pass the post URL or a post ID + subreddit. "
                    "Use to see what people are saying about a specific topic."
                ),
                parameters={
                    "url": "Full Reddit post URL (e.g. 'https://reddit.com/r/python/comments/abc123/...')",
                    "limit": "Max comments to return (default 15)",
                },
            ),
            fn=self._reddit_comments,
        ))
        self.register(RegisteredTool(
            spec=ToolSpec(
                name="reddit_search",
                description=(
                    "Search Reddit posts across all subreddits or within a specific one. "
                    "Use to find discussions, opinions, reviews, or troubleshooting threads."
                ),
                parameters={
                    "query": "Search query string",
                    "subreddit": "Limit to a specific subreddit (optional, leave empty for all)",
                    "sort": "Sort by: relevance, hot, top, new, comments (default 'relevance')",
                    "limit": "Max posts to return (default 10)",
                },
            ),
            fn=self._reddit_search,
        ))

    def register(self, tool: RegisteredTool) -> None:
        self._tools[tool.spec.name] = tool

    def get(self, name: str) -> RegisteredTool | None:
        return self._tools.get(name)

    def list_specs(self) -> list[ToolSpec]:
        """Return portable tool specs for the LLM provider."""
        return [t.spec for t in self._tools.values()]

    async def call(self, tool_name: str, **kwargs: Any) -> str:
        tool = self._tools.get(tool_name)
        if tool is None:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})
        try:
            return await tool.fn(**kwargs)
        except Exception as e:
            logger.error("Tool %s failed: %s", tool_name, e, exc_info=True)
            return json.dumps({"error": str(e)})

    # ── Built-in tool implementations ──────────────────────────

    async def _db_query(self, table: str = "ocr", query: str = "", limit: str = "10") -> str:
        n = min(int(limit), 50)
        if table == "ocr":
            rows = await self._db.search(query, limit=n)
        elif table == "events":
            rows = await self._db.search_events(query, limit=n)
        elif table == "context":
            rows = await self._db.search_context(query, limit=n)
        elif table == "actions":
            rows = await self._db.search_actions(query, limit=n)
        else:
            return json.dumps({"error": f"Unknown table: {table}. Use: ocr, events, context, actions"})
        return json.dumps(rows, default=str)

    async def _db_raw_sql(self, sql: str = "", limit: str = "20") -> str:
        rows = await self._db.execute_raw_sql(sql, limit=min(int(limit), 100))
        return json.dumps(rows, default=str)

    async def _web_search(self, query: str = "") -> str:
        import asyncio as _asyncio
        from duckduckgo_search import DDGS

        logger.info("web_search called: %s", query)

        def _sync_search() -> list:
            with DDGS() as ddgs:
                return list(ddgs.text(query, max_results=5))

        try:
            results = await _asyncio.to_thread(_sync_search)
            return json.dumps(results, default=str)
        except Exception as e:
            logger.error("Web search failed: %s", e)
            return json.dumps({"error": f"Web search failed: {e}"})

    async def _fetch_url(self, url: str = "") -> str:
        import httpx
        from bs4 import BeautifulSoup

        logger.info("fetch_url called: %s", url)
        try:
            async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
                resp = await client.get(url)
                resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)[:5000]
            title = soup.title.string.strip() if soup.title and soup.title.string else ""
            return json.dumps({"url": url, "title": title, "content": text})
        except Exception as e:
            logger.error("fetch_url failed: %s", e)
            return json.dumps({"error": f"Failed to fetch URL: {e}"})

    async def _price_lookup(self, product: str = "") -> str:
        logger.info("price_lookup called: %s", product)
        return await self._web_search(query=f"{product} price buy compare")

    async def _contact_lookup(self, name: str = "") -> str:
        ocr_hits = await self._db.search(name, limit=5)
        event_hits = await self._db.search_events(name, limit=5)
        return json.dumps({
            "ocr_mentions": ocr_hits,
            "event_mentions": event_hits,
        }, default=str)

    async def _contacts_search(self, name: str = "") -> str:
        """Search contacts via the overlay server (has Contacts permission)."""
        import httpx

        logger.info("contacts_search called: %s", name)
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(
                    "http://localhost:9322/contacts/search",
                    params={"q": name},
                )
                return resp.text
        except Exception as e:
            logger.error("contacts_search failed: %s", e)
            return json.dumps({"error": f"Contacts unavailable. Make sure the Z Exp overlay is running. ({e})"})

    async def _calendar_events(self, days_ahead: str = "7") -> str:
        """Fetch calendar events via the overlay's CalendarServer (port 9322)."""
        import httpx

        logger.info("calendar_events called: days_ahead=%s", days_ahead)
        n_days = min(int(days_ahead), 30)
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(
                    f"http://localhost:9322/calendar?days={n_days}",
                )
                return resp.text
        except Exception as e:
            logger.error("calendar_events failed: %s", e)
            return json.dumps({
                "error": f"Calendar unavailable. Make sure the Z Exp overlay is running. ({e})"
            })

    async def _memory_store(
        self,
        entity_type: str = "preference",
        entity_name: str = "",
        fact: str = "",
        source: str = "",
    ) -> str:
        logger.info("memory_store: %s/%s", entity_type, entity_name)
        valid_types = {"person", "product", "price", "pattern", "preference"}
        if entity_type not in valid_types:
            return json.dumps({"error": f"Invalid entity_type. Use one of: {', '.join(sorted(valid_types))}"})
        if not entity_name or not fact:
            return json.dumps({"error": "entity_name and fact are required"})

        row_id = await self._db.insert_memory(
            entity_type=entity_type,
            entity_name=entity_name,
            fact=fact,
            source=source or None,
        )
        return json.dumps({"stored": True, "memory_id": row_id})

    async def _memory_query(
        self,
        query: str = "",
        entity_type: str = "",
        limit: str = "10",
    ) -> str:
        n = min(int(limit), 50)
        rows = await self._db.search_memory(
            query=query,
            entity_type=entity_type or None,
            limit=n,
        )
        return json.dumps(rows, default=str)

    # ── Social media (Bluesky / AT Protocol) ──────────────────
    # Swap this base URL + response parsing to support Twitter/X later.

    _BSKY_BASE = "https://public.api.bsky.app/xrpc"

    async def _social_profile(self, handle: str = "") -> str:
        import httpx

        logger.info("social_profile called: %s", handle)
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    f"{self._BSKY_BASE}/app.bsky.actor.getProfile",
                    params={"actor": handle},
                )
                resp.raise_for_status()
                data = resp.json()
            return json.dumps({
                "handle": data.get("handle", ""),
                "display_name": data.get("displayName", ""),
                "bio": data.get("description", ""),
                "followers": data.get("followersCount", 0),
                "following": data.get("followsCount", 0),
                "posts": data.get("postsCount", 0),
                "avatar": data.get("avatar", ""),
            })
        except Exception as e:
            logger.error("social_profile failed: %s", e)
            return json.dumps({"error": f"Profile lookup failed: {e}"})

    async def _social_feed(self, handle: str = "", limit: str = "10") -> str:
        import httpx

        logger.info("social_feed called: %s limit=%s", handle, limit)
        n = min(int(limit), 50)
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    f"{self._BSKY_BASE}/app.bsky.feed.getAuthorFeed",
                    params={"actor": handle, "limit": n, "filter": "posts_no_replies"},
                )
                resp.raise_for_status()
                data = resp.json()

            posts = []
            for item in data.get("feed", []):
                post = item.get("post", {})
                record = post.get("record", {})
                posts.append({
                    "text": record.get("text", ""),
                    "created_at": record.get("createdAt", ""),
                    "likes": post.get("likeCount", 0),
                    "reposts": post.get("repostCount", 0),
                    "replies": post.get("replyCount", 0),
                })
            return json.dumps(posts, default=str)
        except Exception as e:
            logger.error("social_feed failed: %s", e)
            return json.dumps({"error": f"Feed lookup failed: {e}"})

    async def _social_search(self, query: str = "", limit: str = "10", sort: str = "latest") -> str:
        import httpx

        logger.info("social_search called: q=%s limit=%s sort=%s", query, limit, sort)
        n = min(int(limit), 50)
        if sort not in ("top", "latest"):
            sort = "latest"
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    f"{self._BSKY_BASE}/app.bsky.feed.searchPosts",
                    params={"q": query, "limit": n, "sort": sort},
                )
                # Search may require auth — fall back to web search
                if resp.status_code == 403:
                    logger.info("Bluesky search requires auth, falling back to web_search")
                    return await self._web_search(query=f"site:bsky.app {query}")
                resp.raise_for_status()
                data = resp.json()

            posts = []
            for post in data.get("posts", []):
                author = post.get("author", {})
                record = post.get("record", {})
                posts.append({
                    "author": author.get("handle", ""),
                    "author_name": author.get("displayName", ""),
                    "text": record.get("text", ""),
                    "created_at": record.get("createdAt", ""),
                    "likes": post.get("likeCount", 0),
                    "reposts": post.get("repostCount", 0),
                    "replies": post.get("replyCount", 0),
                })
            return json.dumps(posts, default=str)
        except Exception as e:
            logger.error("social_search failed: %s", e)
            return json.dumps({"error": f"Social search failed: {e}"})

    # ── Mail (via overlay server on port 9322) ─────────────────

    async def _mail_recent(self, limit: str = "10") -> str:
        import httpx

        logger.info("mail_recent called: limit=%s", limit)
        n = min(int(limit), 50)
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    "http://localhost:9322/mail/recent",
                    params={"limit": n},
                )
                return resp.text
        except Exception as e:
            logger.error("mail_recent failed: %s", e)
            return json.dumps({"error": f"Mail unavailable. Make sure the Z Exp overlay is running. ({e})"})

    async def _mail_search(self, query: str = "", limit: str = "20") -> str:
        import httpx

        logger.info("mail_search called: q=%s", query)
        n = min(int(limit), 100)
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    "http://localhost:9322/mail/search",
                    params={"q": query, "limit": n},
                )
                return resp.text
        except Exception as e:
            logger.error("mail_search failed: %s", e)
            return json.dumps({"error": f"Mail search unavailable. ({e})"})

    # ── Reddit (public JSON API, no auth) ─────────────────────

    async def _reddit_subreddit(
        self, subreddit: str = "", sort: str = "hot", limit: str = "10",
    ) -> str:
        import httpx

        logger.info("reddit_subreddit called: r/%s sort=%s", subreddit, sort)
        n = min(int(limit), 50)
        if sort not in ("hot", "new", "top", "rising"):
            sort = "hot"
        try:
            async with httpx.AsyncClient(timeout=10, headers=self._REDDIT_HEADERS, follow_redirects=True) as client:
                resp = await client.get(
                    f"https://www.reddit.com/r/{subreddit}/{sort}.json",
                    params={"limit": n},
                )
                resp.raise_for_status()
                data = resp.json()

            posts = []
            for child in data.get("data", {}).get("children", []):
                d = child.get("data", {})
                posts.append({
                    "title": d.get("title", ""),
                    "author": d.get("author", ""),
                    "score": d.get("score", 0),
                    "num_comments": d.get("num_comments", 0),
                    "url": f"https://reddit.com{d.get('permalink', '')}",
                    "selftext": (d.get("selftext", "") or "")[:500],
                    "created_utc": d.get("created_utc", 0),
                    "subreddit": d.get("subreddit", ""),
                })
            return json.dumps(posts, default=str)
        except Exception as e:
            logger.error("reddit_subreddit failed: %s", e)
            return json.dumps({"error": f"Reddit subreddit fetch failed: {e}"})

    async def _reddit_comments(self, url: str = "", limit: str = "15") -> str:
        import httpx

        logger.info("reddit_comments called: %s", url)
        n = min(int(limit), 50)
        # Normalize URL to .json endpoint
        json_url = url.rstrip("/")
        if not json_url.endswith(".json"):
            json_url += ".json"
        try:
            async with httpx.AsyncClient(timeout=10, headers=self._REDDIT_HEADERS, follow_redirects=True) as client:
                resp = await client.get(json_url, params={"limit": n, "sort": "top"})
                resp.raise_for_status()
                data = resp.json()

            # Reddit returns [post_listing, comments_listing]
            if not isinstance(data, list) or len(data) < 2:
                return json.dumps({"error": "Unexpected Reddit response format"})

            comments = []
            for child in data[1].get("data", {}).get("children", [])[:n]:
                d = child.get("data", {})
                if not d.get("body"):
                    continue
                comments.append({
                    "author": d.get("author", ""),
                    "body": (d.get("body", ""))[:500],
                    "score": d.get("score", 0),
                    "created_utc": d.get("created_utc", 0),
                })
            return json.dumps(comments, default=str)
        except Exception as e:
            logger.error("reddit_comments failed: %s", e)
            return json.dumps({"error": f"Reddit comments fetch failed: {e}"})

    async def _reddit_search(
        self, query: str = "", subreddit: str = "", sort: str = "relevance", limit: str = "10",
    ) -> str:
        import httpx

        logger.info("reddit_search called: q=%s sub=%s sort=%s", query, subreddit, sort)
        n = min(int(limit), 50)
        if sort not in ("relevance", "hot", "top", "new", "comments"):
            sort = "relevance"

        base = f"https://www.reddit.com/r/{subreddit}/search.json" if subreddit else "https://www.reddit.com/search.json"
        params: dict[str, Any] = {"q": query, "limit": n, "sort": sort}
        if subreddit:
            params["restrict_sr"] = "on"

        try:
            async with httpx.AsyncClient(timeout=10, headers=self._REDDIT_HEADERS, follow_redirects=True) as client:
                resp = await client.get(base, params=params)
                resp.raise_for_status()
                data = resp.json()

            posts = []
            for child in data.get("data", {}).get("children", []):
                d = child.get("data", {})
                posts.append({
                    "title": d.get("title", ""),
                    "author": d.get("author", ""),
                    "score": d.get("score", 0),
                    "num_comments": d.get("num_comments", 0),
                    "subreddit": d.get("subreddit", ""),
                    "url": f"https://reddit.com{d.get('permalink', '')}",
                    "selftext": (d.get("selftext", "") or "")[:300],
                    "created_utc": d.get("created_utc", 0),
                })
            return json.dumps(posts, default=str)
        except Exception as e:
            logger.error("reddit_search failed: %s", e)
            return json.dumps({"error": f"Reddit search failed: {e}"})

    # ── iMessage (via overlay server on port 9322) ─────────────
    # The overlay app has Full Disk Access to read ~/Library/Messages/chat.db

    async def _imessage_recent(self, limit: str = "10") -> str:
        import httpx

        logger.info("imessage_recent called: limit=%s", limit)
        n = min(int(limit), 50)
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    f"http://localhost:9322/imessage/recent",
                    params={"limit": n},
                )
                return resp.text
        except Exception as e:
            logger.error("imessage_recent failed: %s", e)
            return json.dumps({"error": f"iMessage unavailable. Make sure the Z Exp overlay is running. ({e})"})

    async def _imessage_search(self, query: str = "", limit: str = "20") -> str:
        import httpx

        logger.info("imessage_search called: q=%s", query)
        n = min(int(limit), 100)
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    f"http://localhost:9322/imessage/search",
                    params={"q": query, "limit": n},
                )
                return resp.text
        except Exception as e:
            logger.error("imessage_search failed: %s", e)
            return json.dumps({"error": f"iMessage search unavailable. ({e})"})

    async def _imessage_conversation(self, contact: str = "", limit: str = "30") -> str:
        import httpx

        logger.info("imessage_conversation called: contact=%s", contact)
        n = min(int(limit), 100)
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    f"http://localhost:9322/imessage/conversation",
                    params={"contact": contact, "limit": n},
                )
                return resp.text
        except Exception as e:
            logger.error("imessage_conversation failed: %s", e)
            return json.dumps({"error": f"iMessage conversation unavailable. ({e})"})

    async def _imessage_send(self, to: str = "", message: str = "") -> str:
        import asyncio as _asyncio

        logger.info("imessage_send called: to=%s", to)
        if not to or not message:
            return json.dumps({"error": "Both 'to' and 'message' are required"})

        safe_msg = message.replace("\\", "\\\\").replace('"', '\\"')
        safe_to = to.replace("\\", "\\\\").replace('"', '\\"')

        script = f'''
        tell application "Messages"
            set targetService to 1st account whose service type = iMessage
            set targetBuddy to participant "{safe_to}" of targetService
            send "{safe_msg}" to targetBuddy
        end tell
        '''
        try:
            proc = await _asyncio.create_subprocess_exec(
                "osascript", "-e", script,
                stdout=_asyncio.subprocess.PIPE,
                stderr=_asyncio.subprocess.PIPE,
            )
            stdout, stderr = await _asyncio.wait_for(proc.communicate(), timeout=10)
            if proc.returncode == 0:
                return json.dumps({"sent": True, "to": to, "message": message})
            else:
                return json.dumps({"error": f"AppleScript failed: {stderr.decode().strip()}"})
        except _asyncio.TimeoutError:
            proc.kill()
            logger.error("imessage_send timed out")
            return json.dumps({"error": "iMessage send timed out"})
        except Exception as e:
            logger.error("imessage_send failed: %s", e)
            return json.dumps({"error": f"iMessage send failed: {e}"})
