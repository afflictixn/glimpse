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

    def _register_builtins(self) -> None:
        self.register(RegisteredTool(
            spec=ToolSpec(
                name="db_query",
                description=(
                    "Query the Glimpse database. Supports full-text search across OCR text, "
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
                    "Execute a raw SELECT query against the Glimpse SQLite database. "
                    "Use for complex queries that the FTS search tools can't handle."
                ),
                parameters={
                    "sql": "SELECT query to execute",
                    "limit": "Max rows (default 20)",
                },
            ),
            fn=self._db_raw_sql,
        ))
        self.register(RegisteredTool(
            spec=ToolSpec(
                name="web_search",
                description=(
                    "Search the web for information. Returns a summary of top results. "
                    "Use for price checks, fact verification, looking up people/companies, etc."
                ),
                parameters={"query": "Search query string"},
            ),
            fn=self._web_search,
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
                name="price_lookup",
                description=(
                    "Look up current prices for a product across sources. "
                    "Searches the web with price-focused queries and returns comparisons."
                ),
                parameters={"product": "Product name or description to look up"},
            ),
            fn=self._price_lookup,
        ))
        self.register(RegisteredTool(
            spec=ToolSpec(
                name="contact_lookup",
                description=(
                    "Look up context about a person from Glimpse event history — "
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
        from duckduckgo_search import DDGS

        logger.info("web_search called: %s", query)
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=5))
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
        logger.info("contacts_search called: %s", name)
        try:
            import Contacts as CNContacts

            store = CNContacts.CNContactStore.alloc().init()
            predicate = CNContacts.CNContact.predicateForContactsMatchingName_(name)
            keys = [
                CNContacts.CNContactGivenNameKey,
                CNContacts.CNContactFamilyNameKey,
                CNContacts.CNContactEmailAddressesKey,
                CNContacts.CNContactPhoneNumbersKey,
                CNContacts.CNContactBirthdayKey,
                CNContacts.CNContactOrganizationNameKey,
                CNContacts.CNContactJobTitleKey,
            ]
            contacts, error = store.unifiedContactsMatchingPredicate_keysToFetch_error_(
                predicate, keys, None,
            )
            if error:
                return json.dumps({"error": str(error)})

            results = []
            for c in (contacts or [])[:10]:
                entry: dict[str, Any] = {
                    "given_name": c.givenName(),
                    "family_name": c.familyName(),
                    "organization": c.organizationName() or None,
                    "job_title": c.jobTitle() or None,
                }
                emails = c.emailAddresses()
                if emails:
                    entry["emails"] = [e.value() for e in emails]
                phones = c.phoneNumbers()
                if phones:
                    entry["phones"] = [p.value().stringValue() for p in phones]
                bday = c.birthday()
                if bday:
                    entry["birthday"] = f"{bday.month()}-{bday.day()}"
                results.append(entry)

            return json.dumps(results, default=str)
        except ImportError:
            return json.dumps({"error": "macOS Contacts framework not available"})
        except Exception as e:
            logger.error("contacts_search failed: %s", e)
            return json.dumps({"error": f"Contacts search failed: {e}"})

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
                "error": f"Calendar unavailable. Make sure the Glimpse overlay is running. ({e})"
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
