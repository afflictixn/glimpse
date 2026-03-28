"""Tests for the general agent tool registry and all tool implementations.

Covers: db_query, db_raw_sql, web_search, fetch_url, price_lookup,
contact_lookup, contacts_search, calendar_events, memory_store, memory_query.

External services (DuckDuckGo, httpx, macOS Contacts/Calendar) are mocked
so tests run anywhere without network or macOS permissions.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from src.general_agent.tools import ToolRegistry, ToolSpec
from src.storage.database import DatabaseManager
from src.storage.models import Event, Frame, OCRResult, AppType


# ── Fixtures ──


@pytest_asyncio.fixture
async def tools(db: DatabaseManager) -> ToolRegistry:
    return ToolRegistry(db)


async def _seed_db(db: DatabaseManager) -> int:
    """Insert a frame + OCR + event so DB search tools have data."""
    fid = await db.insert_frame(Frame(
        timestamp="2025-06-01T10:00:00Z",
        app_name="Chrome",
        window_name="Amazon",
        capture_trigger="click",
    ))
    await db.insert_ocr(fid, OCRResult(
        text="Sony WH-1000XM5 headphones $299 at Best Buy",
        text_json="[]",
        confidence=0.95,
    ))
    await db.insert_event(fid, Event(
        agent_name="classifier",
        app_type=AppType.BROWSER,
        summary="User browsing headphones on Amazon",
        metadata={"url": "https://amazon.com"},
    ))
    return fid


# ── Registry ──


class TestToolRegistry:
    def test_list_specs_returns_all_builtin_tools(self, tools: ToolRegistry):
        specs = tools.list_specs()
        names = {s["name"] for s in specs}
        expected = {
            "db_query", "db_raw_sql", "web_search", "fetch_url",
            "price_lookup", "contact_lookup", "contacts_search",
            "calendar_events", "memory_store", "memory_query",
            "social_profile", "social_feed", "social_search",
        }
        assert expected == names

    def test_get_existing_tool(self, tools: ToolRegistry):
        spec = tools.get("web_search")
        assert spec is not None
        assert spec.name == "web_search"

    def test_get_nonexistent_tool(self, tools: ToolRegistry):
        assert tools.get("nonexistent") is None

    @pytest.mark.asyncio
    async def test_call_unknown_tool(self, tools: ToolRegistry):
        result = json.loads(await tools.call("nonexistent"))
        assert "error" in result
        assert "Unknown tool" in result["error"]

    def test_register_custom_tool(self, tools: ToolRegistry):
        async def dummy(**kwargs):
            return json.dumps({"ok": True})

        tools.register(ToolSpec(
            name="custom", description="test", parameters={}, fn=dummy,
        ))
        assert tools.get("custom") is not None

    @pytest.mark.asyncio
    async def test_call_catches_exceptions(self, tools: ToolRegistry):
        async def failing(**kwargs):
            raise RuntimeError("boom")

        tools.register(ToolSpec(
            name="failing", description="fails", parameters={}, fn=failing,
        ))
        result = json.loads(await tools.call("failing"))
        assert "error" in result
        assert "boom" in result["error"]


# ── db_query ──


@pytest.mark.asyncio
class TestDbQuery:
    async def test_search_ocr(self, tools: ToolRegistry, db: DatabaseManager):
        await _seed_db(db)
        result = json.loads(await tools.call("db_query", table="ocr", query="headphones"))
        assert len(result) >= 1
        assert "headphones" in result[0]["text"]

    async def test_search_events(self, tools: ToolRegistry, db: DatabaseManager):
        await _seed_db(db)
        result = json.loads(await tools.call("db_query", table="events", query="browsing"))
        assert len(result) >= 1

    async def test_search_unknown_table(self, tools: ToolRegistry, db: DatabaseManager):
        result = json.loads(await tools.call("db_query", table="nope", query="x"))
        assert "error" in result

    async def test_limit_respected(self, tools: ToolRegistry, db: DatabaseManager):
        await _seed_db(db)
        result = json.loads(await tools.call("db_query", table="ocr", query="headphones", limit="1"))
        assert len(result) <= 1


# ── db_raw_sql ──


@pytest.mark.asyncio
class TestDbRawSql:
    async def test_select_works(self, tools: ToolRegistry, db: DatabaseManager):
        await _seed_db(db)
        result = json.loads(await tools.call("db_raw_sql", sql="SELECT COUNT(*) as n FROM frames"))
        assert result[0]["n"] >= 1

    async def test_rejects_non_select(self, tools: ToolRegistry, db: DatabaseManager):
        result = json.loads(await tools.call("db_raw_sql", sql="DROP TABLE frames"))
        assert "error" in result


# ── web_search ──


@pytest.mark.asyncio
class TestWebSearch:
    async def test_returns_results(self, tools: ToolRegistry):
        mock_instance = MagicMock()
        mock_instance.__enter__ = MagicMock(return_value=mock_instance)
        mock_instance.__exit__ = MagicMock(return_value=False)
        mock_instance.text.return_value = [
            {"title": "Result 1", "href": "https://example.com", "body": "Some result"},
            {"title": "Result 2", "href": "https://example.com/2", "body": "Another result"},
        ]
        mock_ddgs_cls = MagicMock(return_value=mock_instance)

        mock_module = MagicMock()
        mock_module.DDGS = mock_ddgs_cls

        with patch.dict("sys.modules", {"duckduckgo_search": mock_module}):
            result = json.loads(await tools.call("web_search", query="test search"))
        assert len(result) == 2
        assert result[0]["title"] == "Result 1"

    async def test_handles_failure(self, tools: ToolRegistry):
        mock_module = MagicMock()
        mock_module.DDGS.side_effect = Exception("network error")

        with patch.dict("sys.modules", {"duckduckgo_search": mock_module}):
            result = json.loads(await tools.call("web_search", query="fail"))
        assert "error" in result


# ── fetch_url ──


@pytest.mark.asyncio
class TestFetchUrl:
    async def test_fetches_and_extracts_text(self, tools: ToolRegistry):
        html = """
        <html><head><title>Test Page</title></head>
        <body>
            <nav>Skip nav</nav>
            <main><p>Hello world content here</p></main>
            <script>var x = 1;</script>
            <footer>Footer</footer>
        </body></html>
        """
        mock_response = MagicMock()
        mock_response.text = html
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = json.loads(await tools.call("fetch_url", url="https://example.com"))

        assert result["title"] == "Test Page"
        assert "Hello world content here" in result["content"]
        # Script/nav/footer should be stripped
        assert "var x = 1" not in result["content"]
        assert "Skip nav" not in result["content"]
        assert "Footer" not in result["content"]

    async def test_handles_fetch_failure(self, tools: ToolRegistry):
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=Exception("connection refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = json.loads(await tools.call("fetch_url", url="https://bad.example.com"))

        assert "error" in result

    async def test_truncates_long_content(self, tools: ToolRegistry):
        long_text = "word " * 5000  # way more than 5000 chars
        html = f"<html><body><p>{long_text}</p></body></html>"
        mock_response = MagicMock()
        mock_response.text = html
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = json.loads(await tools.call("fetch_url", url="https://example.com"))

        assert len(result["content"]) <= 5000


# ── price_lookup ──


@pytest.mark.asyncio
class TestPriceLookup:
    async def test_delegates_to_web_search(self, tools: ToolRegistry):
        """price_lookup should call web_search under the hood."""
        called_with = {}

        async def fake_web_search(query: str = "") -> str:
            called_with["query"] = query
            return json.dumps([{"title": "Price result"}])

        tools._web_search = fake_web_search  # type: ignore[assignment]
        result = json.loads(await tools.call("price_lookup", product="iPhone 15"))
        assert "price" in called_with["query"].lower()
        assert "iPhone 15" in called_with["query"]


# ── contact_lookup (DB-based) ──


@pytest.mark.asyncio
class TestContactLookup:
    async def test_searches_ocr_and_events(self, tools: ToolRegistry, db: DatabaseManager):
        fid = await db.insert_frame(Frame(
            timestamp="2025-06-01T10:00:00Z", capture_trigger="click",
            app_name="Messages",
        ))
        await db.insert_ocr(fid, OCRResult(
            text="Meeting with John Smith tomorrow at 3pm",
            text_json="[]", confidence=0.9,
        ))
        await db.insert_event(fid, Event(
            agent_name="classifier", app_type=AppType.OTHER,
            summary="Chat with John Smith about project",
        ))
        result = json.loads(await tools.call("contact_lookup", name="John Smith"))
        assert len(result["ocr_mentions"]) >= 1
        assert len(result["event_mentions"]) >= 1


# ── contacts_search (macOS Contacts) ──


@pytest.mark.asyncio
class TestContactsSearch:
    async def test_returns_contacts(self, tools: ToolRegistry):
        """Mock the macOS Contacts framework."""
        mock_contact = MagicMock()
        mock_contact.givenName.return_value = "Jane"
        mock_contact.familyName.return_value = "Doe"
        mock_contact.organizationName.return_value = "Acme Corp"
        mock_contact.jobTitle.return_value = "Engineer"

        mock_email = MagicMock()
        mock_email.value.return_value = "jane@acme.com"
        mock_contact.emailAddresses.return_value = [mock_email]

        mock_phone_value = MagicMock()
        mock_phone_value.stringValue.return_value = "+1234567890"
        mock_phone = MagicMock()
        mock_phone.value.return_value = mock_phone_value
        mock_contact.phoneNumbers.return_value = [mock_phone]

        mock_birthday = MagicMock()
        mock_birthday.month.return_value = 3
        mock_birthday.day.return_value = 15
        mock_contact.birthday.return_value = mock_birthday

        mock_store = MagicMock()
        mock_store.unifiedContactsMatchingPredicate_keysToFetch_error_.return_value = (
            [mock_contact], None,
        )

        mock_cn = MagicMock()
        mock_cn.CNContactStore.alloc.return_value.init.return_value = mock_store
        mock_cn.CNContact.predicateForContactsMatchingName_.return_value = "predicate"
        mock_cn.CNContactGivenNameKey = "givenName"
        mock_cn.CNContactFamilyNameKey = "familyName"
        mock_cn.CNContactEmailAddressesKey = "emails"
        mock_cn.CNContactPhoneNumbersKey = "phones"
        mock_cn.CNContactBirthdayKey = "birthday"
        mock_cn.CNContactOrganizationNameKey = "org"
        mock_cn.CNContactJobTitleKey = "job"

        with patch.dict("sys.modules", {"Contacts": mock_cn}):
            result = json.loads(await tools.call("contacts_search", name="Jane"))

        assert len(result) == 1
        assert result[0]["given_name"] == "Jane"
        assert result[0]["family_name"] == "Doe"
        assert result[0]["emails"] == ["jane@acme.com"]
        assert result[0]["phones"] == ["+1234567890"]
        assert result[0]["birthday"] == "3-15"
        assert result[0]["organization"] == "Acme Corp"

    async def test_handles_import_error(self, tools: ToolRegistry):
        """Should gracefully handle missing Contacts framework."""
        with patch.dict("sys.modules", {"Contacts": None}):
            result = json.loads(await tools.call("contacts_search", name="Test"))
        assert "error" in result

    async def test_handles_contacts_error(self, tools: ToolRegistry):
        mock_store = MagicMock()
        mock_store.unifiedContactsMatchingPredicate_keysToFetch_error_.return_value = (
            None, "Permission denied",
        )
        mock_cn = MagicMock()
        mock_cn.CNContactStore.alloc.return_value.init.return_value = mock_store
        mock_cn.CNContact.predicateForContactsMatchingName_.return_value = "pred"
        mock_cn.CNContactGivenNameKey = "gn"
        mock_cn.CNContactFamilyNameKey = "fn"
        mock_cn.CNContactEmailAddressesKey = "e"
        mock_cn.CNContactPhoneNumbersKey = "p"
        mock_cn.CNContactBirthdayKey = "b"
        mock_cn.CNContactOrganizationNameKey = "o"
        mock_cn.CNContactJobTitleKey = "j"

        with patch.dict("sys.modules", {"Contacts": mock_cn}):
            result = json.loads(await tools.call("contacts_search", name="Test"))
        assert "error" in result


# ── calendar_events (via overlay CalendarServer on port 9322) ──


@pytest.mark.asyncio
class TestCalendarEvents:
    async def test_returns_events(self, tools: ToolRegistry):
        """Mock httpx to simulate the overlay's calendar HTTP endpoint."""
        calendar_json = json.dumps([
            {"title": "Team Standup", "start": "2025-06-02T09:00:00Z",
             "end": "2025-06-02T09:30:00Z", "all_day": False, "location": "Zoom"},
        ])
        mock_response = MagicMock()
        mock_response.text = calendar_json

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = json.loads(await tools.call("calendar_events", days_ahead="7"))

        assert len(result) == 1
        assert result[0]["title"] == "Team Standup"
        assert result[0]["location"] == "Zoom"
        assert result[0]["all_day"] is False
        # Verify it called the right URL
        mock_client.get.assert_called_once_with("http://localhost:9322/calendar?days=7")

    async def test_handles_overlay_unavailable(self, tools: ToolRegistry):
        """When the overlay isn't running, should return a helpful error."""
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=Exception("Connection refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = json.loads(await tools.call("calendar_events"))
        assert "error" in result
        assert "overlay" in result["error"].lower()

    async def test_caps_days_ahead(self, tools: ToolRegistry):
        """days_ahead should be capped at 30."""
        mock_response = MagicMock()
        mock_response.text = "[]"

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            await tools.call("calendar_events", days_ahead="100")

        # Should have capped to 30
        mock_client.get.assert_called_once_with("http://localhost:9322/calendar?days=30")


# ── memory_store ──


@pytest.mark.asyncio
class TestMemoryStore:
    async def test_stores_memory(self, tools: ToolRegistry, db: DatabaseManager):
        result = json.loads(await tools.call(
            "memory_store",
            entity_type="person",
            entity_name="Alice",
            fact="Works at Google as a PM",
            source="linkedin_screenshot",
        ))
        assert result["stored"] is True
        assert result["memory_id"] >= 1

    async def test_rejects_invalid_type(self, tools: ToolRegistry):
        result = json.loads(await tools.call(
            "memory_store",
            entity_type="invalid",
            entity_name="Test",
            fact="something",
        ))
        assert "error" in result

    async def test_requires_name_and_fact(self, tools: ToolRegistry):
        result = json.loads(await tools.call(
            "memory_store",
            entity_type="person",
            entity_name="",
            fact="something",
        ))
        assert "error" in result

        result = json.loads(await tools.call(
            "memory_store",
            entity_type="person",
            entity_name="Alice",
            fact="",
        ))
        assert "error" in result


# ── memory_query ──


@pytest.mark.asyncio
class TestMemoryQuery:
    async def test_query_returns_stored_memory(self, tools: ToolRegistry, db: DatabaseManager):
        # Store something first
        await tools.call(
            "memory_store",
            entity_type="product",
            entity_name="Sony WH-1000XM5",
            fact="Seen at $299 on Amazon",
            source="price_check",
        )
        result = json.loads(await tools.call("memory_query", query="Sony"))
        assert len(result) >= 1
        assert "Sony" in result[0]["entity_name"]
        assert "$299" in result[0]["fact"]

    async def test_filter_by_entity_type(self, tools: ToolRegistry, db: DatabaseManager):
        await tools.call(
            "memory_store", entity_type="person",
            entity_name="Bob", fact="Met at conference",
        )
        await tools.call(
            "memory_store", entity_type="product",
            entity_name="MacBook", fact="Priced at $1999",
        )
        result = json.loads(await tools.call(
            "memory_query", query="", entity_type="person",
        ))
        types = {r["entity_type"] for r in result}
        assert types == {"person"}

    async def test_empty_query_returns_all(self, tools: ToolRegistry, db: DatabaseManager):
        await tools.call(
            "memory_store", entity_type="preference",
            entity_name="food", fact="Likes Thai food",
        )
        result = json.loads(await tools.call("memory_query", query=""))
        assert len(result) >= 1


# ── social_profile (Bluesky) ──


@pytest.mark.asyncio
class TestSocialProfile:
    async def test_returns_profile(self, tools: ToolRegistry):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "handle": "jay.bsky.social",
            "displayName": "Jay Graber",
            "description": "CEO of Bluesky",
            "followersCount": 500000,
            "followsCount": 1200,
            "postsCount": 3400,
            "avatar": "https://example.com/avatar.jpg",
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = json.loads(await tools.call("social_profile", handle="jay.bsky.social"))

        assert result["handle"] == "jay.bsky.social"
        assert result["display_name"] == "Jay Graber"
        assert result["followers"] == 500000
        assert result["bio"] == "CEO of Bluesky"

    async def test_handles_not_found(self, tools: ToolRegistry):
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=Exception("404 Not Found"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = json.loads(await tools.call("social_profile", handle="nonexistent.bsky.social"))
        assert "error" in result


# ── social_feed (Bluesky) ──


@pytest.mark.asyncio
class TestSocialFeed:
    async def test_returns_posts(self, tools: ToolRegistry):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "feed": [
                {
                    "post": {
                        "record": {"text": "Hello world!", "createdAt": "2026-03-28T10:00:00Z"},
                        "likeCount": 42,
                        "repostCount": 5,
                        "replyCount": 3,
                    }
                },
                {
                    "post": {
                        "record": {"text": "Second post", "createdAt": "2026-03-27T09:00:00Z"},
                        "likeCount": 10,
                        "repostCount": 1,
                        "replyCount": 0,
                    }
                },
            ]
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = json.loads(await tools.call("social_feed", handle="jay.bsky.social", limit="5"))

        assert len(result) == 2
        assert result[0]["text"] == "Hello world!"
        assert result[0]["likes"] == 42
        assert result[1]["text"] == "Second post"

    async def test_handles_error(self, tools: ToolRegistry):
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=Exception("rate limited"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = json.loads(await tools.call("social_feed", handle="test.bsky.social"))
        assert "error" in result


# ── social_search (Bluesky) ──


@pytest.mark.asyncio
class TestSocialSearch:
    async def test_returns_search_results(self, tools: ToolRegistry):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "posts": [
                {
                    "author": {"handle": "alice.bsky.social", "displayName": "Alice"},
                    "record": {"text": "Bluesky is great!", "createdAt": "2026-03-28T12:00:00Z"},
                    "likeCount": 100,
                    "repostCount": 20,
                    "replyCount": 5,
                },
            ]
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = json.loads(await tools.call("social_search", query="bluesky", limit="5", sort="top"))

        assert len(result) == 1
        assert result[0]["author"] == "alice.bsky.social"
        assert result[0]["text"] == "Bluesky is great!"
        assert result[0]["likes"] == 100

    async def test_invalid_sort_defaults_to_latest(self, tools: ToolRegistry):
        mock_response = MagicMock()
        mock_response.json.return_value = {"posts": []}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            await tools.call("social_search", query="test", sort="invalid")

        # Check it was called with sort=latest (the default fallback)
        call_kwargs = mock_client.get.call_args
        assert call_kwargs[1]["params"]["sort"] == "latest"

    async def test_handles_error(self, tools: ToolRegistry):
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=Exception("network error"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = json.loads(await tools.call("social_search", query="test"))
        assert "error" in result


# ── Memory DB methods directly ──


@pytest.mark.asyncio
class TestMemoryDatabase:
    async def test_insert_and_search(self, db: DatabaseManager):
        mid = await db.insert_memory(
            entity_type="person",
            entity_name="Charlie",
            fact="Birthday is March 15",
            source="calendar",
        )
        assert mid >= 1
        results = await db.search_memory("Charlie")
        assert len(results) == 1
        assert results[0]["entity_name"] == "Charlie"

    async def test_update_last_seen(self, db: DatabaseManager):
        mid = await db.insert_memory(
            entity_type="product", entity_name="Widget", fact="Costs $50",
        )
        old = (await db.search_memory("Widget"))[0]["last_seen"]
        await db.update_memory_last_seen(mid)
        new = (await db.search_memory("Widget"))[0]["last_seen"]
        assert new >= old

    async def test_delete_memory(self, db: DatabaseManager):
        mid = await db.insert_memory(
            entity_type="pattern", entity_name="browsing", fact="Checks Reddit when stuck",
        )
        assert await db.delete_memory(mid) is True
        results = await db.search_memory("browsing")
        assert len(results) == 0

    async def test_delete_nonexistent(self, db: DatabaseManager):
        assert await db.delete_memory(99999) is False

    async def test_search_by_entity_type(self, db: DatabaseManager):
        await db.insert_memory(
            entity_type="person", entity_name="Diana", fact="VP at Acme",
        )
        await db.insert_memory(
            entity_type="product", entity_name="Laptop", fact="$1200",
        )
        results = await db.search_memory("", entity_type="person")
        assert all(r["entity_type"] == "person" for r in results)

    async def test_counts_includes_memory(self, db: DatabaseManager):
        counts = await db.get_counts()
        assert "memory" in counts
        assert counts["memory"] == 0
        await db.insert_memory(
            entity_type="preference", entity_name="coffee", fact="Likes espresso",
        )
        counts = await db.get_counts()
        assert counts["memory"] == 1
