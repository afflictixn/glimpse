# Glimpse — Agent Guidelines

## What is Glimpse

Glimpse is a macOS-native screen activity capture and intelligence service. It continuously captures screenshots triggered by user input events, runs OCR and vision-LLM analysis, and exposes everything through a FastAPI REST API backed by SQLite with FTS5 full-text search.

```

## Dataflow

```
┌─────────────────────────── INPUT ───────────────────────────┐
│                                                             │
│  EventTap (background thread)                               │
│  ├─ CGEventTap: mouse clicks, keyboard, scroll, clipboard   │
│  ├─ NSWorkspace: app activation notifications               │
│  └─ pushes CaptureTrigger → asyncio.Queue                   │
│                                                             │
│  ActivityFeed                                               │
│  ├─ TYPING_PAUSE: keyboard idle > 500ms                     │
│  └─ IDLE: no activity > 30s                                 │
│                                                             │
│  FrameComparer                                              │
│  └─ VISUAL_CHANGE: histogram distance > threshold           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────── CAPTURE (CaptureLoop) ──────────────────┐
│  1. capture_screen() → PIL Image                            │
│  2. get_focused_app() → app_name, window_name               │
│  3. SnapshotWriter.save() → JPEG on disk                    │
│  4. DatabaseManager.insert_frame() → frame_id               │
│  5. perform_ocr(image) → OCRResult → insert_ocr + ocr_fts  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────── PROCESSING (parallel) ──────────────────┐
│  ProcessAgents (e.g. GemmaAgent)                            │
│  └─ process(image, ocr_text, app, window) → Event           │
│     └─ insert_event + events_fts                            │
│                                                             │
│  ContextProviders (extensible, none wired yet)              │
│  └─ collect(app, window) → [AdditionalContext]              │
│     └─ insert_context + context_fts                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────── INTELLIGENCE ───────────────────────────┐
│  IntelligenceLayer (async queue consumer)                    │
│  └─ Event → ReasoningAgent.reason(event, db) → Action       │
│     └─ insert_action + actions_fts                          │
└─────────────────────────────────────────────────────────────┘
                    │                          │
                    ▼                          ▼
┌─────────────── STORAGE ──────────┐ ┌──────────── GENERAL AGENT ────────┐
│  SQLite (WAL) + FTS5             │ │  In-process, long-running loop     │
│  frames, ocr_text, events,       │ │  Receives events + actions via     │
│  context, actions + FTS indexes  │ │  direct method calls from capture  │
│  JPEG files ~/.glimpse/data/     │ │  loop and intelligence layer       │
│  Periodic retention cleanup      │ │  Uses tools (DB, web search stubs) │
│                                  │ │  Pushes proposals → overlay via WS │
│                                  │ │  Maintains conversation context    │
└──────────────────────────────────┘ └──────────────────────────────────┘
                    │
                    ▼
┌──────────────────── OUTPUT (FastAPI :3030) ─────────────────┐
│  /search        — OCR full-text search                      │
│  /frames/{id}   — frame image + metadata + OCR + events     │
│  /events        — event listing with FTS                    │
│  /context       — context search + push                     │
│  /actions       — action search by type/agent/event         │
│  /health        — liveness, uptime, row counts              │
│  /raw_sql       — ad-hoc SELECT queries                     │
│  /agent/push    — push events/actions to general agent      │
│  /agent/chat    — user conversation with general agent      │
│  /agent/status  — agent health, queue depth, session state  │
└─────────────────────────────────────────────────────────────┘
```

## Key abstractions

| Interface | File | Implementors |
|-----------|------|-------------|
| `ProcessAgent` (ABC) | `src/process/process_agent.py` | `GemmaAgent`, `NoOpAgent` |
| `ReasoningAgent` (ABC) | `src/intelligence/reasoning_agent.py` | (none wired yet) |
| `ContextProvider` (ABC) | `src/context/context_provider.py` | (none wired yet) |
| `GeneralAgent` | `src/general_agent/agent.py` | Implemented — queue consumer, tool dispatch, overlay push |
| `ToolRegistry` | `src/general_agent/tools.py` | `db_query`, `db_raw_sql`, `web_search` (stub), `price_lookup` (stub), `contact_lookup` |

ProcessAgent, ReasoningAgent, and ContextProvider follow a plug-in pattern — add new implementations and register them in `main.py`. The GeneralAgent is a singleton wired in `main.py` that receives pushes from the capture loop and intelligence layer.

## Database schema

Five core tables with cascading deletes from `frames`:

- **frames** — timestamp, snapshot path, app/window, trigger type, content hash
- **ocr_text** — extracted text + confidence per frame
- **events** — agent-produced summaries with app_type + metadata JSON
- **context** — external context entries (source, content_type, content)
- **actions** — reasoning-produced actions tied to events

Each has a companion FTS5 virtual table for full-text search.

## Pre-commit test checklist

Run the full test suite from the project root before every commit:

```bash
pytest tests/ -v
```

### Test files

| File | What it covers |
|------|---------------|
| `tests/test_health.py` | Domain models, DB insert/search/FTS for OCR, events, context, and actions, row counts, raw SQL guard (rejects non-SELECT), `FrameComparer`, `ActivityFeed`, `NoOpAgent`, `IntelligenceLayer` with stub and failing reasoning agents, full ASGI API integration tests for all routes |
| `tests/test_gemma_agent.py` | `GemmaAgent` prompt building, JSON response parsing, image encoding, fallback on malformed JSON, async handling when Ollama is unreachable; live Ollama integration tests (auto-skipped when Ollama is not running) |

### Quick commands

```bash
# Full suite (recommended before every commit)
pytest tests/ -v

# Only API / storage / core logic tests
pytest tests/test_health.py -v

# Only GemmaAgent unit + integration tests
pytest tests/test_gemma_agent.py -v -s
```

## Running the app

**Quick start (all services):**

```bash
./start.sh          # starts Ollama, Python backend, Swift overlay
./start.sh --kill   # stops everything
```

This launches three processes:
1. **Ollama** (port 11434) — local LLM server for Gemma 3 12B
2. **Python backend** (port 3030) — capture loop, OCR, agents, API, general agent
3. **Swift overlay** — macOS floating panel (Cmd+Shift+O to toggle), WebSocket on port 9321

Logs go to `/tmp/glimpse-backend.log` and `/tmp/glimpse-overlay.log`.

**Backend only:**

```bash
python -m src.main --port 3030 --data-dir ~/.glimpse
```

Requires macOS with accessibility permissions (for CGEventTap) and optionally a local Ollama instance with `gemma3:12b` pulled.

## Conventions

- Python 3.11+, async-first with `asyncio`
- All DB writes go through `DatabaseManager` with an async write lock
- macOS-native APIs via `pyobjc` (Quartz, Vision, AppKit)
- Tests use `pytest-asyncio` and `httpx.AsyncClient` for ASGI testing
- Imports use `from src.…` (project root must be on `sys.path`)

## Update AGENTS.md and CLAUDE.md
Check if these files are still relevant and update if needed before making any git push operation