# Z Exp

macOS-native ambient screen capture and intelligence system. Continuously captures screenshots triggered by user input events, runs OCR and vision-LLM analysis, and surfaces insights through a floating overlay — all backed by a FastAPI REST API and SQLite.

## Architecture

```
Triggers (EventTap, ActivityFeed, FrameComparer)
        │
        ▼
   CaptureLoop ──► JPEG + DB frame + Apple Vision OCR
        │
        ▼
  ProcessAgents ──► Gemma via Ollama / Browser Content
        │
        ▼
IntelligenceLayer ──► ReasoningAgents → Actions
        │
        ├──► GeneralAgent (tools, conversation context) ──► Overlay via WebSocket
        │
        └──► FastAPI REST API (:3030)
```

**Three runtime processes:**

| Service | Port | Description |
|---------|------|-------------|
| Ollama | 11434 | Local LLM server (Gemma) |
| Python backend | 3030 | Capture loop, OCR, agents, API |
| Swift overlay | 9321 (WS) | macOS floating panel (`Cmd+Shift+O`) |

## Quick Start

### Prerequisites

- macOS (Apple Silicon recommended)
- Python 3.11+
- [Ollama](https://ollama.com) installed
- Xcode Command Line Tools (for the Swift overlay)

### Setup

```bash
git clone <repo-url> && cd glimpse

python -m venv venv
source venv/bin/activate
pip install -e .
pip install python-dotenv google-genai openai   # providers not in pyproject.toml
```

### Environment Variables

Create a `.env` file in the project root:

```
GOOGLE_API_KEY=...        # required for Gemini vision/LLM
OPENAI_API_KEY=...        # required if using OpenAI as LLM provider
ELEVENLABS_API_KEY=...    # optional, enables TTS
ELEVENLABS_VOICE_ID=...   # optional, custom voice
```

### Run

```bash
./start.sh              # start everything (Ollama + overlay + backend)
./start.sh --kill       # stop all services
./start.sh --restart    # restart all
./start.sh --restart backend   # restart a single service (backend | overlay | ollama)
```

Or run the backend directly:

```bash
python -m src.main --port 3030
```

Logs: `/tmp/zexp-backend.log`, `/tmp/zexp-overlay.log`

### Verify

```bash
curl http://localhost:3030/health
```

## CLI Options

```
--port              API port (default: 3030)
--data-dir          Data directory (default: ~/.zexp)
--vision-provider   gemini | ollama (default: gemini)
--llm-provider      openai | gemini (default: gemini)
--llm-model         LLM model name
--gemini-vision-model   Gemini model for vision
--ollama-model      Ollama model name (default: gemma3:4b)
--ollama-url        Ollama base URL (default: http://localhost:11434)
--log-level         Logging level (default: INFO)
--debug             Verbose debug logging
--jpeg-quality      Screenshot JPEG quality 1-100 (default: 80)
--retention-days    Data retention period (default: 7)
--max-db-size-mb    Max DB size (default: 10000)
```

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Liveness check, uptime, row counts |
| `GET /search` | Full-text OCR search |
| `GET /frames/{id}` | Frame image + metadata + OCR + events |
| `GET /events` | Event listing with FTS |
| `GET /context` | Context search |
| `GET /actions` | Actions by type/agent/event |
| `POST /raw_sql` | Ad-hoc SELECT queries |
| `POST /agent/push` | Push events/actions to general agent |
| `POST /agent/chat` | Chat with the general agent |
| `GET /agent/status` | Agent health and session state |

## Storage

All data lives under `~/.zexp/` by default:

- `zexp.db` — SQLite (WAL mode) with FTS5 indexes
- `data/` — JPEG snapshots

## Testing

```bash
pytest tests/ -v
```

## Tech Stack

- **Capture & OCR:** PyObjC (Quartz, Vision, AppKit)
- **Processing:** Gemini Flash, Gemma 3 via Ollama
- **Backend:** FastAPI, uvicorn, aiosqlite
- **Overlay:** Swift, WebSocket
- **TTS:** ElevenLabs (optional)
