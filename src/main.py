from __future__ import annotations

import argparse
import asyncio
import logging
import signal
from pathlib import Path

import uvicorn

from src.api.server import create_app
from src.capture.activity_feed import ActivityFeed
from src.capture.event_tap import CaptureTrigger, EventTap
from src.capture.triggers import CaptureLoop
from src.config import Settings, set_settings
from src.general_agent.agent import GeneralAgent
from src.general_agent.tools import ToolRegistry
from src.general_agent.ws_manager import ConnectionManager
from src.llm import create_llm_client
from src.process.browser_content_agent import BrowserContentAgent
from src.process.process_agent import ProcessAgent
from src.context.context_provider import ContextProvider
from src.storage.database import DatabaseManager
from src.storage.snapshot_writer import SnapshotWriter
from src.voice.tts import VoiceClient

logger = logging.getLogger("zexp")

_DEFAULTS = Settings()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Z Exp - macOS screenshot capture & intelligence")
    parser.add_argument("--port", type=int, default=_DEFAULTS.port, help="API server port")
    parser.add_argument("--data-dir", type=str, default=str(_DEFAULTS.data_dir), help="Data directory")
    parser.add_argument("--jpeg-quality", type=int, default=_DEFAULTS.jpeg_quality, help="JPEG quality (1-100)")
    parser.add_argument("--retention-days", type=int, default=_DEFAULTS.max_retention_days, help="Data retention in days")
    parser.add_argument("--max-db-size-mb", type=int, default=_DEFAULTS.max_db_size_mb, help="Max database size in MB")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging for Z Exp internals (suppresses third-party noise)")
    parser.add_argument("--llm-provider", type=str, default=_DEFAULTS.llm_provider, help="LLM provider: openai, gemini, or ollama")
    parser.add_argument("--llm-model", type=str, default=_DEFAULTS.llm_model, help="LLM model name")
    parser.add_argument("--llm-reasoning-effort", type=str, default=_DEFAULTS.llm_reasoning_effort, help="Reasoning effort: low, medium, high, or none to disable")
    return parser.parse_args()


async def cleanup_task(db: DatabaseManager, interval_hours: int) -> None:
    interval_s = interval_hours * 3600
    while True:
        try:
            await asyncio.sleep(interval_s)
            deleted = await db.cleanup()
            if deleted > 0:
                logger.info("Retention cleanup removed %d frames", deleted)
        except asyncio.CancelledError:
            break
        except Exception:
            logger.error("Cleanup task error", exc_info=True)


async def run(settings: Settings) -> None:
    set_settings(settings)
    settings.ensure_dirs()

    db = DatabaseManager(settings)
    await db.initialize()

    tool_registry = ToolRegistry(db)

    process_agents: list[ProcessAgent] = [
        BrowserContentAgent(),
    ]
    context_providers: list[ContextProvider] = []

    # Voice (ElevenLabs TTS)
    voice: VoiceClient | None = None
    if settings.tts_enabled and settings.elevenlabs_api_key:
        voice = VoiceClient(
            api_key=settings.elevenlabs_api_key,
            voice_id=settings.elevenlabs_voice_id,
            model_id=settings.elevenlabs_model_id,
        )
        logger.info("ElevenLabs TTS enabled (voice: %s)", settings.elevenlabs_voice_id)

    llm_client = create_llm_client(
        settings.llm_provider,
        settings.llm_model,
        reasoning_effort=settings.llm_reasoning_effort,
    )

    ws_manager = ConnectionManager()

    general_agent = GeneralAgent(
        db=db,
        tools=tool_registry,
        llm=llm_client,
        ws_manager=ws_manager,
        voice=voice,
        importance_filter_enabled=settings.importance_filter_enabled,
    )

    trigger_queue: asyncio.Queue[CaptureTrigger] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    activity_feed = ActivityFeed(
        typing_pause_delay_ms=settings.typing_pause_delay_ms,
        idle_capture_interval_ms=settings.idle_capture_interval_ms,
    )

    event_tap = EventTap(
        trigger_queue=trigger_queue,
        loop=loop,
        activity_callback=activity_feed.record,
    )

    snapshot_writer = SnapshotWriter(settings)

    capture_loop = CaptureLoop(
        settings=settings,
        db=db,
        snapshot_writer=snapshot_writer,
        trigger_queue=trigger_queue,
        activity_feed=activity_feed,
        process_agents=process_agents,
        context_providers=context_providers,
        general_agent=general_agent,
    )

    app = create_app(
        db,
        general_agent=general_agent,
        snapshot_writer=snapshot_writer,
        process_agents=process_agents,
        context_providers=context_providers,
        ws_manager=ws_manager,
    )

    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=settings.port,
        log_level="debug" if settings.debug else "info",
        access_log=settings.debug,
    )
    server = uvicorn.Server(config)

    shutdown_event = asyncio.Event()

    def handle_signal() -> None:
        logger.info("Shutdown signal received")
        shutdown_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, handle_signal)

    # Attach WS debug log handler when running with --debug
    if settings.debug:
        from src.debug.ws_log_handler import WsLogHandler
        ws_handler = WsLogHandler(ws_manager, asyncio.get_running_loop())
        for name in ("src", "zexp"):
            logging.getLogger(name).addHandler(ws_handler)
        logger.info("Debug log streaming enabled → overlay left panel")

    server_task = asyncio.create_task(server.serve())
    event_tap.start()
    capture_task = asyncio.create_task(capture_loop.run())
    general_agent_task = asyncio.create_task(general_agent.run())
    cleanup = asyncio.create_task(
        cleanup_task(db, settings.cleanup_interval_hours)
    )

    logger.info("Z Exp started on port %d, data dir: %s", settings.port, settings.data_dir)

    await shutdown_event.wait()

    logger.info("Shutting down...")
    await capture_loop.stop()
    await general_agent.stop()
    server.should_exit = True
    capture_task.cancel()
    general_agent_task.cancel()
    cleanup.cancel()

    for t in (capture_task, general_agent_task, cleanup, server_task):
        try:
            await t
        except asyncio.CancelledError:
            pass

    event_tap.stop()
    await db.close()
    logger.info("Z Exp stopped")


def _configure_logging(args: argparse.Namespace) -> None:
    if args.debug:
        logging.basicConfig(
            level=logging.WARNING,
            format="%(asctime)s [%(levelname)s] %(name)s %(module)s:%(lineno)d: %(message)s",
            datefmt="%H:%M:%S",
        )
        logging.getLogger("src").setLevel(logging.DEBUG)
        logging.getLogger("zexp").setLevel(logging.DEBUG)
        logging.getLogger("uvicorn.access").setLevel(logging.INFO)
    else:
        logging.basicConfig(
            level=getattr(logging, args.log_level.upper(), logging.INFO),
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )


def main() -> None:
    args = parse_args()
    _configure_logging(args)

    settings = Settings(
        port=args.port,
        data_dir=Path(args.data_dir),
        jpeg_quality=args.jpeg_quality,
        max_retention_days=args.retention_days,
        max_db_size_mb=args.max_db_size_mb,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        llm_reasoning_effort=args.llm_reasoning_effort if args.llm_reasoning_effort != "none" else None,
        debug=args.debug,
    )

    asyncio.run(run(settings))


if __name__ == "__main__":
    main()
