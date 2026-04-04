from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    port: int = 3030
    data_dir: Path = field(default_factory=lambda: Path.home() / ".zexp")
    jpeg_quality: int = 80

    # Capture timing
    poll_interval_ms: int = 250
    min_capture_interval_ms: int = 1000
    idle_capture_interval_ms: int = 30_000
    typing_pause_delay_ms: int = 500
    visual_check_interval_ms: int = 3_000
    visual_change_threshold: float = 0.05
    capture_timeout_s: int = 30

    # Data retention
    max_retention_days: int = 7
    max_db_size_mb: int = 10_000
    cleanup_interval_hours: int = 1

    # Vision process agent
    vision_provider: str = "ollama"  # "gemini", "openai", or "ollama"
    gemini_vision_model: str = "gemini-3-flash-preview"
    openai_vision_model: str = "gpt-5.4-nano"
    openai_image_detail: str = "low"
    openai_vision_timeout_s: float = 10.0
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "gemma3:4b"
    include_ocr: bool = False
    ollama_timeout_s: int = 30
    ollama_max_image_width: int = 960
    # LLM (general agent)
    llm_provider: str = "ollama"  # "openai", "gemini", or "ollama"
    llm_model: str = "gemma3:4b"
    llm_reasoning_effort: str | None = "medium"  # None, "low", "medium", "high"

    # General agent
    importance_filter_enabled: bool = False

    # ElevenLabs TTS
    elevenlabs_api_key: str = field(default_factory=lambda: os.getenv("ELEVENLABS_API_KEY", ""))
    elevenlabs_voice_id: str = field(default_factory=lambda: os.getenv("ELEVENLABS_VOICE_ID", "jqcCZkN6Knx8BJ5TBdYR"))
    elevenlabs_model_id: str = "eleven_multilingual_v2"
    tts_enabled: bool = field(default_factory=lambda: bool(os.getenv("ELEVENLABS_API_KEY")))

    # Debug
    debug: bool = False

    @property
    def db_path(self) -> Path:
        return self.data_dir / "zexp.db"

    @property
    def snapshots_dir(self) -> Path:
        return self.data_dir / "data"

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def set_settings(s: Settings) -> None:
    global _settings
    _settings = s
