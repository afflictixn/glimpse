from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Settings:
    port: int = 3030
    data_dir: Path = field(default_factory=lambda: Path.home() / ".glimpse")
    jpeg_quality: int = 80

    # Capture timing
    poll_interval_ms: int = 50
    min_capture_interval_ms: int = 200
    idle_capture_interval_ms: int = 30_000
    typing_pause_delay_ms: int = 500
    visual_check_interval_ms: int = 3_000
    visual_change_threshold: float = 0.05
    capture_timeout_s: int = 15

    # Data retention
    max_retention_days: int = 7
    max_db_size_mb: int = 10_000
    cleanup_interval_hours: int = 1

    # Ollama / Gemma agent
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "gemma3:12b"
    include_ocr: bool = False
    ollama_timeout_s: int = 30

    # General agent
    overlay_ws_url: str = "ws://localhost:9321"

    # Debug
    debug: bool = False

    @property
    def db_path(self) -> Path:
        return self.data_dir / "glimpse.db"

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
