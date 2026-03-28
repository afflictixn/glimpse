from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from src.config import Settings

logger = logging.getLogger(__name__)


class SnapshotWriter:
    def __init__(self, settings: Settings) -> None:
        self._dir = settings.snapshots_dir
        self._quality = settings.jpeg_quality
        self._dir.mkdir(parents=True, exist_ok=True)

    def save(self, image: Image.Image, frame_id: int | None = None) -> str:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        suffix = f"_{frame_id}" if frame_id is not None else ""
        filename = f"frame_{ts}{suffix}.jpg"
        path = self._dir / filename
        image.save(str(path), "JPEG", quality=self._quality)
        logger.debug("Saved snapshot: %s", path)
        return str(path)
