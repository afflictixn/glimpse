"""Fast screen capture using python-mss with resize for model efficiency."""

from __future__ import annotations

import io
import base64
from typing import Optional

import mss
import mss.tools
from PIL import Image

from screen_agent.config import Config


class ScreenCapture:
    """Grabs screenshots via MSS and returns them as PIL Images / base64."""

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._sct: Optional[mss.mss] = None

    def _ensure_sct(self) -> mss.mss:
        if self._sct is None:
            self._sct = mss.mss()
        return self._sct

    def grab(self) -> Image.Image:
        """Capture the primary monitor and return a resized PIL Image."""
        sct = self._ensure_sct()
        monitor = sct.monitors[1]  # primary monitor (index 0 is "all")
        raw = sct.grab(monitor)
        img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
        return self._resize(img)

    def grab_base64(self) -> str:
        """Capture and return as a base64-encoded JPEG string."""
        img = self.grab()
        return self.image_to_base64(img)

    @staticmethod
    def image_to_base64(img: Image.Image) -> str:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def _resize(self, img: Image.Image) -> Image.Image:
        w = self._cfg.resize_width
        if img.width <= w:
            return img
        ratio = w / img.width
        h = int(img.height * ratio)
        return img.resize((w, h), Image.LANCZOS)

    def close(self) -> None:
        if self._sct is not None:
            self._sct.close()
            self._sct = None
