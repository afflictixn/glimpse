from __future__ import annotations

import hashlib

import numpy as np
from PIL import Image


class FrameComparer:
    def __init__(self) -> None:
        self._prev_hash: str | None = None
        self._prev_gray: np.ndarray | None = None

    def compare(self, image: Image.Image) -> float:
        small = image.resize(
            (image.width // 4, image.height // 4), Image.Resampling.NEAREST
        )
        raw = small.tobytes()
        current_hash = hashlib.md5(raw).hexdigest()

        if self._prev_hash is not None and current_hash == self._prev_hash:
            return 0.0

        gray = np.array(small.convert("L"), dtype=np.float64)

        if self._prev_gray is None:
            self._prev_hash = current_hash
            self._prev_gray = gray
            return 1.0

        hist_prev, _ = np.histogram(self._prev_gray, bins=256, range=(0, 256), density=True)
        hist_curr, _ = np.histogram(gray, bins=256, range=(0, 256), density=True)

        hist_prev = hist_prev / (hist_prev.sum() + 1e-10)
        hist_curr = hist_curr / (hist_curr.sum() + 1e-10)

        bc = np.sum(np.sqrt(hist_prev * hist_curr))
        distance = np.sqrt(max(0.0, 1.0 - bc))

        self._prev_hash = current_hash
        self._prev_gray = gray
        return float(distance)

    def reset(self) -> None:
        self._prev_hash = None
        self._prev_gray = None
