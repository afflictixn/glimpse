"""Change detection using perceptual hashing and SSIM."""

from __future__ import annotations

from collections import deque
from typing import Optional

import imagehash
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

from screen_agent.config import Config


class ChangeDetector:
    """Decides whether a new frame differs enough from the previous one."""

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._prev_hash: Optional[imagehash.ImageHash] = None
        self._prev_gray: Optional[np.ndarray] = None
        self._hash_history: deque[imagehash.ImageHash] = deque(
            maxlen=cfg.context_buffer_size,
        )

    def has_changed(self, img: Image.Image) -> bool:
        """Return True if *img* is meaningfully different from the last frame."""
        current_hash = imagehash.phash(img)
        self._hash_history.append(current_hash)

        if self._prev_hash is None:
            self._prev_hash = current_hash
            self._prev_gray = self._to_gray(img)
            return True  # first frame is always "new"

        hamming = self._prev_hash - current_hash
        if hamming < self._cfg.phash_threshold:
            return False

        current_gray = self._to_gray(img)
        score = ssim(self._prev_gray, current_gray)

        self._prev_hash = current_hash
        self._prev_gray = current_gray

        return score < self._cfg.ssim_threshold

    @staticmethod
    def _to_gray(img: Image.Image) -> np.ndarray:
        """Convert to a fixed-size grayscale array for SSIM comparison."""
        small = img.resize((256, 256)).convert("L")
        return np.array(small)
