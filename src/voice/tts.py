"""ElevenLabs text-to-speech client.

Converts agent text responses to speech audio via the ElevenLabs streaming API.
Audio is returned as raw MP3 bytes or streamed to a callback.
"""
from __future__ import annotations

import asyncio
import io
import logging
import subprocess
import tempfile
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

ELEVENLABS_TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech"
VOICE_ENABLED_FILE = Path.home() / ".zexp" / "voice_enabled"


class VoiceClient:
    """Async ElevenLabs TTS client with local playback support."""

    def __init__(
        self,
        api_key: str,
        voice_id: str,
        model_id: str = "eleven_multilingual_v2",
    ) -> None:
        self._api_key = api_key
        self._voice_id = voice_id
        self._model_id = model_id
        self._http = httpx.AsyncClient(timeout=30.0)
        self._playback_lock = asyncio.Lock()

    async def synthesize(self, text: str) -> bytes:
        """Convert text to MP3 audio bytes via ElevenLabs API."""
        url = f"{ELEVENLABS_TTS_URL}/{self._voice_id}"
        headers = {
            "xi-api-key": self._api_key,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        }
        payload = {
            "text": text,
            "model_id": self._model_id,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
            },
        }

        response = await self._http.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.content

    @staticmethod
    def is_enabled() -> bool:
        """Check the shared voice_enabled file written by the overlay."""
        try:
            return VOICE_ENABLED_FILE.read_text().strip().lower() not in ("false", "0", "no")
        except (FileNotFoundError, OSError):
            return True  # default to enabled if file doesn't exist yet

    async def speak(self, text: str) -> None:
        """Synthesize text and play it through the system speakers.

        Uses macOS `afplay` for playback. Serialized so responses
        don't overlap.
        """
        if not text or not text.strip():
            return

        if not self.is_enabled():
            logger.debug("Voice disabled via %s — skipping TTS", VOICE_ENABLED_FILE)
            return

        try:
            audio = await self.synthesize(text)
        except httpx.HTTPStatusError as e:
            logger.error("ElevenLabs TTS failed (%s): %s", e.response.status_code, e.response.text[:200])
            return
        except httpx.HTTPError as e:
            logger.error("ElevenLabs TTS request error: %s", e)
            return

        async with self._playback_lock:
            await self._play_audio(audio)

    _PLAYBACK_TIMEOUT = 30  # max seconds for audio playback

    async def _play_audio(self, audio_bytes: bytes) -> None:
        """Write MP3 to a temp file and play via afplay (macOS)."""
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(audio_bytes)
            tmp_path = f.name

        try:
            proc = await asyncio.create_subprocess_exec(
                "afplay", tmp_path,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(proc.wait(), timeout=self._PLAYBACK_TIMEOUT)
        except asyncio.TimeoutError:
            logger.warning("Audio playback timed out after %ds, killing", self._PLAYBACK_TIMEOUT)
            try:
                proc.kill()
            except Exception:
                pass
        except FileNotFoundError:
            logger.warning("afplay not found — cannot play audio (macOS only)")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    async def close(self) -> None:
        await self._http.aclose()
