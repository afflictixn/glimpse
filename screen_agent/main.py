"""Entry point — runs the capture/analysis loop with the Swift overlay client."""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
import threading
import time
from typing import Optional

from screen_agent.analyzer import LocalAnalyzer
from screen_agent.capture import ScreenCapture
from screen_agent.config import Config
from screen_agent.detector import ChangeDetector
from screen_agent.escalator import Escalator
from screen_agent.memory import Memory
from screen_agent.overlay_client import OverlayClient, UserAction
from screen_agent.proposer import ActionProposer, Proposal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("screen_agent")


class ScreenAgent:
    """Orchestrates capture → detect → analyze → propose → overlay."""

    def __init__(self) -> None:
        self._cfg = Config()
        self._capture = ScreenCapture(self._cfg)
        self._detector = ChangeDetector(self._cfg)
        self._analyzer = LocalAnalyzer(self._cfg)
        self._proposer = ActionProposer(self._cfg)
        self._memory = Memory(self._cfg)
        self._escalator = Escalator(self._cfg, self._memory)

        self._overlay = OverlayClient(self._cfg.overlay_server_url)
        self._overlay.on_action = self._handle_action
        self._overlay.on_pause = self._on_pause_toggled

        self._running = False
        self._paused = False

        self._current_proposal: Optional[Proposal] = None
        self._current_proposal_id: Optional[int] = None

    def start(self) -> None:
        self._overlay.send_set_assistant_label(self._escalator.provider_name)

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

        log.info(
            "Screen Agent started — model=%s, interval=%.1fs, overlay=%s",
            self._cfg.vision_model,
            self._cfg.capture_interval_sec,
            self._cfg.overlay_server_url,
        )

        try:
            asyncio.run(self._overlay.run())
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        finally:
            self._cleanup()

    def stop(self) -> None:
        """Signal all loops to stop. Thread-safe — can be called from any thread."""
        log.info("Shutting down…")
        self._running = False
        self._overlay.request_stop()

    def _cleanup(self) -> None:
        self._running = False
        self._capture.close()
        self._memory.close()

    # -- background capture loop ----------------------------------------

    def _loop(self) -> None:
        while self._running:
            if self._paused:
                time.sleep(0.5)
                continue

            try:
                img = self._capture.grab()

                if not self._detector.has_changed(img):
                    time.sleep(self._cfg.capture_interval_sec)
                    continue

                log.info("Analyzing screen…")
                result = self._analyzer.analyze(img)
                log.info(
                    "Analysis: score=%d cat=%s — %s",
                    result.score, result.category, result.description[:80],
                )

                analysis_id = self._memory.save_analysis(
                    description=result.description,
                    score=result.score,
                    category=result.category,
                    summary=result.summary,
                    raw=result.raw_classify,
                )

                proposal = self._proposer.propose(
                    result, self._analyzer.recent_context,
                )
                if proposal is not None:
                    self._current_proposal = proposal
                    self._current_proposal_id = self._memory.save_proposal(
                        text=proposal.text,
                        action_type=proposal.action_type,
                        confidence=proposal.confidence,
                        analysis_id=analysis_id,
                    )
                    self._overlay.send_show_proposal(
                        proposal.text, self._current_proposal_id,
                    )
                    log.info("Proposal: %s", proposal.text[:80])

            except Exception:
                log.exception("Error in analysis loop")

            time.sleep(self._cfg.capture_interval_sec)

    # -- user action handling -------------------------------------------

    def _handle_action(self, action: UserAction, text: str) -> None:
        pid = self._current_proposal_id

        if action == UserAction.DISMISS:
            if pid is not None:
                self._memory.record_user_response(pid, "dismiss")
            self._overlay.send_hide()

        elif action == UserAction.ACCEPT:
            if pid is not None:
                self._memory.record_user_response(pid, "accept")
            log.info("User accepted proposal")
            self._overlay.send_hide()

        elif action == UserAction.ESCALATE:
            if pid is not None:
                self._memory.record_user_response(pid, "escalate")
            self._escalate_to_api()

        elif action == UserAction.FOLLOW_UP:
            self._handle_follow_up(text)

    def _escalate_to_api(self) -> None:
        if self._current_proposal is None:
            return

        provider = self._escalator.provider_name
        self._overlay.send_show_conversation(f"Asking {provider}…")

        def _run() -> None:
            try:
                reply = self._escalator.escalate(
                    proposal_text=self._current_proposal.text,
                    current=self._current_proposal.source_analysis,
                    recent=self._analyzer.recent_context,
                    proposal_id=self._current_proposal_id,
                )
                self._overlay.send_show_conversation(reply)
            except Exception:
                log.exception("%s escalation failed", provider)
                self._overlay.send_show_conversation(
                    f"Failed to reach {provider}. "
                    "Is OPENAI_API_KEY or GEMINI_API_KEY set?"
                )

        threading.Thread(target=_run, daemon=True).start()

    def _handle_follow_up(self, text: str) -> None:
        def _run() -> None:
            try:
                reply = self._escalator.follow_up(text)
                self._overlay.send_append_conversation("assistant", reply)
            except Exception:
                log.exception("Follow-up failed")

        threading.Thread(target=_run, daemon=True).start()

    def _on_pause_toggled(self, paused: bool) -> None:
        self._paused = paused
        log.info("Agent %s", "paused" if paused else "resumed")


def main() -> None:
    agent = ScreenAgent()
    signal.signal(signal.SIGINT, lambda *_: agent.stop())
    signal.signal(signal.SIGTERM, lambda *_: agent.stop())
    agent.start()
    log.info("Exited.")


if __name__ == "__main__":
    main()
