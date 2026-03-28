from __future__ import annotations

import asyncio
import logging
import threading
from enum import Enum

logger = logging.getLogger(__name__)


class CaptureTrigger(str, Enum):
    APP_SWITCH = "app_switch"
    WINDOW_FOCUS = "window_focus"
    CLICK = "click"
    TYPING_PAUSE = "typing_pause"
    CLIPBOARD = "clipboard"
    VISUAL_CHANGE = "visual_change"
    IDLE = "idle"
    MANUAL = "manual"


class ActivityKind(str, Enum):
    MOUSE = "mouse"
    KEYBOARD = "keyboard"
    SCROLL = "scroll"


class EventTap:
    def __init__(
        self,
        trigger_queue: asyncio.Queue[CaptureTrigger],
        loop: asyncio.AbstractEventLoop,
        activity_callback: callable | None = None,
    ) -> None:
        self._queue = trigger_queue
        self._loop = loop
        self._activity_callback = activity_callback
        self._thread: threading.Thread | None = None
        self._running = False
        self._tap = None
        self._run_loop_source = None

    def _push_trigger(self, trigger: CaptureTrigger) -> None:
        try:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, trigger)
        except Exception:
            logger.debug("Failed to push trigger %s", trigger, exc_info=True)

    def _notify_activity(self, kind: ActivityKind) -> None:
        if self._activity_callback:
            try:
                self._activity_callback(kind)
            except Exception:
                pass

    def _event_callback(self, proxy, event_type, event, refcon):
        from Quartz import (
            kCGEventKeyDown,
            kCGEventKeyUp,
            kCGEventLeftMouseDown,
            kCGEventMouseMoved,
            kCGEventRightMouseDown,
            kCGEventScrollWheel,
        )

        if event_type in (kCGEventLeftMouseDown, kCGEventRightMouseDown):
            self._push_trigger(CaptureTrigger.CLICK)
            self._notify_activity(ActivityKind.MOUSE)
        elif event_type == kCGEventKeyDown:
            self._notify_activity(ActivityKind.KEYBOARD)
            flags = event.flags() if hasattr(event, "flags") else 0
            try:
                from Quartz import CGEventGetFlags, CGEventGetIntegerValueField, kCGKeyboardEventKeycode
                flags = CGEventGetFlags(event)
                keycode = CGEventGetIntegerValueField(event, kCGKeyboardEventKeycode)
                cmd_mask = 1 << 20  # kCGEventFlagMaskCommand
                if flags & cmd_mask:
                    if keycode in (8, 7, 9):  # C, X, V
                        self._push_trigger(CaptureTrigger.CLIPBOARD)
            except Exception:
                pass
        elif event_type == kCGEventKeyUp:
            self._notify_activity(ActivityKind.KEYBOARD)
        elif event_type == kCGEventScrollWheel:
            self._notify_activity(ActivityKind.SCROLL)
        elif event_type == kCGEventMouseMoved:
            self._notify_activity(ActivityKind.MOUSE)

        return event

    def _setup_workspace_observers(self) -> None:
        try:
            from AppKit import NSWorkspace
            from Foundation import NSNotificationCenter

            ws = NSWorkspace.sharedWorkspace()
            nc = ws.notificationCenter()

            def app_activated(notification):
                self._push_trigger(CaptureTrigger.APP_SWITCH)

            nc.addObserverForName_object_queue_usingBlock_(
                "NSWorkspaceDidActivateApplicationNotification",
                None,
                None,
                app_activated,
            )
            logger.debug("Workspace observers installed")
        except Exception:
            logger.warning("Failed to install workspace observers", exc_info=True)

    def _run_tap(self) -> None:
        try:
            from Quartz import (
                CFMachPortCreateRunLoopSource,
                CFRunLoopAddSource,
                CFRunLoopGetCurrent,
                CFRunLoopRun,
                CGEventMaskBit,
                CGEventTapCreate,
                kCFAllocatorDefault,
                kCFRunLoopCommonModes,
                kCGEventKeyDown,
                kCGEventKeyUp,
                kCGEventLeftMouseDown,
                kCGEventMouseMoved,
                kCGEventRightMouseDown,
                kCGEventScrollWheel,
                kCGEventTapOptionListenOnly,
                kCGHeadInsertEventTap,
                kCGSessionEventTap,
            )

            mask = (
                CGEventMaskBit(kCGEventLeftMouseDown)
                | CGEventMaskBit(kCGEventRightMouseDown)
                | CGEventMaskBit(kCGEventKeyDown)
                | CGEventMaskBit(kCGEventKeyUp)
                | CGEventMaskBit(kCGEventScrollWheel)
                | CGEventMaskBit(kCGEventMouseMoved)
            )

            self._tap = CGEventTapCreate(
                kCGSessionEventTap,
                kCGHeadInsertEventTap,
                kCGEventTapOptionListenOnly,
                mask,
                self._event_callback,
                None,
            )

            if self._tap is None:
                logger.error(
                    "CGEventTapCreate returned None (missing accessibility permission?)"
                )
                return

            self._run_loop_source = CFMachPortCreateRunLoopSource(
                kCFAllocatorDefault, self._tap, 0
            )
            run_loop = CFRunLoopGetCurrent()
            CFRunLoopAddSource(run_loop, self._run_loop_source, kCFRunLoopCommonModes)

            self._setup_workspace_observers()

            logger.info("EventTap started on background thread")
            CFRunLoopRun()

        except Exception:
            logger.error("EventTap thread crashed", exc_info=True)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_tap, daemon=True, name="event-tap")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._tap:
            try:
                from Quartz import CFRunLoopStop, CFRunLoopGetCurrent
            except Exception:
                pass
