from __future__ import annotations

import logging
from dataclasses import dataclass

from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class WindowInfo:
    app_name: str
    window_title: str
    pid: int
    bounds: dict
    on_screen: bool


def capture_screen(monitor_id: int = 0) -> Image.Image:
    from Quartz import (
        CGRectInfinite,
        CGWindowListCreateImage,
        kCGWindowImageDefault,
        kCGWindowListOptionOnScreenOnly,
    )
    from AppKit import NSBitmapImageRep

    cg_image = CGWindowListCreateImage(
        CGRectInfinite,
        kCGWindowListOptionOnScreenOnly,
        0,
        kCGWindowImageDefault,
    )
    if cg_image is None:
        raise RuntimeError("CGWindowListCreateImage returned None (missing screen recording permission?)")

    bitmap = NSBitmapImageRep.alloc().initWithCGImage_(cg_image)
    width = bitmap.pixelsWide()
    height = bitmap.pixelsHigh()
    raw_data = bitmap.bitmapData()

    channels = bitmap.samplesPerPixel()
    bytes_per_row = bitmap.bytesPerRow()

    buf = bytes(raw_data[: bytes_per_row * height])

    if channels == 4:
        img = Image.frombytes("RGBA", (width, height), buf, "raw", "BGRA", bytes_per_row)
        return img.convert("RGB")
    else:
        return Image.frombytes("RGB", (width, height), buf, "raw", "RGB", bytes_per_row)


def get_window_info() -> list[WindowInfo]:
    from Quartz import (
        CGWindowListCopyWindowInfo,
        kCGNullWindowID,
        kCGWindowListOptionAll,
    )

    window_list = CGWindowListCopyWindowInfo(kCGWindowListOptionAll, kCGNullWindowID)
    if window_list is None:
        return []

    results = []
    for win in window_list:
        owner = win.get("kCGWindowOwnerName", "")
        title = win.get("kCGWindowName", "")
        pid = win.get("kCGWindowOwnerPID", 0)
        bounds = win.get("kCGWindowBounds", {})
        on_screen = win.get("kCGWindowIsOnscreen", False)
        if owner:
            results.append(WindowInfo(
                app_name=owner,
                window_title=title or "",
                pid=pid,
                bounds=dict(bounds) if bounds else {},
                on_screen=bool(on_screen),
            ))
    return results


def get_focused_app() -> tuple[str, str]:
    from AppKit import NSWorkspace

    ws = NSWorkspace.sharedWorkspace()
    front = ws.frontmostApplication()
    app_name = front.localizedName() if front else ""

    window_title = ""
    try:
        from ApplicationServices import (
            AXUIElementCreateApplication,
            AXUIElementCopyAttributeValue,
        )
        from CoreFoundation import kCFAllocatorDefault

        pid = front.processIdentifier() if front else 0
        if pid:
            app_ref = AXUIElementCreateApplication(pid)
            err, focused_window = AXUIElementCopyAttributeValue(
                app_ref, "AXFocusedWindow", None
            )
            if err == 0 and focused_window:
                err2, title = AXUIElementCopyAttributeValue(
                    focused_window, "AXTitle", None
                )
                if err2 == 0 and title:
                    window_title = str(title)
    except Exception:
        logger.debug("AX window title lookup failed", exc_info=True)

    return app_name, window_title
