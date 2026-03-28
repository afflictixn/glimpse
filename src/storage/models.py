from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class AppType(StrEnum):
    BROWSER = "browser"
    TERMINAL = "terminal"
    IDE = "ide"
    OTHER = "other"


@dataclass
class Frame:
    id: int | None = None
    timestamp: str = ""
    snapshot_path: str | None = None
    app_name: str | None = None
    window_name: str | None = None
    focused: bool = True
    capture_trigger: str = ""
    content_hash: str | None = None


@dataclass
class OCRResult:
    text: str = ""
    text_json: str = ""
    confidence: float = 0.0


@dataclass
class Event:
    frame_id: int | None = None
    agent_name: str = ""
    app_type: AppType = AppType.OTHER
    summary: str = ""
    metadata: dict = field(default_factory=dict)
    id: int | None = None


@dataclass
class AdditionalContext:
    frame_id: int | None = None
    source: str = ""
    content_type: str = ""
    content: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class Action:
    event_id: int | None = None
    frame_id: int | None = None
    agent_name: str = ""
    action_type: str = ""
    action_description: str = ""
    metadata: dict = field(default_factory=dict)
