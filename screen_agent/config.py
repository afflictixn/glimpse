"""Centralised configuration for the Screen Watcher Agent."""

from dataclasses import dataclass, field


@dataclass
class Config:
    # --- Screen capture ---
    capture_interval_sec: float = 2.0
    resize_width: int = 1024  # px, height scales proportionally

    # --- Change detection ---
    phash_threshold: int = 8  # hamming distance; lower = more sensitive
    ssim_threshold: float = 0.92  # 0‑1; higher = more sensitive

    # --- Ollama / local vision model ---
    ollama_url: str = "http://localhost:11434"
    vision_model: str = "gemma3:12b"
    ollama_timeout: int = 120  # seconds

    # --- Interest classification ---
    interest_threshold: int = 3  # 1‑5 scale; >= this triggers a proposal

    # --- Context / memory ---
    context_buffer_size: int = 10  # rolling window of recent analyses
    db_path: str = "screen_agent.db"

    # --- Escalation (strong reasoning API) ---
    escalation_provider: str = "auto"  # "auto" | "gemini" | "openai"
    gemini_model: str = "gemini-2.0-flash"
    openai_model: str = "gpt-5.4-mini"
    escalation_context_frames: int = 5  # how many past descriptions to send

    # --- Overlay server ---
    overlay_server_url: str = "ws://localhost:9321"

    # --- Categories the classifier can assign ---
    categories: list[str] = field(default_factory=lambda: [
        "recipe", "shopping", "coding", "research",
        "travel", "entertainment", "communication", "other",
    ])
