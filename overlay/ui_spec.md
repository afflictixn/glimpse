# Overlay UI Spec — Three Modes

## Overview

The overlay has three distinct UI modes that the user flows between. Each mode increases the level of engagement — from passive ambient awareness to full interactive conversation.

## Modes

### 1. Ambient (default)

The "leave me alone" mode. Z is watching and will speak up when it matters.

- **Visible:** Edge glow orb only (left + right screen edges, breathing animation)
- **Voice:** Speaks proactive suggestions aloud (if voice is enabled in Settings)
- **Text:** Brief suggestion pill appears center-right, auto-dismisses after ~8s
- **Input:** None — no text field, no conversation visible
- **Behavior:** Suggestion pill can be tapped to escalate to Floating Overlay. Ignoring it returns to idle pulse.

### 2. Floating Overlay

The "quick back-and-forth" mode. A compact, draggable chat bubble on top of everything. You can respond without leaving what you're doing.

- **Visible:** Small floating window (~300px wide, anchored bottom-right by default, draggable). Shows the last few messages + a compact input field. Edge glow remains active behind it.
- **Voice:** Still active (respects Settings toggle)
- **Text:** Messages shown inline in the floating bubble — most recent at bottom, limited scroll history (last ~5 messages visible)
- **Input:** Single-line text field at bottom. Send with Enter.
- **Behavior:**
  - Expand button → escalates to Full Panel
  - Dismiss (click outside, swipe, or Escape) → returns to Ambient
  - Auto-shrinks back to Ambient after ~30s of inactivity (no new messages, no user input)

### 3. Full Panel

The "sit down and talk" mode. Full conversation history, settings, screenshots.

- **Visible:** Right-side panel (28% screen width, full height). Edge glow remains active.
- **Voice:** Active (respects Settings toggle)
- **Text:** Full scrollable conversation history with message bubbles
- **Input:** Multi-line text field with screenshot capture button
- **Settings:** Accessible from this mode — gear icon in header opens settings section
  - Voice: on/off toggle
  - Voice volume slider
  - Voice speed slider
  - Voice selection (available system voices)
  - Model selection (for chat LLM)
- **Behavior:**
  - Close button or hotkey → returns to Ambient
  - Can minimize to Floating Overlay (future)

## Transitions

```
                    tap suggestion pill
        Ambient ─────────────────────────► Floating Overlay
           ▲                                     │
           │ dismiss / inactivity timeout         │ expand button
           │                                     ▼
           ◄──────────── close ──────────── Full Panel
```

- **Ambient → Floating Overlay:** User taps the suggestion pill, or uses a quick-reply hotkey
- **Floating Overlay → Full Panel:** User clicks expand, or uses Cmd+Shift+O
- **Floating Overlay → Ambient:** User dismisses (Escape, click outside), or inactivity timeout (~30s)
- **Full Panel → Ambient:** User closes the panel (close button, Cmd+Shift+O)
- **Any → Ambient:** Escape key always returns to Ambient
- **Cmd+Shift+O:** Cycles Ambient → Full Panel → Ambient (direct toggle for power users)

## UIMode Enum

```swift
enum UIMode: Equatable {
    case ambient
    case floatingOverlay
    case fullPanel
}
```

Lives on `OverlayState` as `@Published var uiMode: UIMode = .ambient`.

## Settings Model

```swift
final class Settings: ObservableObject {
    @Published var voiceEnabled: Bool       // default: true
    @Published var voiceVolume: Float       // 0.0–1.0, default: 0.6
    @Published var voiceRate: Float         // 0.0–1.0, default: 0.5
    @Published var selectedVoiceId: String? // nil = auto-select best
}
```

Persisted via `UserDefaults`. `VoiceService` reads from `Settings` — when `voiceEnabled` is false, `speak()` is a no-op.

## Window Management

| Mode | EdgeGlowWindow | SuggestionWindow | FloatingOverlayWindow | ChatPanel |
|------|---------------|-----------------|----------------------|-----------|
| Ambient | visible | show/hide per suggestion | hidden | hidden |
| Floating Overlay | visible | hidden | visible | hidden |
| Full Panel | visible | hidden | hidden | visible |

Edge glow is always visible. Only one "content" window is shown at a time.

## Voice Integration

Voice is controlled by `Settings`, not by the mode. All three modes call `voiceService.speak()` for proactive suggestions — Settings.voiceEnabled gates whether sound actually plays. The user toggles voice from:

1. Full Panel settings section (primary)
2. Status bar menu (quick toggle)
