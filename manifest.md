# Overlay — Ambient AI Companion for macOS

## Vision

Your laptop is alive. It sees what you see, learns what you like, and helps before you ask. Not a chatbot — a presence. It communicates through color, glow, and perfectly timed suggestions. It feels like your laptop has taste.

## Core Principles

- **Proactive, not reactive** — it watches your screen and acts when it has something genuinely useful
- **Taste, not tools** — it's not a form filler or autocomplete. It has opinions and notices things you'd miss
- **Ambient, not intrusive** — communicates through the breathing UI first, words second. A soft glow means "I have something" — you choose to look
- **Learns you** — remembers your preferences, what you like, what you skip. Gets better over time without you configuring anything

## UI / UX

### Philosophy

The overlay is not an app. It's a presence. You feel it more than you see it. No panels, no windows, no chat bubbles by default — just breathing edges on the sides of your screen, a voice, and tiny text that appears and fades.

### The Edges

Two thin glowing lines — left and right sides of the screen. That's the whole visual footprint when idle. They breathe: slow, rhythmic pulse in and out. This is the heartbeat of the system.

The edges are the only thing always visible. Everything else appears and disappears.

### Color = Vibe, Animation = Message Type

**Color** is atmospheric. It shifts based on context, time of day, what you're doing, your energy. It's never static. It makes the whole thing feel alive without you thinking about why.

- Morning: warm, bright tones
- Deep work session: cool, muted
- Late night: deep, dim
- Weekend: looser, more saturated
- Adapts to what's on screen — picks up the vibe of whatever you're doing

**Animation** is the signal. You read the rhythm, not the color:

| Animation | Meaning |
|---|---|
| Slow calm pulse | Idle — I'm here, nothing to say |
| Gentle warm shimmer | I have a suggestion, no rush |
| Quick double-pulse | You're going to like this |
| Sustained bright glow | This is important, look |
| Sharp flash + hold | Warning — scam, privacy leak, something's wrong |
| Fade to almost nothing | You're focused, I'm backing off |

### Voice — Primary Output

Voice is how it talks to you. Not text-first with voice as a bonus — voice IS the communication. It speaks its suggestions, observations, and warnings out loud.

- Calm, natural tone. Not robotic, not overly cheerful. Like a friend sitting next to you
- Contextually aware of volume/interruption — whisper-level during a video call, normal when you're browsing alone
- Doesn't repeat itself. Says it once. If you missed it, the text is there

### Text — Subtle Echo

Small text appears when it speaks. Not a chat bubble. More like a subtitle:

- Appears center-right or right-aligned
- Small, clean font — almost HUD-like
- Fades in when it speaks, fades out after a few seconds
- One-liner by default. Tap/hover to expand the full thought
- Stays dismissible — swipe or ignore and it goes away

### Proactive Suggestion Flow

1. Model notices something on screen
2. Edge animation shifts — gentle shimmer or double-pulse depending on importance
3. One-liner text materializes: "cheaper on Amazon" / "this has bad reviews" / "you haven't talked to her in 6 weeks"
4. Voice speaks the suggestion (same content, natural phrasing)
5. If you engage (click the text, respond by voice, or type) → it expands into a fuller response
6. If you ignore → text fades after a few seconds, edges return to idle pulse
7. If you ignore 3+ suggestions in a row → it goes quiet for a while, backs off

### Active Conversation Mode

When you want to talk to it (hotkey, click, or just start speaking):

- The edges brighten and widen slightly — the overlay "wakes up"
- A minimal text area appears along the right edge for input/output
- Conversation is still voice-first, text as backup
- When you're done, it fades back to just the breathing edges
- No permanent chat log visible — but history is accessible if you pull it up

### Interaction Patterns

**Dismiss:** Ignore it. It fades. Or swipe the text away.

**Engage:** Click the one-liner, speak to it, or type. It expands.

**Snooze:** "Not now" — it goes quiet for a set period.

**Teach:** "Don't show me this kind of thing" — it learns and stops suggesting that category.

**The overlay learns from your ignores.** If you never engage with shopping suggestions but always engage with social context ones, it shifts priority over time. No settings page. It just adapts.

### What's NOT in the UI

- No settings panel. It learns from behavior
- No notification badges or counters
- No dock icon. It's an accessory — lives in the status bar
- No visible chat history by default
- No buttons, menus, or toolbars in the idle state
- Nothing that looks like a traditional app

## What It's NOT

- Not a chatbot you have to prompt
- Not a productivity tool with dashboards
- Not Siri/Alexa — no voice, no wake word
- Not a notification machine — it speaks through glow and color first

## Technical Backbone

- Local 16B vision+reasoning model (runs continuously in background)
- Periodic screen capture (~every few seconds when active)
- Preference learning stored locally — never leaves the machine
- macOS floating panel (NSPanel) with Metal-backed animations for the breathing UI
