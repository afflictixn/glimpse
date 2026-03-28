import AppKit
import Carbon.HIToolbox

/// Trigger types matching Python's CaptureTrigger enum values.
enum CaptureTrigger: String {
    case appSwitch = "app_switch"
    case click = "click"
    case typingPause = "typing_pause"
    case clipboard = "clipboard"
    case idle = "idle"
    case manual = "manual"
}

/// Monitors keyboard, mouse, and app-switch events. Detects typing pauses, idle, and clipboard shortcuts.
/// Replaces Python's EventTap + ActivityFeed.
final class EventMonitor {
    var onTrigger: ((CaptureTrigger) -> Void)?
    var onHotkey: (() -> Void)?

    private var globalMonitor: Any?
    private var localMonitor: Any?
    private var workspaceObserver: NSObjectProtocol?

    private let typingPauseDelayMs: Double
    private let idleIntervalMs: Double

    private var lastActivityTime: TimeInterval = 0
    private var lastKeyboardTime: TimeInterval = 0
    private var wasTyping = false
    private var typingPauseTimer: Timer?
    private var idleTimer: Timer?

    private let lock = NSLock()

    init(typingPauseDelayMs: Double = 500, idleIntervalMs: Double = 30_000) {
        self.typingPauseDelayMs = typingPauseDelayMs
        self.idleIntervalMs = idleIntervalMs
    }

    func start() {
        let now = ProcessInfo.processInfo.systemUptime
        lastActivityTime = now
        lastKeyboardTime = 0

        // Global monitor for events when our app is NOT focused
        globalMonitor = NSEvent.addGlobalMonitorForEvents(
            matching: [.keyDown, .keyUp, .leftMouseDown, .rightMouseDown, .mouseMoved, .scrollWheel]
        ) { [weak self] event in
            self?.handleEvent(event)
        }

        // Local monitor for events when our app IS focused
        localMonitor = NSEvent.addLocalMonitorForEvents(
            matching: [.keyDown, .keyUp, .leftMouseDown, .rightMouseDown, .mouseMoved, .scrollWheel]
        ) { [weak self] event in
            // Return nil to consume hotkey events
            if self?.handleEvent(event) == true {
                return nil
            }
            return event
        }

        // App switch detection
        let ws = NSWorkspace.shared
        workspaceObserver = ws.notificationCenter.addObserver(
            forName: NSWorkspace.didActivateApplicationNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            self?.recordActivity()
            self?.onTrigger?(.appSwitch)
        }

        // Idle timer — fires periodically to check for idle
        startIdleTimer()

        print("[zexp] EventMonitor started")
    }

    func stop() {
        if let m = globalMonitor { NSEvent.removeMonitor(m) }
        if let m = localMonitor { NSEvent.removeMonitor(m) }
        if let obs = workspaceObserver {
            NSWorkspace.shared.notificationCenter.removeObserver(obs)
        }
        typingPauseTimer?.invalidate()
        idleTimer?.invalidate()
        globalMonitor = nil
        localMonitor = nil
        workspaceObserver = nil
    }

    // MARK: - Event Handling

    /// Returns true if the event was a hotkey (should be consumed by local monitor).
    @discardableResult
    private func handleEvent(_ event: NSEvent) -> Bool {
        switch event.type {
        case .leftMouseDown, .rightMouseDown:
            recordActivity()
            onTrigger?(.click)

        case .keyDown:
            recordActivity()
            recordKeyboard()

            // Check for Cmd+Shift+O hotkey
            let required: NSEvent.ModifierFlags = [.command, .shift]
            if event.modifierFlags.intersection(.deviceIndependentFlagsMask).contains(required),
               event.keyCode == 0x1F {
                print("[zexp] hotkey Cmd+Shift+O detected")
                DispatchQueue.main.async { [weak self] in
                    self?.onHotkey?()
                }
                return true
            }

            // Check for clipboard shortcuts (Cmd+C/X/V)
            if event.modifierFlags.contains(.command) {
                let keyCode = event.keyCode
                // C=8, X=7, V=9
                if keyCode == 8 || keyCode == 7 || keyCode == 9 {
                    onTrigger?(.clipboard)
                }
            }

        case .keyUp:
            recordActivity()

        case .scrollWheel, .mouseMoved:
            recordActivity()

        default:
            break
        }
        return false
    }

    private func recordActivity() {
        lock.lock()
        lastActivityTime = ProcessInfo.processInfo.systemUptime
        lock.unlock()
        resetIdleTimer()
    }

    private func recordKeyboard() {
        lock.lock()
        lastKeyboardTime = ProcessInfo.processInfo.systemUptime
        wasTyping = true
        lock.unlock()
        resetTypingPauseTimer()
    }

    // MARK: - Typing Pause Detection

    private func resetTypingPauseTimer() {
        typingPauseTimer?.invalidate()
        let delay = typingPauseDelayMs / 1000.0
        typingPauseTimer = Timer.scheduledTimer(withTimeInterval: delay, repeats: false) { [weak self] _ in
            self?.checkTypingPause()
        }
    }

    private func checkTypingPause() {
        lock.lock()
        let typing = wasTyping
        wasTyping = false
        lock.unlock()

        if typing {
            onTrigger?(.typingPause)
        }
    }

    // MARK: - Idle Detection

    private func startIdleTimer() {
        let interval = idleIntervalMs / 1000.0
        idleTimer = Timer.scheduledTimer(withTimeInterval: interval, repeats: true) { [weak self] _ in
            self?.checkIdle()
        }
    }

    private func resetIdleTimer() {
        idleTimer?.invalidate()
        startIdleTimer()
    }

    private func checkIdle() {
        lock.lock()
        let elapsed = (ProcessInfo.processInfo.systemUptime - lastActivityTime) * 1000
        lock.unlock()

        if elapsed >= idleIntervalMs {
            onTrigger?(.idle)
        }
    }
}
