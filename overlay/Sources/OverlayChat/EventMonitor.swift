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
/// Uses Carbon RegisterEventHotKey for Cmd+Shift+O — this does NOT require Accessibility permission,
/// so the hotkey works even when the binary changes and macOS revokes Accessibility.
final class EventMonitor {
    var onTrigger: ((CaptureTrigger) -> Void)?
    var onHotkey: (() -> Void)?

    private var globalMonitor: Any?
    private var localMonitor: Any?
    private var workspaceObserver: NSObjectProtocol?
    private var hotKeyRef: EventHotKeyRef?
    private var carbonHandlerRef: EventHandlerRef?

    private let typingPauseDelayMs: Double
    private let idleIntervalMs: Double

    private var lastActivityTime: TimeInterval = 0
    private var lastKeyboardTime: TimeInterval = 0
    private var wasTyping = false
    private var typingPauseTimer: Timer?
    private var idleTimer: Timer?

    private let lock = NSLock()

    /// Global static ref so the C callback can reach us
    private static weak var shared: EventMonitor?

    init(typingPauseDelayMs: Double = 500, idleIntervalMs: Double = 30_000) {
        self.typingPauseDelayMs = typingPauseDelayMs
        self.idleIntervalMs = idleIntervalMs
    }

    func start() {
        let now = ProcessInfo.processInfo.systemUptime
        lastActivityTime = now
        lastKeyboardTime = 0
        EventMonitor.shared = self

        // --- Carbon hotkey: Cmd+Shift+O (no Accessibility permission needed) ---
        registerCarbonHotkey()

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
            self?.handleEvent(event)
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

    // MARK: - Carbon Global Hotkey (works without Accessibility permission)

    private func registerCarbonHotkey() {
        // C callback — must be a plain function, no captures
        let handler: EventHandlerUPP = { _, event, _ -> OSStatus in
            DispatchQueue.main.async {
                print("[zexp] hotkey Cmd+Shift+O detected (Carbon)")
                EventMonitor.shared?.onHotkey?()
            }
            return noErr
        }

        var eventType = EventTypeSpec(
            eventClass: OSType(kEventClassKeyboard),
            eventKind: UInt32(kEventHotKeyPressed)
        )

        InstallEventHandler(
            GetApplicationEventTarget(),
            handler,
            1,
            &eventType,
            nil,
            &carbonHandlerRef
        )

        // Cmd=0x100, Shift=0x200 → cmdKey | shiftKey
        let hotKeyID = EventHotKeyID(signature: OSType(0x5A455850), id: 1) // "ZEXP"
        let status = RegisterEventHotKey(
            UInt32(kVK_ANSI_O),
            UInt32(cmdKey | shiftKey),
            hotKeyID,
            GetApplicationEventTarget(),
            0,
            &hotKeyRef
        )

        if status == noErr {
            print("[zexp] Carbon hotkey registered: Cmd+Shift+O")
        } else {
            print("[zexp] Carbon hotkey registration failed: \(status)")
        }
    }

    func stop() {
        if let ref = hotKeyRef { UnregisterEventHotKey(ref) }
        if let ref = carbonHandlerRef { RemoveEventHandler(ref) }
        if let m = globalMonitor { NSEvent.removeMonitor(m) }
        if let m = localMonitor { NSEvent.removeMonitor(m) }
        if let obs = workspaceObserver {
            NSWorkspace.shared.notificationCenter.removeObserver(obs)
        }
        typingPauseTimer?.invalidate()
        idleTimer?.invalidate()
        hotKeyRef = nil
        carbonHandlerRef = nil
        globalMonitor = nil
        localMonitor = nil
        workspaceObserver = nil
        EventMonitor.shared = nil
    }

    // MARK: - Event Handling

    private func handleEvent(_ event: NSEvent) {
        switch event.type {
        case .leftMouseDown, .rightMouseDown:
            recordActivity()
            onTrigger?(.click)

        case .keyDown:
            recordActivity()
            recordKeyboard()

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
