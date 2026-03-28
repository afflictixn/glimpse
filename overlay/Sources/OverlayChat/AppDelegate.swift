import AppKit
import SwiftUI
import Combine

/// NSPanel that can become key (for text input in chat)
final class KeyablePanel: NSPanel {
    override var canBecomeKey: Bool { true }
    override var canBecomeMain: Bool { false }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    private var edgeGlowWindow: NSPanel!
    private var suggestionWindow: NSPanel!
    private var floatingOverlayWindow: KeyablePanel!
    private var chatPanel: KeyablePanel!
    private var statusItem: NSStatusItem!

    private var overlayState: OverlayState!
    private var chatViewModel: ChatViewModel!
    private var wsServer: WebSocketServer!
    private var calendarServer: CalendarServer!
    private var eventMonitor: EventMonitor!
    private var captureManager: CaptureManager!

    private var cancellables = Set<AnyCancellable>()

    // MARK: - Lifecycle

    func applicationDidFinishLaunching(_ notification: Notification) {
        print("[zexp] launching...")
        NSApp.setActivationPolicy(.regular)

        // Prompt for accessibility permissions (needed for global hotkeys)
        requestAccessibilityIfNeeded()

        overlayState = OverlayState()
        chatViewModel = ChatViewModel()

        setupEdgeGlowWindow()
        setupSuggestionWindow()
        setupFloatingOverlayWindow()
        setupChatPanel()
        setupStatusBar()
        setupCapturePipeline()
        setupWebSocket()
        setupCalendarServer()
        bindState()

        // Startup demo
        DispatchQueue.main.asyncAfter(deadline: .now() + 2) { [weak self] in
            self?.overlayState.handleProposal(text: "z is alive", importance: .medium)
        }
    }

    // MARK: - Accessibility Permissions

    private func requestAccessibilityIfNeeded() {
        let options = [kAXTrustedCheckOptionPrompt.takeRetainedValue(): true] as CFDictionary
        let trusted = AXIsProcessTrustedWithOptions(options)
        if trusted {
            print("[zexp] accessibility: granted")
        } else {
            print("[zexp] accessibility: not granted — macOS should be prompting now")
        }
    }

    // MARK: - Edge Glow Window (always visible, click-through)

    private func setupEdgeGlowWindow() {
        guard let screen = NSScreen.main else { return }

        edgeGlowWindow = NSPanel(
            contentRect: screen.frame,
            styleMask: [.borderless, .nonactivatingPanel],
            backing: .buffered,
            defer: false
        )

        edgeGlowWindow.level = .floating
        edgeGlowWindow.collectionBehavior = [.canJoinAllSpaces, .stationary, .fullScreenAuxiliary]
        edgeGlowWindow.isOpaque = false
        edgeGlowWindow.backgroundColor = .clear
        edgeGlowWindow.hasShadow = false
        edgeGlowWindow.ignoresMouseEvents = true
        edgeGlowWindow.hidesOnDeactivate = false
        edgeGlowWindow.animationBehavior = .none

        let glowView = EdgeGlowView(state: overlayState)
        let hostingView = NSHostingView(rootView: glowView)
        edgeGlowWindow.contentView = hostingView
        edgeGlowWindow.orderFrontRegardless()
    }

    // MARK: - Suggestion Window (small, interactive, shows one-liner text)

    private func setupSuggestionWindow() {
        guard let screen = NSScreen.main else { return }
        let visibleFrame = screen.visibleFrame

        // Top center — Dynamic Island style
        let width: CGFloat = 500
        let height: CGFloat = 70
        let origin = CGPoint(
            x: visibleFrame.midX - width / 2,
            y: visibleFrame.maxY - height
        )

        suggestionWindow = NSPanel(
            contentRect: NSRect(origin: origin, size: CGSize(width: width, height: height)),
            styleMask: [.borderless, .nonactivatingPanel],
            backing: .buffered,
            defer: false
        )

        suggestionWindow.level = .floating
        suggestionWindow.collectionBehavior = [.canJoinAllSpaces, .stationary, .fullScreenAuxiliary]
        suggestionWindow.isOpaque = false
        suggestionWindow.backgroundColor = .clear
        suggestionWindow.hasShadow = false
        suggestionWindow.hidesOnDeactivate = false
        suggestionWindow.animationBehavior = .none

        let suggestionView = SuggestionView(state: overlayState)
        let hostingView = NSHostingView(rootView: suggestionView)
        suggestionWindow.contentView = hostingView
        suggestionWindow.orderFrontRegardless()
    }

    // MARK: - Floating Overlay Window (hidden by default)

    private func setupFloatingOverlayWindow() {
        guard let screen = NSScreen.main else { return }
        let visibleFrame = screen.visibleFrame

        let width = min(max(visibleFrame.width * 0.22, 280), 400)
        let height = min(max(visibleFrame.height * 0.3, 320), 420)
        let origin = CGPoint(
            x: visibleFrame.maxX - width - 40,
            y: visibleFrame.minY + 60
        )

        floatingOverlayWindow = KeyablePanel(
            contentRect: NSRect(origin: origin, size: CGSize(width: width, height: height)),
            styleMask: [.borderless, .nonactivatingPanel],
            backing: .buffered,
            defer: false
        )

        floatingOverlayWindow.level = .floating
        floatingOverlayWindow.collectionBehavior = [.canJoinAllSpaces, .stationary, .fullScreenAuxiliary]
        floatingOverlayWindow.isOpaque = false
        floatingOverlayWindow.backgroundColor = .clear
        floatingOverlayWindow.hasShadow = true
        floatingOverlayWindow.isMovableByWindowBackground = true
        floatingOverlayWindow.becomesKeyOnlyIfNeeded = true
        floatingOverlayWindow.hidesOnDeactivate = false
        floatingOverlayWindow.animationBehavior = .utilityWindow

        let floatingView = FloatingOverlayView(state: overlayState, viewModel: chatViewModel, settings: overlayState.settings)
        let hostingView = NSHostingView(rootView: floatingView)
        hostingView.wantsLayer = true
        hostingView.layer?.cornerRadius = 14
        hostingView.layer?.cornerCurve = CALayerCornerCurve.continuous
        hostingView.layer?.masksToBounds = true
        floatingOverlayWindow.contentView = hostingView
        floatingOverlayWindow.orderOut(nil)
    }

    // MARK: - Chat Panel (hidden by default)

    private func setupChatPanel() {
        guard let screen = NSScreen.main else { return }
        let visibleFrame = screen.visibleFrame
        let panelWidth = visibleFrame.width * 0.28
        let panelHeight = visibleFrame.height

        let origin = CGPoint(
            x: visibleFrame.maxX - panelWidth,
            y: visibleFrame.minY
        )

        let frame = NSRect(origin: origin, size: CGSize(width: panelWidth, height: panelHeight))

        chatPanel = KeyablePanel(
            contentRect: frame,
            styleMask: [.borderless, .nonactivatingPanel, .resizable],
            backing: .buffered,
            defer: false
        )

        chatPanel.level = .floating
        chatPanel.collectionBehavior = [.canJoinAllSpaces, .stationary, .fullScreenAuxiliary]
        chatPanel.isOpaque = false
        chatPanel.backgroundColor = .clear
        chatPanel.hasShadow = true
        chatPanel.isMovableByWindowBackground = true
        chatPanel.becomesKeyOnlyIfNeeded = true
        chatPanel.hidesOnDeactivate = false
        chatPanel.animationBehavior = .utilityWindow

        let chatView = ChatView(viewModel: chatViewModel, state: overlayState, settings: overlayState.settings)
        let hostingView = NSHostingView(rootView: chatView)
        hostingView.wantsLayer = true
        hostingView.layer?.cornerRadius = 12
        hostingView.layer?.masksToBounds = true
        chatPanel.contentView = hostingView

        // Hidden by default — edges only
        chatPanel.orderOut(nil)
    }

    // MARK: - State Binding

    private func bindState() {
        overlayState.$uiMode
            .removeDuplicates()
            .sink { [weak self] mode in
                guard let self else { return }
                self.transitionTo(mode)
            }
            .store(in: &cancellables)
    }

    private func transitionTo(_ mode: OverlayState.UIMode) {
        switch mode {
        case .ambient:
            fadeOut(floatingOverlayWindow)
            fadeOut(chatPanel)
            // Suggestion window is managed by OverlayState.currentSuggestion — always ready

        case .floatingOverlay:
            fadeOut(chatPanel)
            fadeIn(floatingOverlayWindow)

        case .fullPanel:
            fadeOut(floatingOverlayWindow)
            fadeIn(chatPanel)
        }
    }

    private func fadeIn(_ panel: NSPanel) {
        guard panel.alphaValue < 1 || !panel.isVisible else { return }
        panel.alphaValue = 0
        panel.orderFrontRegardless()
        NSAnimationContext.runAnimationGroup { ctx in
            ctx.duration = 0.25
            ctx.timingFunction = CAMediaTimingFunction(name: .easeOut)
            panel.animator().alphaValue = 1
        }
    }

    private func fadeOut(_ panel: NSPanel) {
        guard panel.isVisible else { return }
        NSAnimationContext.runAnimationGroup({ ctx in
            ctx.duration = 0.2
            ctx.timingFunction = CAMediaTimingFunction(name: .easeIn)
            panel.animator().alphaValue = 0
        }, completionHandler: {
            panel.orderOut(nil)
        })
    }

    // MARK: - WebSocket Server (receives from Python backend)

    private func setupWebSocket() {
        wsServer = WebSocketServer(port: 9321)
        wsServer.onMessage = { [weak self] message in
            Task { @MainActor in
                guard let self else { return }
                // Feed to chat view model (existing behavior)
                self.chatViewModel.handleInboundMessage(message)

                // Also feed to overlay state for edge glow + voice
                switch message {
                case .showProposal(let text, _):
                    self.overlayState.handleProposal(text: text, importance: .medium)
                case .showConversation, .appendConversation:
                    break // these go straight to chat
                case .hide:
                    self.overlayState.dismissSuggestion()
                case .setAssistantLabel:
                    break
                }
            }
        }
        wsServer.start()
    }

    // MARK: - Status Bar

    private func setupStatusBar() {
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.squareLength)

        if let button = statusItem.button {
            button.image = NSImage(systemSymbolName: "sparkles", accessibilityDescription: "z exp")
        }

        let menu = NSMenu()
        menu.addItem(NSMenuItem(title: "Toggle Chat  (Cmd+Shift+O)", action: #selector(toggleChat), keyEquivalent: ""))

        let voiceItem = NSMenuItem(title: "Voice", action: #selector(toggleVoice), keyEquivalent: "")
        voiceItem.state = overlayState.settings.voiceEnabled ? .on : .off
        menu.addItem(voiceItem)

        menu.addItem(NSMenuItem.separator())

        // Demo menu for testing animations
        let demoMenu = NSMenu()
        let demoItem = NSMenuItem(title: "Demo", action: nil, keyEquivalent: "")
        demoItem.submenu = demoMenu
        demoMenu.addItem(NSMenuItem(title: "Suggestion", action: #selector(demoSuggestion), keyEquivalent: ""))
        demoMenu.addItem(NSMenuItem(title: "Warning", action: #selector(demoWarning), keyEquivalent: ""))
        demoMenu.addItem(NSMenuItem(title: "Excited", action: #selector(demoExcited), keyEquivalent: ""))
        demoMenu.addItem(NSMenuItem(title: "Reset", action: #selector(demoReset), keyEquivalent: ""))
        menu.addItem(demoItem)

        menu.addItem(NSMenuItem.separator())
        menu.addItem(NSMenuItem(title: "Quit", action: #selector(quitApp), keyEquivalent: "q"))

        statusItem.menu = menu
    }

    // MARK: - Capture Pipeline (replaces Python's EventTap + CaptureLoop)

    private func setupCapturePipeline() {
        captureManager = CaptureManager()
        eventMonitor = EventMonitor()

        // Wire triggers → capture manager
        eventMonitor.onTrigger = { [weak self] trigger in
            self?.captureManager.handleTrigger(trigger)
        }

        // Wire hotkey (Cmd+Shift+O) → overlay toggle
        eventMonitor.onHotkey = { [weak self] in
            self?.overlayState.toggleChat()
        }

        eventMonitor.start()
        print("[zexp] Capture pipeline started (EventMonitor → CaptureManager → Python ingest)")
    }

    // MARK: - Actions

    @objc private func toggleChat() {
        overlayState.toggleChat()
    }

    @objc private func toggleVoice() {
        overlayState.settings.voiceEnabled.toggle()
        // Update menu item state
        if let menu = statusItem.menu,
           let voiceItem = menu.items.first(where: { $0.title == "Voice" }) {
            voiceItem.state = overlayState.settings.voiceEnabled ? .on : .off
        }
    }

    @objc private func demoSuggestion() {
        overlayState.handleProposal(text: "This product is $14 cheaper on Amazon", importance: .medium)
    }

    @objc private func demoWarning() {
        overlayState.handleWarning(text: "This site looks like your bank but the URL is off by one character")
    }

    @objc private func demoExcited() {
        overlayState.handleProposal(text: "That Airbnb you liked in Lisbon dropped 30%", importance: .high)
        overlayState.mode = .excited
    }

    @objc private func demoReset() {
        overlayState.dismissSuggestion()
    }

    // MARK: - Calendar Server

    private func setupCalendarServer() {
        calendarServer = CalendarServer()
        calendarServer.start()
    }

    @objc private func quitApp() {
        eventMonitor.stop()
        wsServer.stop()
        calendarServer.stop()
        NSApp.terminate(nil)
    }
}
