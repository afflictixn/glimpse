import AppKit
import SwiftUI
import Carbon.HIToolbox

/// Custom NSPanel subclass that can become key window even when borderless,
/// allowing text fields inside to receive keyboard input.
final class KeyablePanel: NSPanel {
    override var canBecomeKey: Bool { true }
    override var canBecomeMain: Bool { false }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    private var panel: KeyablePanel!
    private var statusItem: NSStatusItem!
    private var viewModel: ChatViewModel!

    // MARK: - Lifecycle

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.accessory)
        viewModel = ChatViewModel()
        setupPanel()
        setupStatusBar()
        setupGlobalHotkey()
    }

    // MARK: - Floating Panel

    private func setupPanel() {
        guard let screen = NSScreen.main else { return }
        let visibleFrame = screen.visibleFrame
        let panelWidth = visibleFrame.width * 0.30
        let panelHeight = visibleFrame.height

        let origin = CGPoint(
            x: visibleFrame.maxX - panelWidth,
            y: visibleFrame.minY
        )

        let frame = NSRect(origin: origin, size: CGSize(width: panelWidth, height: panelHeight))

        panel = KeyablePanel(
            contentRect: frame,
            styleMask: [.borderless, .nonactivatingPanel, .resizable],
            backing: .buffered,
            defer: false
        )

        // Critical: these prevent the "compress to desktop" behavior
        panel.level = .floating
        panel.collectionBehavior = [.canJoinAllSpaces, .stationary, .fullScreenAuxiliary]

        panel.isOpaque = false
        panel.backgroundColor = .clear
        panel.hasShadow = true
        panel.isMovableByWindowBackground = true
        // Allow panel to become key when user clicks text field
        panel.becomesKeyOnlyIfNeeded = true
        panel.hidesOnDeactivate = false
        panel.animationBehavior = .utilityWindow

        // SwiftUI content
        let chatView = ChatView(viewModel: viewModel)
        let hostingView = NSHostingView(rootView: chatView)
        hostingView.wantsLayer = true
        hostingView.layer?.cornerRadius = 12
        hostingView.layer?.masksToBounds = true
        panel.contentView = hostingView

        panel.orderFrontRegardless()
    }

    // MARK: - Status Bar

    private func setupStatusBar() {
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.squareLength)

        if let button = statusItem.button {
            button.image = NSImage(systemSymbolName: "bubble.left.fill", accessibilityDescription: "Overlay Chat")
        }

        let menu = NSMenu()
        menu.addItem(NSMenuItem(title: "Toggle Overlay  (Cmd+Shift+O)", action: #selector(togglePanel), keyEquivalent: ""))
        menu.addItem(NSMenuItem.separator())
        menu.addItem(NSMenuItem(title: "Quit", action: #selector(quitApp), keyEquivalent: "q"))

        statusItem.menu = menu
    }

    // MARK: - Global Hotkey (Cmd+Shift+O)

    private func setupGlobalHotkey() {
        NSEvent.addGlobalMonitorForEvents(matching: .keyDown) { [weak self] event in
            self?.handleHotkeyEvent(event)
        }

        NSEvent.addLocalMonitorForEvents(matching: .keyDown) { [weak self] event in
            if self?.handleHotkeyEvent(event) == true {
                return nil
            }
            return event
        }
    }

    @discardableResult
    private func handleHotkeyEvent(_ event: NSEvent) -> Bool {
        let required: NSEvent.ModifierFlags = [.command, .shift]
        guard event.modifierFlags.intersection(.deviceIndependentFlagsMask).contains(required),
              event.keyCode == 0x1F // "O"
        else { return false }

        DispatchQueue.main.async { [weak self] in
            self?.togglePanel()
        }
        return true
    }

    // MARK: - Actions

    @objc private func togglePanel() {
        if panel.isVisible {
            panel.orderOut(nil)
        } else {
            panel.orderFrontRegardless()
        }
    }

    @objc private func quitApp() {
        NSApp.terminate(nil)
    }
}
