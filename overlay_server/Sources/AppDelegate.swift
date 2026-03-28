import AppKit
import SwiftUI

final class AppDelegate: NSObject, NSApplicationDelegate {
    private var statusItem: NSStatusItem!
    private var panel: OverlayPanel!
    private var server: WebSocketServer!
    private let viewModel = OverlayViewModel()
    private var isPaused = false

    private var port: UInt16 {
        if let arg = ProcessInfo.processInfo.arguments.dropFirst().first,
           let p = UInt16(arg) {
            return p
        }
        return 9321
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.accessory)

        panel = OverlayPanel(viewModel: viewModel)

        viewModel.onAction = { [weak self] message in
            self?.handleOutbound(message)
        }

        setupStatusItem()
        startServer()

        print("[app] overlay server ready on port \(port)")
    }

    // MARK: - Status Bar

    private func setupStatusItem() {
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.squareLength)

        if let button = statusItem.button {
            let image = makeEyeIcon()
            image.isTemplate = true
            button.image = image
            button.toolTip = "Screen Agent"
        }

        let menu = NSMenu()

        let pauseItem = NSMenuItem(title: "Pause", action: #selector(togglePause(_:)), keyEquivalent: "")
        pauseItem.target = self
        menu.addItem(pauseItem)

        let showItem = NSMenuItem(title: "Show / Hide", action: #selector(togglePanel(_:)), keyEquivalent: "")
        showItem.target = self
        menu.addItem(showItem)

        menu.addItem(.separator())

        let quitItem = NSMenuItem(title: "Quit", action: #selector(quit(_:)), keyEquivalent: "q")
        quitItem.target = self
        menu.addItem(quitItem)

        statusItem.menu = menu
    }

    @objc private func togglePause(_ sender: NSMenuItem) {
        isPaused.toggle()
        sender.title = isPaused ? "Resume" : "Pause"
        viewModel.sendPause(isPaused)
        server.broadcast(.pauseToggle(paused: isPaused))
    }

    @objc private func togglePanel(_ sender: NSMenuItem) {
        if panel.isVisible {
            panel.orderOut(nil)
        } else {
            panel.showProposal()
        }
    }

    @objc private func quit(_ sender: NSMenuItem) {
        server.stop()
        NSApp.terminate(nil)
    }

    // MARK: - WebSocket

    private func startServer() {
        server = WebSocketServer(port: port)
        server.onMessage = { [weak self] message in
            self?.handleInbound(message)
        }
        server.start()
    }

    private func handleInbound(_ message: InboundMessage) {
        switch message {
        case .showProposal(let text, _):
            viewModel.setProposal(text)
            panel.showProposal()

        case .showConversation(let text):
            viewModel.showConversationWith(text)
            panel.reposition()

        case .appendConversation(let role, let text):
            viewModel.appendConversation(role: role, text: text)

        case .setAssistantLabel(let label):
            viewModel.assistantLabel = label

        case .hide:
            viewModel.hide()
            panel.orderOut(nil)
        }
    }

    private func handleOutbound(_ message: OutboundMessage) {
        server.broadcast(message)
    }

    // MARK: - Icon

    private func makeEyeIcon() -> NSImage {
        let size = NSSize(width: 18, height: 18)
        let image = NSImage(size: size, flipped: false) { rect in
            let eyePath = NSBezierPath(ovalIn: NSRect(x: 2, y: 4, width: 14, height: 10))
            NSColor.controlTextColor.setFill()
            eyePath.fill()

            let irisBg = NSBezierPath(ovalIn: NSRect(x: 5.5, y: 5.5, width: 7, height: 7))
            NSColor.controlBackgroundColor.setFill()
            irisBg.fill()

            let pupil = NSBezierPath(ovalIn: NSRect(x: 7, y: 7, width: 4, height: 4))
            NSColor.controlTextColor.setFill()
            pupil.fill()

            return true
        }
        return image
    }
}
