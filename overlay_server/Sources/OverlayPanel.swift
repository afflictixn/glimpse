import AppKit
import SwiftUI

final class OverlayPanel: NSPanel {
    private let viewModel: OverlayViewModel
    private let position: ScreenCorner

    enum ScreenCorner: String {
        case topLeft = "top-left"
        case topRight = "top-right"
        case bottomLeft = "bottom-left"
        case bottomRight = "bottom-right"
    }

    init(viewModel: OverlayViewModel, position: ScreenCorner = .bottomRight) {
        self.viewModel = viewModel
        self.position = position

        super.init(
            contentRect: NSRect(x: 0, y: 0, width: 380, height: 200),
            styleMask: [.nonactivatingPanel, .borderless],
            backing: .buffered,
            defer: true
        )

        isOpaque = false
        backgroundColor = .clear
        hasShadow = true
        level = .floating
        collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        isMovableByWindowBackground = true
        hidesOnDeactivate = false
        animationBehavior = .utilityWindow

        let hostView = NSHostingView(rootView: OverlayContentView(viewModel: viewModel))
        contentView = hostView
    }

    func showProposal() {
        reposition()
        orderFrontRegardless()
    }

    func reposition() {
        guard let screen = NSScreen.main else { return }
        let visible = screen.visibleFrame
        let margin: CGFloat = 20

        layoutIfNeeded()
        let size = fittingSize

        let origin: NSPoint
        switch position {
        case .topLeft:
            origin = NSPoint(
                x: visible.minX + margin,
                y: visible.maxY - size.height - margin
            )
        case .topRight:
            origin = NSPoint(
                x: visible.maxX - size.width - margin,
                y: visible.maxY - size.height - margin
            )
        case .bottomLeft:
            origin = NSPoint(
                x: visible.minX + margin,
                y: visible.minY + margin
            )
        case .bottomRight:
            origin = NSPoint(
                x: visible.maxX - size.width - margin,
                y: visible.minY + margin
            )
        }
        setFrameOrigin(origin)
        setContentSize(size)
    }

    private var fittingSize: NSSize {
        guard let hosting = contentView as? NSHostingView<OverlayContentView> else {
            return NSSize(width: 380, height: 200)
        }
        let ideal = hosting.fittingSize
        return NSSize(width: 380, height: max(ideal.height, 120))
    }
}
