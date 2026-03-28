import SwiftUI
import AppKit

// MARK: - Custom blur with controllable radius

final class CustomBlurEffectView: NSVisualEffectView {
    var customBlurRadius: CGFloat = 8.0 {
        didSet { applyBlurRadius() }
    }

    override func updateLayer() {
        super.updateLayer()
        // Re-apply after macOS resets internal layers
        DispatchQueue.main.async { [weak self] in
            self?.applyBlurRadius()
        }
    }

    private func applyBlurRadius() {
        guard let backdrop = findBackdropLayer(in: self.layer) else { return }
        backdrop.setValue(customBlurRadius, forKeyPath: "filters.gaussianBlur.inputRadius")
    }

    private func findBackdropLayer(in layer: CALayer?) -> CALayer? {
        guard let layer = layer else { return nil }
        if String(describing: type(of: layer)) == "CABackdropLayer" {
            return layer
        }
        for sublayer in layer.sublayers ?? [] {
            if let found = findBackdropLayer(in: sublayer) {
                return found
            }
        }
        return nil
    }
}

// MARK: - SwiftUI wrapper

struct VisualEffectBackground: NSViewRepresentable {
    let material: NSVisualEffectView.Material
    let blendingMode: NSVisualEffectView.BlendingMode
    let blurRadius: CGFloat?
    let cornerRadius: CGFloat

    init(material: NSVisualEffectView.Material = .hudWindow,
         blendingMode: NSVisualEffectView.BlendingMode = .behindWindow,
         blurRadius: CGFloat? = nil,
         cornerRadius: CGFloat = 0) {
        self.material = material
        self.blendingMode = blendingMode
        self.blurRadius = blurRadius
        self.cornerRadius = cornerRadius
    }

    func makeNSView(context: Context) -> NSVisualEffectView {
        let view: NSVisualEffectView
        if let radius = blurRadius {
            let custom = CustomBlurEffectView()
            custom.customBlurRadius = radius
            view = custom
        } else {
            view = NSVisualEffectView()
        }
        view.material = material
        view.blendingMode = blendingMode
        view.state = .active
        view.isEmphasized = true
        view.appearance = NSAppearance(named: .darkAqua)
        if cornerRadius > 0 {
            view.wantsLayer = true
            view.layer?.cornerRadius = cornerRadius
            view.layer?.cornerCurve = .continuous
            view.layer?.masksToBounds = true
        }
        return view
    }

    func updateNSView(_ nsView: NSVisualEffectView, context: Context) {
        nsView.material = material
        nsView.blendingMode = blendingMode
        if let radius = blurRadius, let custom = nsView as? CustomBlurEffectView {
            custom.customBlurRadius = radius
        }
    }
}
