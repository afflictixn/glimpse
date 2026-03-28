import SwiftUI

struct EdgeGlowView: View {
    @ObservedObject var state: OverlayState

    // Mono blue — same hue, wide brightness range for visible movement
    private static let palette: [Color] = [
        Color(hue: 0.6, saturation: 0.55, brightness: 1.0),
        Color(hue: 0.6, saturation: 0.35, brightness: 0.70),
        Color(hue: 0.6, saturation: 0.50, brightness: 0.92),
        Color(hue: 0.6, saturation: 0.60, brightness: 0.65),
        Color(hue: 0.6, saturation: 0.40, brightness: 0.95),
        Color(hue: 0.6, saturation: 0.50, brightness: 0.78),
    ]

    // 4 glow layers — best depth / light falloff
    private static let lineWidths: [CGFloat] = [2, 5, 9, 14]
    private static let blurRadii: [CGFloat]  = [3, 8, 18, 28]
    private static let animDurations: [Double] = [1.2, 1.6, 2.0, 2.5]

    @State private var stops: [Gradient.Stop] = randomStops()
    @State private var glowOpacity: Double = 0.06

    var body: some View {
        let gradient = AngularGradient(
            gradient: Gradient(stops: stops),
            center: .center
        )

        ZStack {
            ForEach(0..<Self.lineWidths.count, id: \.self) { i in
                RoundedRectangle(cornerRadius: 22, style: .continuous)
                    .stroke(gradient, lineWidth: Self.lineWidths[i])
                    .blur(radius: Self.blurRadii[i])
                    .animation(
                        .easeInOut(duration: Self.animDurations[i]),
                        value: stops
                    )
            }

            // Bright accent rim
            RoundedRectangle(cornerRadius: 22, style: .continuous)
                .stroke(Color.white.opacity(0.12), lineWidth: 1.5)
                .blur(radius: 3)
        }
        .drawingGroup()
        .opacity(glowOpacity)
        .allowsHitTesting(false)
        .task {
            while !Task.isCancelled {
                // Skip updates when invisible — saves GPU in idle
                if state.mode != .idle && state.mode != .hidden {
                    stops = Self.randomStops()
                }
                try? await Task.sleep(nanoseconds: 3_000_000_000)
            }
        }
        .onChange(of: state.mode) { _ in
            withAnimation(.easeInOut(duration: 1.0)) {
                glowOpacity = Self.opacityForMode(state.mode)
            }
        }
    }

    private static func opacityForMode(_ mode: OverlayState.AnimationMode) -> Double {
        switch mode {
        case .idle:       return 0.06
        case .hidden:     return 0.0
        case .suggestion: return 0.7
        case .excited:    return 0.85
        case .important:  return 0.9
        case .warning:    return 1.0
        }
    }

    /// Mirror stops left↔right so the glow is symmetrical.
    private static func randomStops() -> [Gradient.Stop] {
        var stops: [Gradient.Stop] = []
        for color in palette {
            let loc = Double.random(in: 0.02...0.48)
            stops.append(Gradient.Stop(color: color, location: loc))
            stops.append(Gradient.Stop(color: color, location: 1.0 - loc))
        }
        return stops.sorted { $0.location < $1.location }
    }
}
