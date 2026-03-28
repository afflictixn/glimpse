import SwiftUI

struct EdgeGlowView: View {
    @ObservedObject var state: OverlayState
    let glowWidth: CGFloat = 50

    var body: some View {
        TimelineView(.animation) { timeline in
            let time = timeline.date.timeIntervalSinceReferenceDate
            let intensity = state.intensity(at: time)

            GeometryReader { geo in
                ZStack {
                    // Left edge
                    HStack(spacing: 0) {
                        edgeGlow(intensity: intensity, leading: true)
                            .frame(width: glowWidth)
                        Spacer()
                    }

                    // Right edge
                    HStack(spacing: 0) {
                        Spacer()
                        edgeGlow(intensity: intensity, leading: false)
                            .frame(width: glowWidth)
                    }
                }
                .frame(width: geo.size.width, height: geo.size.height)
            }
        }
        .allowsHitTesting(false)
    }

    @ViewBuilder
    private func edgeGlow(intensity: Double, leading: Bool) -> some View {
        let startPoint: UnitPoint = leading ? .leading : .trailing
        let endPoint: UnitPoint = leading ? .trailing : .leading

        ZStack {
            // Outer soft glow
            LinearGradient(
                colors: [
                    state.accentColor.opacity(intensity * 0.4),
                    state.accentColor.opacity(intensity * 0.1),
                    Color.clear
                ],
                startPoint: startPoint,
                endPoint: endPoint
            )
            .blur(radius: 12)

            // Core glow — bright line at the edge
            LinearGradient(
                colors: [
                    state.accentColor.opacity(intensity * 0.9),
                    state.accentColor.opacity(intensity * 0.3),
                    Color.clear
                ],
                startPoint: startPoint,
                endPoint: endPoint
            )
            .frame(width: 20)
            .blur(radius: 4)
            .frame(maxWidth: .infinity, alignment: leading ? .leading : .trailing)
        }
    }
}
