import SwiftUI

struct SuggestionView: View {
    @ObservedObject var state: OverlayState
    @State private var isVisible: Bool = false
    @State private var isExpanded: Bool = false
    @State private var dismissTask: Task<Void, Never>? = nil

    private let glowColor = Color(hue: 0.6, saturation: 0.5, brightness: 0.9)

    var body: some View {
        Group {
            if let suggestion = state.currentSuggestion {
                VStack(spacing: 6) {
                    // Compact pill — only as wide as the text
                    HStack(spacing: 8) {
                        // Subtle breathing dot
                        Circle()
                            .fill(glowColor)
                            .frame(width: 5, height: 5)
                            .opacity(0.7)

                        Text(suggestion)
                            .font(.system(size: 12, weight: .medium, design: .rounded))
                            .foregroundColor(.white.opacity(0.85))
                            .lineLimit(isExpanded ? 4 : 1)

                        if !isExpanded {
                            Image(systemName: "chevron.down")
                                .font(.system(size: 8, weight: .semibold))
                                .foregroundColor(.white.opacity(0.3))
                        }
                    }
                    .padding(.horizontal, 14)
                    .padding(.vertical, 8)
                    .background(
                        ZStack {
                            Capsule(style: .continuous)
                                .fill(Color.black.opacity(0.6))

                            Capsule(style: .continuous)
                                .fill(glowColor.opacity(0.05))

                            Capsule(style: .continuous)
                                .strokeBorder(glowColor.opacity(0.25), lineWidth: 0.5)
                        }
                    )
                    .shadow(color: glowColor.opacity(0.2), radius: 15, x: 0, y: 2)
                    .onTapGesture {
                        withAnimation(.easeInOut(duration: 0.2)) {
                            isExpanded.toggle()
                        }
                        cancelAutoDismiss()
                    }

                    // Expanded actions
                    if isExpanded {
                        HStack(spacing: 8) {
                            pillButton("dismiss") {
                                withAnimation { state.dismissSuggestion() }
                                isVisible = false
                                isExpanded = false
                            }
                            pillButton("open chat") {
                                state.openFloatingOverlay()
                                isVisible = false
                                isExpanded = false
                            }
                        }
                        .transition(.opacity.combined(with: .move(edge: .top)))
                    }
                }
                // Slide down from top edge
                .opacity(isVisible ? 1 : 0)
                .offset(y: isVisible ? 0 : -30)
                .onAppear {
                    withAnimation(.spring(response: 0.5, dampingFraction: 0.8, blendDuration: 0)) {
                        isVisible = true
                    }
                    scheduleAutoDismiss()
                }
                .onChange(of: state.currentSuggestion) { _ in
                    isExpanded = false
                    isVisible = false
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.05) {
                        withAnimation(.spring(response: 0.5, dampingFraction: 0.8, blendDuration: 0)) {
                            isVisible = true
                        }
                    }
                    scheduleAutoDismiss()
                }
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
        .padding(.top, 8)
    }

    private func pillButton(_ label: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Text(label)
                .font(.system(size: 9, weight: .medium, design: .rounded))
                .foregroundColor(.white.opacity(0.45))
                .padding(.horizontal, 8)
                .padding(.vertical, 3)
                .background(
                    Capsule(style: .continuous)
                        .fill(Color.white.opacity(0.06))
                )
        }
        .buttonStyle(.plain)
    }

    private func scheduleAutoDismiss() {
        dismissTask?.cancel()
        dismissTask = Task {
            try? await Task.sleep(nanoseconds: 8_000_000_000)
            if !isExpanded && !Task.isCancelled {
                withAnimation(.easeIn(duration: 0.5)) {
                    isVisible = false
                }
                try? await Task.sleep(nanoseconds: 500_000_000)
                if !isExpanded && !Task.isCancelled {
                    state.dismissSuggestion()
                }
            }
        }
    }

    private func cancelAutoDismiss() {
        dismissTask?.cancel()
        dismissTask = nil
    }
}
