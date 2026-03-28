import SwiftUI

struct SuggestionView: View {
    @ObservedObject var state: OverlayState
    @State private var isVisible: Bool = false
    @State private var isExpanded: Bool = false
    @State private var dismissTask: Task<Void, Never>? = nil

    var body: some View {
        Group {
            if let suggestion = state.currentSuggestion {
                VStack(alignment: .trailing, spacing: 6) {
                    Text(suggestion)
                        .font(.system(size: 12, weight: .medium, design: .rounded))
                        .foregroundColor(.white.opacity(0.85))
                        .lineLimit(isExpanded ? nil : 1)
                        .padding(.horizontal, 14)
                        .padding(.vertical, 8)
                        .background(
                            RoundedRectangle(cornerRadius: 10)
                                .fill(.ultraThinMaterial)
                                .opacity(0.85)
                        )
                        .onTapGesture {
                            withAnimation(.easeInOut(duration: 0.2)) {
                                isExpanded.toggle()
                            }
                            cancelAutoDismiss()
                        }

                    if isExpanded {
                        HStack(spacing: 12) {
                            Button("dismiss") {
                                withAnimation { state.dismissSuggestion() }
                                isVisible = false
                                isExpanded = false
                            }
                            .font(.system(size: 10))
                            .foregroundColor(.white.opacity(0.4))
                            .buttonStyle(.plain)

                            Button("open chat") {
                                state.showChatPanel = true
                                state.dismissSuggestion()
                                isVisible = false
                                isExpanded = false
                            }
                            .font(.system(size: 10))
                            .foregroundColor(.white.opacity(0.4))
                            .buttonStyle(.plain)
                        }
                    }
                }
                .frame(maxWidth: 300)
                .opacity(isVisible ? 1 : 0)
                .offset(x: isVisible ? 0 : 20)
                .onAppear {
                    withAnimation(.easeOut(duration: 0.4)) {
                        isVisible = true
                    }
                    scheduleAutoDismiss()
                }
                .onChange(of: state.currentSuggestion) { _ in
                    isExpanded = false
                    withAnimation(.easeOut(duration: 0.4)) {
                        isVisible = true
                    }
                    scheduleAutoDismiss()
                }
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
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
