import SwiftUI

struct FloatingOverlayView: View {
    @ObservedObject var state: OverlayState
    @ObservedObject var viewModel: ChatViewModel
    @State private var inactivityTask: Task<Void, Never>?

    /// Scale with screen: ~300 on 13", ~340 on 14", ~380 on 16"
    private static var responsiveWidth: CGFloat {
        let screenWidth = NSScreen.main?.frame.width ?? 1440
        return min(max(screenWidth * 0.22, 280), 400)
    }

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider().overlay(Color.white.opacity(0.15))
            recentMessages
            Divider().overlay(Color.white.opacity(0.15))
            compactInput
        }
        .frame(width: Self.responsiveWidth)
        .background(Color.black.opacity(0.5))
        .background(VisualEffectBackground(material: .hudWindow, blurRadius: 5, cornerRadius: 14))
        .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .strokeBorder(Color(hue: 0.6, saturation: 0.5, brightness: 0.9).opacity(0.15), lineWidth: 0.5)
        )
        .shadow(color: .black.opacity(0.3), radius: 20, y: 8)
        .preferredColorScheme(.dark)
        .onAppear { resetInactivityTimer() }
        .onChange(of: viewModel.messages.count) { _ in resetInactivityTimer() }
        .onChange(of: viewModel.inputText) { _ in resetInactivityTimer() }
    }

    // MARK: - Header

    private var header: some View {
        HStack(spacing: 8) {
            Text("z exp")
                .font(.system(size: 11, weight: .semibold, design: .monospaced))
                .foregroundColor(.white)

            Spacer()

            Button(action: { state.expandToFullPanel() }) {
                Image(systemName: "arrow.up.left.and.arrow.down.right")
                    .font(.system(size: 11, weight: .medium))
                    .foregroundColor(.white.opacity(0.85))
                    .frame(width: 28, height: 28)
                    .background(Color.white.opacity(0.08))
                    .clipShape(Circle())
            }
            .buttonStyle(.plain)
            .help("Expand to full panel")

            Button(action: { state.dismissToAmbient() }) {
                Image(systemName: "xmark")
                    .font(.system(size: 11, weight: .medium))
                    .foregroundColor(.white.opacity(0.85))
                    .frame(width: 28, height: 28)
                    .background(Color.white.opacity(0.08))
                    .clipShape(Circle())
            }
            .buttonStyle(.plain)
            .help("Dismiss")
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
    }

    // MARK: - Recent Messages (last 5)

    private var recentMessages: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(spacing: 6) {
                    ForEach(recentSlice) { message in
                        compactBubble(message)
                    }

                    if viewModel.isStreaming {
                        ThinkingBubble()
                    }

                    Color.clear.frame(height: 1).id("float-bottom")
                }
                .padding(.horizontal, 10)
                .padding(.vertical, 6)
            }
            .frame(maxHeight: 200)
            .onChange(of: viewModel.messages.count) { _ in
                withAnimation(.easeOut(duration: 0.1)) {
                    proxy.scrollTo("float-bottom", anchor: .bottom)
                }
            }
        }
    }

    private var recentSlice: [ChatMessage] {
        let msgs = viewModel.messages.filter { $0.role != .system }
        return Array(msgs.suffix(5))
    }

    @ViewBuilder
    private func compactBubble(_ message: ChatMessage) -> some View {
        if message.content.isEmpty { EmptyView() }
        else {
        HStack {
            if message.role == .user { Spacer(minLength: 40) }

            Text(message.content)
                .font(.system(size: 11))
                .foregroundColor(.white.opacity(message.role == .user ? 0.9 : 0.75))
                .lineLimit(nil)
                .fixedSize(horizontal: false, vertical: true)
                .padding(.horizontal, 10)
                .padding(.vertical, 6)
                .background(
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .fill(message.role == .user
                            ? Color(hue: 0.6, saturation: 0.5, brightness: 0.9).opacity(0.15)
                            : Color.white.opacity(0.05))
                )

            if message.role != .user { Spacer(minLength: 40) }
        }
        }
    }

    // MARK: - Compact Input

    private var compactInput: some View {
        HStack(spacing: 6) {
            TextField("", text: $viewModel.inputText, prompt: Text("Reply...").foregroundColor(.white.opacity(0.5)))
                .textFieldStyle(.plain)
                .font(.system(size: 12))
                .foregroundColor(.white)
                .onSubmit {
                    viewModel.send()
                    resetInactivityTimer()
                }

            if viewModel.isStreaming {
                Button(action: { viewModel.stopStreaming() }) {
                    Image(systemName: "stop.circle.fill")
                        .font(.system(size: 16))
                        .foregroundColor(.orange)
                }
                .buttonStyle(.plain)
            } else {
                Button(action: {
                    viewModel.send()
                    resetInactivityTimer()
                }) {
                    Image(systemName: "arrow.up.circle.fill")
                        .font(.system(size: 16))
                        .foregroundColor(
                            viewModel.inputText.trimmingCharacters(in: .whitespaces).isEmpty
                                ? .white.opacity(0.2)
                                : .white
                        )
                }
                .buttonStyle(.plain)
                .disabled(viewModel.inputText.trimmingCharacters(in: .whitespaces).isEmpty)
            }
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 8)
    }

    // MARK: - Inactivity Auto-dismiss

    private func resetInactivityTimer() {
        inactivityTask?.cancel()
        inactivityTask = Task {
            try? await Task.sleep(nanoseconds: 120_000_000_000) // 2min
            guard !Task.isCancelled else { return }
            await MainActor.run {
                // Only auto-dismiss if not actively streaming
                if !viewModel.isStreaming {
                    state.dismissToAmbient()
                }
            }
        }
    }
}
