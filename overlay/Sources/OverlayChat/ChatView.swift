import SwiftUI

struct ChatView: View {
    @ObservedObject var viewModel: ChatViewModel
    @ObservedObject var settings: Settings
    @State private var showSettings: Bool = false

    var body: some View {
        VStack(spacing: 0) {
            headerBar
            Divider().overlay(Color.white.opacity(0.15))

            if showSettings {
                SettingsView(settings: settings)
            } else {
                // Proactive suggestion banner
                if let suggestion = viewModel.currentSuggestion {
                    suggestionBanner(suggestion)
                    Divider().overlay(Color.white.opacity(0.15))
                }

                messagesArea
                Divider().overlay(Color.white.opacity(0.15))
                inputBar
            }
        }
        .background(
            ZStack {
                Color.black.opacity(0.6)
                Color(hue: 0.6, saturation: 0.5, brightness: 0.9).opacity(0.05)
            }
        )
    }

    // MARK: - Header

    private var headerBar: some View {
        HStack(spacing: 10) {
            Circle()
                .fill(statusColor)
                .frame(width: 7, height: 7)

            Text("glimpse")
                .font(.system(size: 13, weight: .semibold, design: .monospaced))
                .foregroundColor(.white)

            Spacer()

            Button(action: { withAnimation(.easeInOut(duration: 0.15)) { showSettings.toggle() } }) {
                Image(systemName: showSettings ? "chevron.left" : "gearshape")
                    .font(.system(size: 12))
                    .foregroundColor(.white.opacity(0.6))
            }
            .buttonStyle(.plain)
            .help(showSettings ? "Back to chat" : "Settings")

            if !showSettings {
                Button(action: { viewModel.clearChat() }) {
                    Image(systemName: "trash")
                        .font(.system(size: 12))
                        .foregroundColor(.white.opacity(0.6))
                    }
                    .buttonStyle(.plain)
                    .help("Clear chat")
            }

            Button(action: { NSApp.keyWindow?.orderOut(nil) }) {
                Image(systemName: "xmark")
                    .font(.system(size: 12))
                    .foregroundColor(.white.opacity(0.6))
            }
            .buttonStyle(.plain)
            .help("Close (Cmd+Shift+O to reopen)")
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
    }

    private var statusColor: Color {
        switch viewModel.connectionStatus {
        case .connected: return .green
        case .connecting: return .yellow
        case .disconnected: return .red
        }
    }

    // MARK: - Suggestion Banner

    private func suggestionBanner(_ suggestion: ProactiveSuggestion) -> some View {
        HStack(spacing: 8) {
            Circle()
                .fill(Color.blue.opacity(0.6))
                .frame(width: 6, height: 6)

            Text(suggestion.text)
                .font(.system(size: 12))
                .foregroundColor(.white.opacity(0.85))
                .lineLimit(2)

            Spacer()

            Button(action: { viewModel.expandSuggestion() }) {
                Image(systemName: "arrow.up.left.and.arrow.down.right")
                    .font(.system(size: 10))
                    .foregroundColor(.white.opacity(0.5))
            }
            .buttonStyle(.plain)
            .help("Expand")

            Button(action: { viewModel.dismissSuggestion() }) {
                Image(systemName: "xmark")
                    .font(.system(size: 10))
                    .foregroundColor(.white.opacity(0.5))
            }
            .buttonStyle(.plain)
            .help("Dismiss")
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 8)
        .background(Color.blue.opacity(0.08))
    }

    // MARK: - Messages

    private var messagesArea: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(spacing: 10) {
                    ForEach(viewModel.messages) { message in
                        MessageBubble(message: message)
                    }

                    if viewModel.isStreaming {
                        HStack {
                            ProgressView()
                                .controlSize(.small)
                                .scaleEffect(0.7)
                            Text("thinking...")
                                .font(.system(size: 11))
                                .foregroundColor(.white.opacity(0.4))
                            Spacer()
                        }
                        .padding(.leading, 8)
                        .id("streaming-indicator")
                    }

                    Color.clear
                        .frame(height: 1)
                        .id("bottom")
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
            }
            .onChange(of: viewModel.scrollAnchor) { _ in
                withAnimation(.easeOut(duration: 0.15)) {
                    proxy.scrollTo("bottom", anchor: .bottom)
                }
            }
            .onChange(of: viewModel.messages.count) { _ in
                withAnimation(.easeOut(duration: 0.15)) {
                    proxy.scrollTo("bottom", anchor: .bottom)
                }
            }
        }
    }

    // MARK: - Input Bar

    private var inputBar: some View {
        VStack(spacing: 8) {
            if let imgData = viewModel.pendingScreenshot, let nsImage = NSImage(data: imgData) {
                HStack(spacing: 6) {
                    Image(nsImage: nsImage)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .frame(height: 50)
                        .clipShape(RoundedRectangle(cornerRadius: 6))

                    Button(action: { viewModel.clearScreenshot() }) {
                        Image(systemName: "xmark.circle.fill")
                            .foregroundColor(.white.opacity(0.6))
                    }
                    .buttonStyle(.plain)

                    Spacer()
                }
                .padding(.horizontal, 12)
                .padding(.top, 8)
            }

            HStack(spacing: 8) {
                Button(action: { viewModel.captureScreen() }) {
                    Image(systemName: "camera.viewfinder")
                        .font(.system(size: 16))
                        .foregroundColor(.white.opacity(0.6))
                }
                .buttonStyle(.plain)
                .help("Capture screen")

                TextField("Message...", text: $viewModel.inputText)
                    .textFieldStyle(.plain)
                    .font(.system(size: 13))
                    .foregroundColor(.white)
                    .onSubmit { viewModel.send() }

                if viewModel.isStreaming {
                    Button(action: { viewModel.stopStreaming() }) {
                        Image(systemName: "stop.circle.fill")
                            .font(.system(size: 20))
                            .foregroundColor(.orange)
                    }
                    .buttonStyle(.plain)
                } else {
                    Button(action: { viewModel.send() }) {
                        Image(systemName: "arrow.up.circle.fill")
                            .font(.system(size: 20))
                            .foregroundColor(
                                viewModel.inputText.trimmingCharacters(in: .whitespaces).isEmpty
                                    ? .white.opacity(0.2)
                                    : .blue
                            )
                    }
                    .buttonStyle(.plain)
                    .disabled(viewModel.inputText.trimmingCharacters(in: .whitespaces).isEmpty)
                }
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 10)
        }
    }
}
