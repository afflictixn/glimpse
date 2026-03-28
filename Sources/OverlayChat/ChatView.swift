import SwiftUI

struct ChatView: View {
    @ObservedObject var viewModel: ChatViewModel

    var body: some View {
        VStack(spacing: 0) {
            headerBar
            Divider().overlay(Color.white.opacity(0.15))
            messagesArea
            Divider().overlay(Color.white.opacity(0.15))
            inputBar
        }
        .background(VisualEffectBackground(material: .hudWindow))
    }

    // MARK: - Header

    private var headerBar: some View {
        HStack(spacing: 10) {
            Text("overlay")
                .font(.system(size: 13, weight: .semibold, design: .monospaced))
                .foregroundColor(.white)

            Spacer()

            Button(action: { viewModel.clearChat() }) {
                Image(systemName: "trash")
                    .font(.system(size: 12))
                    .foregroundColor(.white.opacity(0.6))
            }
            .buttonStyle(.plain)
            .help("Clear chat")
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
    }

    // MARK: - Messages

    private var messagesArea: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(spacing: 10) {
                    ForEach(viewModel.messages) { message in
                        MessageBubble(message: message)
                    }

                    // Streaming indicator
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
            // Screenshot preview
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
                // Capture screen button
                Button(action: { viewModel.captureScreen() }) {
                    Image(systemName: "camera.viewfinder")
                        .font(.system(size: 16))
                        .foregroundColor(.white.opacity(0.6))
                }
                .buttonStyle(.plain)
                .help("Capture screen")

                // Text field
                TextField("Message...", text: $viewModel.inputText)
                    .textFieldStyle(.plain)
                    .font(.system(size: 13))
                    .foregroundColor(.white)
                    .onSubmit { viewModel.send() }

                // Send / Stop
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
