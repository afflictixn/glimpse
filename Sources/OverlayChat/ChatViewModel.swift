import SwiftUI
import Combine

@MainActor
final class ChatViewModel: ObservableObject {
    @Published var messages: [ChatMessage] = []
    @Published var inputText: String = ""
    @Published var isStreaming: Bool = false
    @Published var pendingScreenshot: Data? = nil
    @Published var scrollAnchor: UUID? = nil

    private let screenCapture = ScreenCaptureService()
    private var streamTask: Task<Void, Never>?

    init() {}

    // MARK: - Send

    func send() {
        let text = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty, !isStreaming else { return }

        // User message
        let userMsg = ChatMessage(role: .user, content: text, imageData: pendingScreenshot)
        messages.append(userMsg)
        inputText = ""
        pendingScreenshot = nil
        scrollAnchor = userMsg.id

        // Placeholder for assistant response
        let assistantMsg = ChatMessage(role: .assistant, content: "")
        messages.append(assistantMsg)
        let assistantIdx = messages.count - 1
        isStreaming = true

        // TODO: plug in real model backend here
        // For now, simulate a streamed reply so the UI is testable
        streamTask = Task {
            let reply = "Got it — you said: \"\(text)\". (Model backend not connected yet.)"
            for char in reply {
                if Task.isCancelled { break }
                try? await Task.sleep(nanoseconds: 20_000_000) // 20ms per char
                messages[assistantIdx].content.append(char)
                scrollAnchor = messages[assistantIdx].id
            }
            isStreaming = false
        }
    }

    // MARK: - Stop

    func stopStreaming() {
        streamTask?.cancel()
        streamTask = nil
        isStreaming = false
    }

    // MARK: - Screen Capture

    func captureScreen() {
        pendingScreenshot = screenCapture.captureScreen()
    }

    func clearScreenshot() {
        pendingScreenshot = nil
    }

    // MARK: - Clear Chat

    func clearChat() {
        stopStreaming()
        messages.removeAll()
    }
}
