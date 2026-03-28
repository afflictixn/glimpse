import SwiftUI
import Combine

@MainActor
final class ChatViewModel: ObservableObject {
    @Published var messages: [ChatMessage] = []
    @Published var inputText: String = ""
    @Published var isStreaming: Bool = false
    @Published var pendingScreenshot: Data? = nil
    @Published var scrollAnchor: UUID? = nil

    // Proactive suggestions from WebSocket
    @Published var currentSuggestion: ProactiveSuggestion? = nil
    @Published var connectionStatus: ConnectionStatus = .connecting

    let glimpseService = GlimpseService()
    private let screenCapture = ScreenCaptureService()
    private var streamTask: Task<Void, Never>?

    enum ConnectionStatus {
        case connecting, connected, disconnected
    }

    init() {
        checkConnections()
    }

    // MARK: - Connection Check

    func checkConnections() {
        connectionStatus = .connecting
        Task {
            let ollama = await glimpseService.isOllamaAvailable()
            let glimpse = await glimpseService.isGlimpseAvailable()

            if ollama {
                connectionStatus = .connected
            } else {
                connectionStatus = .disconnected
            }

            if !ollama {
                messages.append(ChatMessage(
                    role: .system,
                    content: "Ollama not reachable at localhost:11434. Chat won't work until it's running."
                ))
            }
            if !glimpse {
                messages.append(ChatMessage(
                    role: .system,
                    content: "Glimpse backend not reachable at localhost:3030. Screen context unavailable."
                ))
            }
        }
    }

    // MARK: - Send

    func send() {
        let text = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty, !isStreaming else { return }

        let userMsg = ChatMessage(role: .user, content: text, imageData: pendingScreenshot)
        messages.append(userMsg)
        inputText = ""
        pendingScreenshot = nil
        scrollAnchor = userMsg.id

        let assistantMsg = ChatMessage(role: .assistant, content: "")
        messages.append(assistantMsg)
        let assistantIdx = messages.count - 1
        isStreaming = true

        streamTask = Task {
            do {
                let response = try await glimpseService.sendChat(message: text)
                messages[assistantIdx].content = response
                scrollAnchor = messages[assistantIdx].id
            } catch {
                if messages[assistantIdx].content.isEmpty {
                    messages[assistantIdx].content = "Error: \(error.localizedDescription)"
                }
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

    // MARK: - Proactive Suggestions (from WebSocket)

    func handleInboundMessage(_ message: InboundMessage) {
        switch message {
        case .showProposal(let text, let proposalId):
            currentSuggestion = ProactiveSuggestion(
                text: text,
                proposalId: proposalId,
                importance: .medium
            )
        case .showConversation(let text):
            currentSuggestion = nil
            messages.append(ChatMessage(role: .assistant, content: text))
            scrollAnchor = messages.last?.id
        case .appendConversation(let role, let text):
            let chatRole: ChatMessage.Role = role == "user" ? .user : .assistant
            messages.append(ChatMessage(role: chatRole, content: text))
            scrollAnchor = messages.last?.id
        case .setAssistantLabel:
            break
        case .hide:
            currentSuggestion = nil
        }
    }

    func dismissSuggestion() {
        currentSuggestion = nil
    }

    func expandSuggestion() {
        guard let suggestion = currentSuggestion else { return }
        messages.append(ChatMessage(role: .assistant, content: suggestion.text))
        scrollAnchor = messages.last?.id
        currentSuggestion = nil
    }
}

// MARK: - ProactiveSuggestion

struct ProactiveSuggestion: Identifiable {
    let id = UUID()
    let text: String
    let proposalId: Int?
    let importance: Importance
    let timestamp = Date()

    enum Importance {
        case low, medium, high
    }
}
