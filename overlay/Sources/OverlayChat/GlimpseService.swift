import Foundation

/// Client for the Glimpse FastAPI backend (localhost:3030)
/// and Ollama (localhost:11434) for direct chat.
final class GlimpseService {
    let glimpseURL: URL
    let ollamaURL: URL
    let chatModel: String

    init(
        glimpseURL: URL = URL(string: "http://localhost:3030")!,
        ollamaURL: URL = URL(string: "http://localhost:11434")!,
        chatModel: String = "gemma3:12b"
    ) {
        self.glimpseURL = glimpseURL
        self.ollamaURL = ollamaURL
        self.chatModel = chatModel
    }

    // MARK: - Health

    func isGlimpseAvailable() async -> Bool {
        let url = glimpseURL.appendingPathComponent("health")
        guard let (_, response) = try? await URLSession.shared.data(from: url),
              let http = response as? HTTPURLResponse else { return false }
        return http.statusCode == 200
    }

    func isOllamaAvailable() async -> Bool {
        let url = ollamaURL.appendingPathComponent("api/tags")
        guard let (_, response) = try? await URLSession.shared.data(from: url),
              let http = response as? HTTPURLResponse else { return false }
        return http.statusCode == 200
    }

    // MARK: - Glimpse Data Routes

    func fetchRecentActions(limit: Int = 10) async throws -> [ActionResult] {
        var components = URLComponents(url: glimpseURL.appendingPathComponent("actions"), resolvingAgainstBaseURL: false)!
        components.queryItems = [URLQueryItem(name: "limit", value: "\(limit)")]
        let (data, _) = try await URLSession.shared.data(from: components.url!)
        return try JSONDecoder().decode([ActionResult].self, from: data)
    }

    func fetchRecentEvents(limit: Int = 10) async throws -> [EventResult] {
        var components = URLComponents(url: glimpseURL.appendingPathComponent("events"), resolvingAgainstBaseURL: false)!
        components.queryItems = [URLQueryItem(name: "limit", value: "\(limit)")]
        let (data, _) = try await URLSession.shared.data(from: components.url!)
        return try JSONDecoder().decode([EventResult].self, from: data)
    }

    func searchOCR(query: String, limit: Int = 10) async throws -> [OCRSearchResult] {
        var components = URLComponents(url: glimpseURL.appendingPathComponent("search"), resolvingAgainstBaseURL: false)!
        components.queryItems = [
            URLQueryItem(name: "q", value: query),
            URLQueryItem(name: "limit", value: "\(limit)"),
        ]
        let (data, _) = try await URLSession.shared.data(from: components.url!)
        return try JSONDecoder().decode([OCRSearchResult].self, from: data)
    }

    // MARK: - Chat via Ollama (streaming)
    //
    // The General Agent's POST /chat doesn't exist yet, so we talk to Ollama
    // directly for now. The streaming format is identical, so switching later
    // is a one-line URL change.

    func streamChat(
        messages: [ChatMessage],
        contextSummary: String? = nil
    ) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    let url = ollamaURL.appendingPathComponent("api/chat")
                    var request = URLRequest(url: url)
                    request.httpMethod = "POST"
                    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
                    request.timeoutInterval = 300

                    var apiMessages: [[String: Any]] = []

                    // Inject context from Glimpse as a system message
                    if let ctx = contextSummary, !ctx.isEmpty {
                        apiMessages.append([
                            "role": "system",
                            "content": "You are an ambient AI assistant. Here is recent screen context:\n\(ctx)\n\nUse this context to help the user. Be concise and proactive."
                        ])
                    } else {
                        apiMessages.append([
                            "role": "system",
                            "content": "You are an ambient AI assistant called Glimpse. You can see the user's screen. Be concise, helpful, and proactive."
                        ])
                    }

                    for msg in messages {
                        var m: [String: Any] = [
                            "role": msg.role.rawValue,
                            "content": msg.content,
                        ]
                        if let imgData = msg.imageData {
                            m["images"] = [imgData.base64EncodedString()]
                        }
                        apiMessages.append(m)
                    }

                    let body: [String: Any] = [
                        "model": chatModel,
                        "messages": apiMessages,
                        "stream": true,
                    ]
                    request.httpBody = try JSONSerialization.data(withJSONObject: body)

                    let (bytes, response) = try await URLSession.shared.bytes(for: request)

                    if let http = response as? HTTPURLResponse, http.statusCode != 200 {
                        continuation.finish(throwing: GlimpseError.httpError(http.statusCode))
                        return
                    }

                    for try await line in bytes.lines {
                        if Task.isCancelled { break }
                        guard let data = line.data(using: .utf8) else { continue }
                        guard let chunk = try? JSONDecoder().decode(OllamaChatChunk.self, from: data) else { continue }

                        if let content = chunk.message?.content, !content.isEmpty {
                            continuation.yield(content)
                        }
                        if chunk.done {
                            break
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }

            continuation.onTermination = { _ in
                task.cancel()
            }
        }
    }

    // MARK: - Build context summary from Glimpse data

    func buildContextSummary() async -> String? {
        var parts: [String] = []

        if let events = try? await fetchRecentEvents(limit: 3) {
            for event in events {
                var line = "[\(event.app_type)] \(event.summary)"
                if let app = event.app_name { line += " (app: \(app))" }
                parts.append(line)
            }
        }

        if let actions = try? await fetchRecentActions(limit: 3) {
            for action in actions {
                parts.append("[\(action.action_type)] \(action.action_description)")
            }
        }

        return parts.isEmpty ? nil : parts.joined(separator: "\n")
    }

    // MARK: - Types

    private struct OllamaChatChunk: Decodable {
        let message: OllamaMessage?
        let done: Bool

        struct OllamaMessage: Decodable {
            let role: String
            let content: String
        }
    }

    enum GlimpseError: LocalizedError {
        case httpError(Int)
        case backendUnavailable

        var errorDescription: String? {
            switch self {
            case .httpError(let code):
                return "Backend returned HTTP \(code)"
            case .backendUnavailable:
                return "Glimpse backend not reachable"
            }
        }
    }
}
