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

    // MARK: - Chat via Glimpse backend (POST /agent/chat)
    //
    // Routes through the Python GeneralAgent which has access to tools
    // (web search, memory, contacts, calendar, DB queries, etc.)

    private struct AgentChatRequest: Encodable {
        let message: String
    }

    private struct AgentChatResponse: Decodable {
        let response: String
    }

    func sendChat(message: String) async throws -> String {
        let url = glimpseURL.appendingPathComponent("agent/chat")
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.timeoutInterval = 120

        request.httpBody = try JSONEncoder().encode(AgentChatRequest(message: message))

        let (data, response) = try await URLSession.shared.data(for: request)

        if let http = response as? HTTPURLResponse, http.statusCode != 200 {
            throw GlimpseError.httpError(http.statusCode)
        }

        let decoded = try JSONDecoder().decode(AgentChatResponse.self, from: data)
        return decoded.response
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
