import Foundation

final class OllamaService {
    let baseURL: URL
    let model: String

    init(baseURL: URL = URL(string: "http://localhost:11434")!, model: String = "qwen2.5vl") {
        self.baseURL = baseURL
        self.model = model
    }

    // MARK: - API Types

    private struct ChatRequest: Encodable {
        let model: String
        let messages: [Message]
        let stream: Bool

        struct Message: Encodable {
            let role: String
            let content: String
            let images: [String]?
        }
    }

    private struct StreamChunk: Decodable {
        let message: ResponseMessage?
        let done: Bool

        struct ResponseMessage: Decodable {
            let role: String
            let content: String
        }
    }

    // MARK: - Streaming Chat

    func streamChat(messages: [ChatMessage]) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    let url = baseURL.appendingPathComponent("api/chat")
                    var request = URLRequest(url: url)
                    request.httpMethod = "POST"
                    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
                    request.timeoutInterval = 300

                    let apiMessages = messages.map { msg in
                        ChatRequest.Message(
                            role: msg.role.rawValue,
                            content: msg.content,
                            images: msg.imageData.map { [$0.base64EncodedString()] }
                        )
                    }

                    let body = ChatRequest(model: model, messages: apiMessages, stream: true)
                    request.httpBody = try JSONEncoder().encode(body)

                    let (bytes, response) = try await URLSession.shared.bytes(for: request)

                    if let http = response as? HTTPURLResponse, http.statusCode != 200 {
                        continuation.finish(throwing: OllamaError.httpError(http.statusCode))
                        return
                    }

                    for try await line in bytes.lines {
                        if Task.isCancelled { break }
                        guard let data = line.data(using: .utf8) else { continue }
                        guard let chunk = try? JSONDecoder().decode(StreamChunk.self, from: data) else { continue }

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

    // MARK: - Health Check

    func isAvailable() async -> Bool {
        let url = baseURL.appendingPathComponent("api/tags")
        guard let (_, response) = try? await URLSession.shared.data(from: url),
              let http = response as? HTTPURLResponse else { return false }
        return http.statusCode == 200
    }

    enum OllamaError: LocalizedError {
        case httpError(Int)

        var errorDescription: String? {
            switch self {
            case .httpError(let code):
                return "Ollama returned HTTP \(code)"
            }
        }
    }
}
