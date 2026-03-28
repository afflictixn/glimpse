import Foundation

/// WebSocket client that connects to the Python backend's /ws endpoint.
/// Replaces the old WebSocketServer — the Python backend is now the server,
/// and the overlay (plus any browser) connects as a client.
final class WebSocketClient {
    private let url: URL
    private var task: URLSessionWebSocketTask?
    private let session = URLSession(configuration: .default)
    private var isRunning = false

    var onMessage: ((InboundMessage) -> Void)?

    init(url: URL = URL(string: "ws://localhost:3030/ws")!) {
        self.url = url
    }

    func start() {
        isRunning = true
        connect()
    }

    func stop() {
        isRunning = false
        task?.cancel(with: .goingAway, reason: nil)
        task = nil
    }

    func send(_ message: OutboundMessage) {
        guard let data = message.jsonData(),
              let text = String(data: data, encoding: .utf8) else { return }
        task?.send(.string(text)) { error in
            if let error {
                print("[ws-client] send error: \(error)")
            }
        }
    }

    // MARK: - Private

    private func connect() {
        guard isRunning else { return }
        task = session.webSocketTask(with: url)
        task?.resume()
        print("[ws-client] connecting to \(url)")
        recv()
    }

    private func recv() {
        task?.receive { [weak self] result in
            guard let self, self.isRunning else { return }
            switch result {
            case .success(.string(let text)):
                if let data = text.data(using: .utf8),
                   let msg = InboundMessage(from: data) {
                    self.onMessage?(msg)
                }
            case .success:
                break // binary frames ignored
            case .failure(let error):
                print("[ws-client] disconnected: \(error.localizedDescription)")
                self.reconnect()
                return
            }
            self.recv()
        }
    }

    private func reconnect() {
        guard isRunning else { return }
        task?.cancel(with: .goingAway, reason: nil)
        task = nil
        print("[ws-client] reconnecting in 3s...")
        DispatchQueue.global().asyncAfter(deadline: .now() + 3) { [weak self] in
            self?.connect()
        }
    }
}
