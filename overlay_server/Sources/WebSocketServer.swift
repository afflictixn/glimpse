import Foundation
import Network

final class WebSocketServer {
    private let listener: NWListener
    private var connections: [Int: NWConnection] = [:]
    private var nextId = 0
    private let queue = DispatchQueue(label: "ws-server", qos: .userInitiated)

    var onMessage: ((InboundMessage) -> Void)?

    init(port: UInt16 = 9321) {
        let params = NWParameters.tcp
        let wsOptions = NWProtocolWebSocket.Options()
        params.defaultProtocolStack.applicationProtocols.insert(wsOptions, at: 0)
        listener = try! NWListener(using: params, on: NWEndpoint.Port(rawValue: port)!)
    }

    func start() {
        listener.stateUpdateHandler = { state in
            switch state {
            case .ready:
                print("[ws] listening on port \(self.listener.port?.rawValue ?? 0)")
            case .failed(let err):
                print("[ws] listener failed: \(err)")
            default:
                break
            }
        }
        listener.newConnectionHandler = { [weak self] conn in
            self?.accept(conn)
        }
        listener.start(queue: queue)
    }

    func stop() {
        listener.cancel()
        for conn in connections.values {
            conn.cancel()
        }
        connections.removeAll()
    }

    func broadcast(_ message: OutboundMessage) {
        guard let data = message.jsonData() else { return }
        let meta = NWProtocolWebSocket.Metadata(opcode: .text)
        let ctx = NWConnection.ContentContext(identifier: "ws", metadata: [meta])
        for conn in connections.values {
            conn.send(content: data, contentContext: ctx, isComplete: true, completion: .idempotent)
        }
    }

    // MARK: - Private

    private func accept(_ connection: NWConnection) {
        let id = nextId
        nextId += 1
        connections[id] = connection

        connection.stateUpdateHandler = { [weak self] state in
            switch state {
            case .ready:
                print("[ws] client \(id) connected")
            case .failed, .cancelled:
                print("[ws] client \(id) disconnected")
                self?.connections.removeValue(forKey: id)
            default:
                break
            }
        }
        connection.start(queue: queue)
        recv(connection)
    }

    private func recv(_ connection: NWConnection) {
        connection.receiveMessage { [weak self] content, context, _, error in
            if let error {
                print("[ws] recv error: \(error)")
                return
            }
            if let data = content, self?.isWebSocketText(context) == true,
               let msg = InboundMessage(from: data) {
                DispatchQueue.main.async {
                    self?.onMessage?(msg)
                }
            }
            self?.recv(connection)
        }
    }

    private func isWebSocketText(_ ctx: NWConnection.ContentContext?) -> Bool {
        guard let meta = ctx?.protocolMetadata(definition: NWProtocolWebSocket.definition)
                as? NWProtocolWebSocket.Metadata else { return false }
        return meta.opcode == .text
    }
}
