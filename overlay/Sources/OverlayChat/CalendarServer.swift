import Foundation
import EventKit
import Network

/// Tiny HTTP server on port 9322 that exposes calendar events.
/// The Python backend calls GET /calendar?days=7 to fetch events
/// through the overlay's app-level EventKit permission.
final class CalendarServer {
    private let store = EKEventStore()
    private var listener: NWListener?
    private let queue = DispatchQueue(label: "calendar-server", qos: .utility)
    private(set) var hasAccess = false

    func start() {
        requestAccess()
        startListener()
    }

    func stop() {
        listener?.cancel()
    }

    // MARK: - Permission

    private func requestAccess() {
        let status = EKEventStore.authorizationStatus(for: .event)

        if #available(macOS 14.0, *) {
            if status == .fullAccess {
                hasAccess = true
                print("[calendar] already authorized (fullAccess)")
                return
            }
        }
        if status == .authorized {
            hasAccess = true
            print("[calendar] already authorized")
            return
        }

        if #available(macOS 14.0, *) {
            store.requestFullAccessToEvents { [weak self] granted, error in
                self?.hasAccess = granted
                print("[calendar] fullAccess request: granted=\(granted)")
            }
        } else {
            store.requestAccess(to: .event) { [weak self] granted, error in
                self?.hasAccess = granted
                print("[calendar] access request: granted=\(granted)")
            }
        }
    }

    // MARK: - HTTP Listener

    private func startListener() {
        let params = NWParameters.tcp
        listener = try? NWListener(using: params, on: 9322)

        listener?.stateUpdateHandler = { state in
            switch state {
            case .ready:
                print("[calendar] HTTP server listening on port 9322")
            case .failed(let err):
                print("[calendar] listener failed: \(err)")
            default:
                break
            }
        }

        listener?.newConnectionHandler = { [weak self] conn in
            self?.handleConnection(conn)
        }

        listener?.start(queue: queue)
    }

    // MARK: - Request Handling

    private func handleConnection(_ connection: NWConnection) {
        connection.start(queue: queue)
        connection.receive(minimumIncompleteLength: 1, maximumLength: 4096) { [weak self] data, _, _, error in
            guard let self = self, let data = data else {
                connection.cancel()
                return
            }

            let request = String(data: data, encoding: .utf8) ?? ""
            let response = self.handleRequest(request)
            let httpResponse = self.buildHTTPResponse(body: response)

            connection.send(content: httpResponse.data(using: .utf8), completion: .contentProcessed { _ in
                connection.cancel()
            })
        }
    }

    private func handleRequest(_ raw: String) -> String {
        // Parse days from query: GET /calendar?days=7
        var days = 7
        if let range = raw.range(of: "days=") {
            let start = range.upperBound
            let rest = raw[start...]
            if let end = rest.firstIndex(where: { !$0.isNumber }) {
                days = Int(rest[start..<end]) ?? 7
            } else {
                days = Int(rest) ?? 7
            }
        }
        days = min(days, 30)

        guard hasAccess else {
            return "{\"error\": \"Calendar access not granted. Allow in System Settings > Privacy > Calendars for GlimpseOverlay.\"}"
        }

        let start = Date()
        let end = Calendar.current.date(byAdding: .day, value: days, to: start)!
        let predicate = store.predicateForEvents(withStart: start, end: end, calendars: nil)
        let events = store.events(matching: predicate)

        if events.isEmpty {
            return "{\"events\": [], \"note\": \"No events in the next \(days) days.\"}"
        }

        var results: [[String: Any]] = []
        for event in events.prefix(50) {
            var entry: [String: Any] = [
                "title": event.title ?? "",
                "start": ISO8601DateFormatter().string(from: event.startDate),
                "end": ISO8601DateFormatter().string(from: event.endDate),
                "all_day": event.isAllDay,
            ]
            if let loc = event.location, !loc.isEmpty {
                entry["location"] = loc
            }
            if let notes = event.notes, !notes.isEmpty {
                entry["notes"] = String(notes.prefix(200))
            }
            results.append(entry)
        }

        guard let jsonData = try? JSONSerialization.data(withJSONObject: results),
              let jsonStr = String(data: jsonData, encoding: .utf8) else {
            return "{\"error\": \"Failed to serialize events\"}"
        }
        return jsonStr
    }

    private func buildHTTPResponse(body: String) -> String {
        let contentLength = body.utf8.count
        return """
        HTTP/1.1 200 OK\r
        Content-Type: application/json\r
        Content-Length: \(contentLength)\r
        Access-Control-Allow-Origin: *\r
        Connection: close\r
        \r
        \(body)
        """
    }
}
