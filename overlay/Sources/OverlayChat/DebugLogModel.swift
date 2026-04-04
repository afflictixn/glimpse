import Foundation

struct DebugLogEntry: Identifiable {
    let id = UUID()
    let timestamp: String
    let level: String
    let source: String
    let message: String
}

final class DebugLogModel: ObservableObject {
    static let maxEntries = 200

    @Published var entries: [DebugLogEntry] = []
    @Published var isActive = false

    func append(timestamp: String, level: String, source: String, message: String) {
        let entry = DebugLogEntry(timestamp: timestamp, level: level, source: source, message: message)
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            self.isActive = true
            self.entries.append(entry)
            if self.entries.count > Self.maxEntries {
                self.entries.removeFirst(self.entries.count - Self.maxEntries)
            }
        }
    }
}
