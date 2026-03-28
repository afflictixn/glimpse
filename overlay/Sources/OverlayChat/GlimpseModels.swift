import Foundation

// MARK: - API Response Models (match Python Pydantic schemas)

struct ActionResult: Codable, Identifiable {
    let action_id: Int
    let event_id: Int
    let frame_id: Int
    let agent_name: String
    let action_type: String
    let action_description: String
    let metadata: [String: AnyCodable]?
    let created_at: String
    let event_summary: String
    let event_app_type: String
    let timestamp: String
    let app_name: String?
    let window_name: String?

    var id: Int { action_id }
}

struct EventResult: Codable, Identifiable {
    let event_id: Int
    let frame_id: Int
    let agent_name: String
    let app_type: String
    let summary: String
    let metadata: [String: AnyCodable]?
    let created_at: String
    let timestamp: String
    let app_name: String?
    let window_name: String?

    var id: Int { event_id }
}

struct OCRSearchResult: Codable {
    let frame_id: Int
    let text: String
    let app_name: String?
    let window_name: String?
    let timestamp: String
    let capture_trigger: String
    let confidence: Double?
    let snapshot_path: String?
}

struct HealthResponse: Codable {
    let status: String
    let uptime_seconds: Double
    let counts: HealthCounts
}

struct HealthCounts: Codable {
    let frames: Int
    let events: Int
    let actions: Int
}

// MARK: - Chat streaming

struct ChatRequest: Encodable {
    let message: String
    let screenshot_b64: String?
}

struct ChatChunk: Decodable {
    let role: String?
    let content: String
    let done: Bool
}

// MARK: - AnyCodable (lightweight JSON wrapper)

struct AnyCodable: Codable {
    let value: Any

    init(_ value: Any) {
        self.value = value
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let str = try? container.decode(String.self) {
            value = str
        } else if let int = try? container.decode(Int.self) {
            value = int
        } else if let double = try? container.decode(Double.self) {
            value = double
        } else if let bool = try? container.decode(Bool.self) {
            value = bool
        } else if let dict = try? container.decode([String: AnyCodable].self) {
            value = dict.mapValues { $0.value }
        } else if let arr = try? container.decode([AnyCodable].self) {
            value = arr.map { $0.value }
        } else {
            value = NSNull()
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        if let str = value as? String {
            try container.encode(str)
        } else if let int = value as? Int {
            try container.encode(int)
        } else if let double = value as? Double {
            try container.encode(double)
        } else if let bool = value as? Bool {
            try container.encode(bool)
        } else {
            try container.encodeNil()
        }
    }
}
