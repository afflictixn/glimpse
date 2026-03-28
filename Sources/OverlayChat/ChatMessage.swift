import Foundation

struct ChatMessage: Identifiable {
    let id: UUID
    let role: Role
    var content: String
    var imageData: Data?
    let timestamp: Date

    enum Role: String {
        case user
        case assistant
        case system
    }

    init(role: Role, content: String, imageData: Data? = nil) {
        self.id = UUID()
        self.role = role
        self.content = content
        self.imageData = imageData
        self.timestamp = Date()
    }
}
