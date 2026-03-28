import Foundation

// MARK: - Inbound (Python → Swift)

enum InboundMessage {
    case showProposal(text: String, proposalId: Int?)
    case showConversation(text: String)
    case appendConversation(role: String, text: String)
    case setAssistantLabel(label: String)
    case hide
}

extension InboundMessage {
    init?(from data: Data) {
        guard let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let type = obj["type"] as? String else { return nil }
        switch type {
        case "show_proposal":
            guard let text = obj["text"] as? String else { return nil }
            let pid = obj["proposalId"] as? Int
            self = .showProposal(text: text, proposalId: pid)
        case "show_conversation":
            guard let text = obj["text"] as? String else { return nil }
            self = .showConversation(text: text)
        case "append_conversation":
            guard let role = obj["role"] as? String,
                  let text = obj["text"] as? String else { return nil }
            self = .appendConversation(role: role, text: text)
        case "set_assistant_label":
            guard let label = obj["label"] as? String else { return nil }
            self = .setAssistantLabel(label: label)
        case "hide":
            self = .hide
        default:
            return nil
        }
    }
}

// MARK: - Outbound (Swift → Python)

struct OutboundAction: Encodable {
    let type = "action"
    let action: String
    var text: String?
}

struct OutboundPauseToggle: Encodable {
    let type = "pause_toggle"
    let paused: Bool
}

enum OutboundMessage {
    case action(action: String, text: String? = nil)
    case pauseToggle(paused: Bool)

    func jsonData() -> Data? {
        let encoder = JSONEncoder()
        switch self {
        case .action(let action, let text):
            return try? encoder.encode(OutboundAction(action: action, text: text))
        case .pauseToggle(let paused):
            return try? encoder.encode(OutboundPauseToggle(paused: paused))
        }
    }
}
