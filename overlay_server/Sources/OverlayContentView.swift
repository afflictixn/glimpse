import SwiftUI

// MARK: - ViewModel

final class OverlayViewModel: ObservableObject {
    @Published var proposalText: String = ""
    @Published var conversationHTML: [ConversationEntry] = []
    @Published var showConversation: Bool = false
    @Published var assistantLabel: String = "AI"
    @Published var isVisible: Bool = false

    var onAction: ((OutboundMessage) -> Void)?

    struct ConversationEntry: Identifiable {
        let id = UUID()
        let role: String
        let text: String
    }

    func setProposal(_ text: String) {
        proposalText = text
        conversationHTML = []
        showConversation = false
        isVisible = true
    }

    func showConversationWith(_ text: String) {
        conversationHTML = [ConversationEntry(role: "assistant", text: text)]
        showConversation = true
    }

    func appendConversation(role: String, text: String) {
        conversationHTML.append(ConversationEntry(role: role, text: text))
    }

    func hide() {
        isVisible = false
    }

    func send(_ action: String, text: String? = nil) {
        onAction?(.action(action: action, text: text))
    }

    func sendPause(_ paused: Bool) {
        onAction?(.pauseToggle(paused: paused))
    }
}

// MARK: - Card View

struct OverlayContentView: View {
    @ObservedObject var viewModel: OverlayViewModel
    @State private var followUpText: String = ""

    private let cardBg = Color(red: 30/255, green: 30/255, blue: 36/255, opacity: 0.94)
    private let accentGreen = Color(red: 76/255, green: 175/255, blue: 80/255)
    private let accentBlue = Color(red: 33/255, green: 150/255, blue: 243/255)

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            header
            proposalBody
            buttonRow

            if viewModel.showConversation {
                conversationArea
                followUpRow
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 14)
        .frame(width: 380)
        .background(cardBg)
        .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
        .shadow(color: .black.opacity(0.45), radius: 24, x: 0, y: 4)
    }

    // MARK: - Subviews

    private var header: some View {
        HStack(spacing: 6) {
            Text("🔍")
                .font(.system(size: 16))
            Text("Screen Agent")
                .font(.system(size: 13, weight: .bold))
                .foregroundColor(Color(white: 0.88))
            Spacer()
            Button(action: { viewModel.send("dismiss") }) {
                Text("✕")
                    .font(.system(size: 14))
                    .foregroundColor(Color(white: 0.53))
            }
            .buttonStyle(.plain)
            .onHover { hovering in
                if hovering {
                    NSCursor.pointingHand.push()
                } else {
                    NSCursor.pop()
                }
            }
        }
    }

    private var proposalBody: some View {
        Text(viewModel.proposalText)
            .font(.system(size: 12))
            .foregroundColor(Color(white: 0.8))
            .lineSpacing(3)
            .fixedSize(horizontal: false, vertical: true)
    }

    private var buttonRow: some View {
        HStack(spacing: 8) {
            actionButton("Yes, do it", color: accentGreen) {
                viewModel.send("accept")
            }
            actionButton("Tell me more", color: accentBlue) {
                viewModel.send("escalate")
            }
            Spacer()
        }
    }

    private var conversationArea: some View {
        ScrollViewReader { proxy in
            ScrollView {
                VStack(alignment: .leading, spacing: 6) {
                    ForEach(viewModel.conversationHTML) { entry in
                        HStack(alignment: .top, spacing: 0) {
                            Text(entry.role == "assistant" ? "\(viewModel.assistantLabel): " : "You: ")
                                .font(.system(size: 12, weight: .bold))
                                .foregroundColor(entry.role == "assistant"
                                    ? Color(red: 0.67, green: 0.93, blue: 1.0)
                                    : Color(red: 1.0, green: 0.93, blue: 0.67))
                            Text(entry.text)
                                .font(.system(size: 12))
                                .foregroundColor(Color(white: 0.87))
                                .fixedSize(horizontal: false, vertical: true)
                        }
                        .id(entry.id)
                    }
                }
                .padding(8)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .onChange(of: viewModel.conversationHTML.count) { _ in
                if let last = viewModel.conversationHTML.last {
                    proxy.scrollTo(last.id, anchor: .bottom)
                }
            }
        }
        .frame(minHeight: 80, maxHeight: 200)
        .background(Color.black.opacity(0.3))
        .clipShape(RoundedRectangle(cornerRadius: 6))
        .overlay(
            RoundedRectangle(cornerRadius: 6)
                .stroke(Color(white: 0.27), lineWidth: 1)
        )
    }

    private var followUpRow: some View {
        HStack(spacing: 8) {
            TextField("Ask a follow-up…", text: $followUpText)
                .textFieldStyle(.plain)
                .font(.system(size: 12))
                .foregroundColor(Color(white: 0.87))
                .padding(.horizontal, 10)
                .padding(.vertical, 6)
                .background(Color.white.opacity(0.08))
                .clipShape(RoundedRectangle(cornerRadius: 6))
                .overlay(
                    RoundedRectangle(cornerRadius: 6)
                        .stroke(Color(white: 0.33), lineWidth: 1)
                )
                .onSubmit { sendFollowUp() }

            actionButton("Send", color: accentBlue) {
                sendFollowUp()
            }
            .frame(width: 60)
        }
    }

    // MARK: - Helpers

    private func actionButton(_ label: String, color: Color, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Text(label)
                .font(.system(size: 11))
                .foregroundColor(.white)
                .padding(.horizontal, 14)
                .padding(.vertical, 6)
                .background(color)
                .clipShape(RoundedRectangle(cornerRadius: 6))
        }
        .buttonStyle(.plain)
        .onHover { hovering in
            if hovering { NSCursor.pointingHand.push() } else { NSCursor.pop() }
        }
    }

    private func sendFollowUp() {
        let text = followUpText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        viewModel.appendConversation(role: "user", text: text)
        viewModel.send("follow_up", text: text)
        followUpText = ""
    }
}
