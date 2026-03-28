import SwiftUI

struct MessageBubble: View {
    let message: ChatMessage

    private var isUser: Bool { message.role == .user }
    private var isSystem: Bool { message.role == .system }

    var body: some View {
        if isSystem {
            // System messages (connection status, etc.)
            HStack {
                Image(systemName: "info.circle")
                    .font(.system(size: 10))
                    .foregroundColor(.yellow.opacity(0.7))
                Text(message.content)
                    .font(.system(size: 11))
                    .foregroundColor(.white.opacity(0.5))
                    .italic()
                Spacer()
            }
            .padding(.horizontal, 4)
        } else {
            HStack(alignment: .top, spacing: 8) {
                if isUser { Spacer(minLength: 30) }

                VStack(alignment: isUser ? .trailing : .leading, spacing: 6) {
                    if let imgData = message.imageData, let nsImage = NSImage(data: imgData) {
                        Image(nsImage: nsImage)
                            .resizable()
                            .aspectRatio(contentMode: .fit)
                            .frame(maxHeight: 120)
                            .clipShape(RoundedRectangle(cornerRadius: 8))
                    }

                    if !message.content.isEmpty {
                        Text(message.content)
                            .textSelection(.enabled)
                            .foregroundColor(.white)
                            .padding(.horizontal, 12)
                            .padding(.vertical, 8)
                            .background(
                                RoundedRectangle(cornerRadius: 14)
                                    .fill(isUser ? Color.blue.opacity(0.55) : Color.white.opacity(0.12))
                            )
                    }
                }

                if !isUser { Spacer(minLength: 30) }
            }
        }
    }
}
