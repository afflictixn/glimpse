import SwiftUI
import AppKit

// MARK: - Color Swatch Helpers

/// Parse a 3, 6, or 8 character hex string (no leading #) into an NSColor.
private func nsColor(fromHex hex: String) -> NSColor? {
    var hexSanitized = hex.trimmingCharacters(in: .whitespacesAndNewlines)
    if hexSanitized.hasPrefix("#") { hexSanitized = String(hexSanitized.dropFirst()) }

    var rgb: UInt64 = 0
    guard Scanner(string: hexSanitized).scanHexInt64(&rgb) else { return nil }

    switch hexSanitized.count {
    case 3: // RGB → RRGGBB
        let r = Double((rgb >> 8) & 0xF) / 15
        let g = Double((rgb >> 4) & 0xF) / 15
        let b = Double(rgb & 0xF) / 15
        return NSColor(red: r, green: g, blue: b, alpha: 1)
    case 6: // RRGGBB
        let r = Double((rgb >> 16) & 0xFF) / 255
        let g = Double((rgb >> 8) & 0xFF) / 255
        let b = Double(rgb & 0xFF) / 255
        return NSColor(red: r, green: g, blue: b, alpha: 1)
    case 8: // RRGGBBAA
        let r = Double((rgb >> 24) & 0xFF) / 255
        let g = Double((rgb >> 16) & 0xFF) / 255
        let b = Double((rgb >> 8) & 0xFF) / 255
        let a = Double(rgb & 0xFF) / 255
        return NSColor(red: r, green: g, blue: b, alpha: a)
    default:
        return nil
    }
}

/// Create a tiny colored-square image for inline use in Text.
private func colorSwatchImage(hex: String, size: CGFloat = 11) -> NSImage? {
    guard let color = nsColor(fromHex: hex) else { return nil }
    let img = NSImage(size: NSSize(width: size, height: size))
    img.lockFocus()
    color.setFill()
    let path = NSBezierPath(roundedRect: NSRect(x: 0, y: 0, width: size, height: size),
                            xRadius: 2.5, yRadius: 2.5)
    path.fill()
    // thin border so light colors remain visible on dark backgrounds
    NSColor.white.withAlphaComponent(0.35).setStroke()
    path.lineWidth = 0.5
    path.stroke()
    img.unlockFocus()
    return img
}

/// Build a rich `Text` that replaces hex color codes with ■ swatch + styled code.
/// Matches #RGB, #RRGGBB, #RRGGBBAA patterns.
private func richColorText(from content: String) -> Text {
    let pattern = try! NSRegularExpression(pattern: "#([0-9A-Fa-f]{3}|[0-9A-Fa-f]{6}|[0-9A-Fa-f]{8})\\b")
    let nsContent = content as NSString
    let matches = pattern.matches(in: content, range: NSRange(location: 0, length: nsContent.length))

    if matches.isEmpty {
        return Text(content)
    }

    var result = Text("")
    var cursor = 0

    for match in matches {
        let matchRange = match.range
        // Append text before the match
        if matchRange.location > cursor {
            let before = nsContent.substring(with: NSRange(location: cursor, length: matchRange.location - cursor))
            result = result + Text(before)
        }
        let hexToken = nsContent.substring(with: matchRange) // e.g. "#FF5636"
        let hexDigits = String(hexToken.dropFirst())          // e.g. "FF5636"

        if let swatch = colorSwatchImage(hex: hexDigits) {
            result = result
                + Text(Image(nsImage: swatch)).baselineOffset(-1)
                + Text(" ")
                + Text(hexToken)
                    .font(.system(size: 12, weight: .medium, design: .monospaced))
                    .foregroundColor(Color(nsColor: nsColor(fromHex: hexDigits)!))
        } else {
            result = result + Text(hexToken)
        }
        cursor = matchRange.location + matchRange.length
    }

    // Append any trailing text
    if cursor < nsContent.length {
        let trailing = nsContent.substring(from: cursor)
        result = result + Text(trailing)
    }

    return result
}

// MARK: - Typing Dots Animation

struct TypingDotsView: View {
    @State private var phase: Int = 0
    private let dotCount = 3
    private let dotSize: CGFloat = 5
    private let color = Color.white.opacity(0.6)

    var body: some View {
        HStack(spacing: 4) {
            ForEach(0..<dotCount, id: \.self) { i in
                Circle()
                    .fill(color)
                    .frame(width: dotSize, height: dotSize)
                    .offset(y: phase == i ? -4 : 0)
            }
        }
        .onAppear {
            animate()
        }
    }

    private func animate() {
        func step(_ i: Int) {
            withAnimation(.easeInOut(duration: 0.3)) {
                phase = i
            }
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
                step((i + 1) % (dotCount + 1))
            }
        }
        step(0)
    }
}

// MARK: - Thinking Bubble

struct ThinkingBubble: View {
    var body: some View {
        HStack(alignment: .top, spacing: 8) {
            TypingDotsView()
                .padding(.horizontal, 12)
                .padding(.vertical, 10)
                .background(
                    RoundedRectangle(cornerRadius: 12, style: .continuous)
                        .fill(Color.white.opacity(0.06))
                )
            Spacer(minLength: 30)
        }
    }
}

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
                        richColorText(from: message.content)
                            .textSelection(.enabled)
                            .foregroundColor(.white.opacity(isUser ? 0.9 : 0.8))
                            .padding(.horizontal, 12)
                            .padding(.vertical, 8)
                            .background(
                                RoundedRectangle(cornerRadius: 12, style: .continuous)
                                    .fill(isUser
                                        ? Color(hue: 0.6, saturation: 0.5, brightness: 0.9).opacity(0.2)
                                        : Color.white.opacity(0.06))
                            )
                    }
                }

                if !isUser { Spacer(minLength: 30) }
            }
        }
    }
}
