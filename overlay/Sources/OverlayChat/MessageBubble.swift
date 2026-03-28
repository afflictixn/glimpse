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

// MARK: - Inline Token

struct InlineToken {
    enum Kind {
        case bold(String)
        case code(String)
        case hex(token: String, digits: String)
    }
    let range: NSRange
    let kind: Kind
}

/// Parse a line for `code`, **bold**, and #hex inline tokens, sorted by position.
func inlineTokens(in text: String) -> [InlineToken] {
    let ns = text as NSString
    let len = ns.length
    var tokens: [InlineToken] = []

    // `code spans` (must come first — content inside backticks is handled specially)
    let codeRe = try! NSRegularExpression(pattern: "`([^`]+)`")
    for m in codeRe.matches(in: text, range: NSRange(0..<len)) {
        let content = ns.substring(with: m.range(at: 1))
        tokens.append(InlineToken(range: m.range, kind: .code(content)))
    }

    // **bold**
    let boldRe = try! NSRegularExpression(pattern: "\\*\\*(.+?)\\*\\*")
    for m in boldRe.matches(in: text, range: NSRange(0..<len)) {
        tokens.append(InlineToken(range: m.range, kind: .bold(ns.substring(with: m.range(at: 1)))))
    }

    // #hex colors (bare, not inside backticks — those are caught by code span)
    let hexRe = try! NSRegularExpression(pattern: "#([0-9A-Fa-f]{3}|[0-9A-Fa-f]{6}|[0-9A-Fa-f]{8})\\b")
    for m in hexRe.matches(in: text, range: NSRange(0..<len)) {
        let tok = ns.substring(with: m.range)
        tokens.append(InlineToken(range: m.range, kind: .hex(token: tok, digits: String(tok.dropFirst()))))
    }

    // Sort by position, then deduplicate overlapping (keep earliest)
    tokens.sort { $0.range.location < $1.range.location }
    return tokens
}

/// Regex to check if a string is purely a hex color like #FF8C00
private let hexOnlyRe = try! NSRegularExpression(pattern: "^#([0-9A-Fa-f]{3}|[0-9A-Fa-f]{6}|[0-9A-Fa-f]{8})$")

/// Build a `Text` from a single line, rendering `code`, **bold**, and #hex inline.
func parseInline(_ text: String, compact: Bool = false) -> Text {
    let ns = text as NSString
    let tokens = inlineTokens(in: text)
    if tokens.isEmpty { return Text(text) }

    let hexFontSize: CGFloat = compact ? 10 : 12
    let codeFontSize: CGFloat = compact ? 10 : 12
    let swatchSize: CGFloat = compact ? 9 : 11

    var result = Text("")
    var cursor = 0

    for token in tokens {
        if token.range.location < cursor { continue } // skip overlapping

        if token.range.location > cursor {
            let plain = ns.substring(with: NSRange(cursor..<token.range.location))
            result = result + Text(plain)
        }

        switch token.kind {
        case .bold(let inner):
            result = result + Text(inner).bold()

        case .code(let content):
            // If the code span is a hex color, render as swatch
            let trimmed = content.trimmingCharacters(in: .whitespaces)
            if hexOnlyRe.firstMatch(in: trimmed, range: NSRange(0..<trimmed.utf16.count)) != nil {
                let digits = String(trimmed.dropFirst())
                if let swatch = colorSwatchImage(hex: digits, size: swatchSize) {
                    result = result
                        + Text(Image(nsImage: swatch)).baselineOffset(-1)
                        + Text(trimmed)
                            .font(.system(size: hexFontSize, weight: .medium, design: .monospaced))
                            .foregroundColor(Color(nsColor: nsColor(fromHex: digits)!))
                } else {
                    result = result + Text(content)
                        .font(.system(size: codeFontSize, design: .monospaced))
                }
            } else {
                result = result + Text(content)
                    .font(.system(size: codeFontSize, design: .monospaced))
                    .foregroundColor(.white.opacity(0.7))
            }

        case .hex(let tok, let digits):
            if let swatch = colorSwatchImage(hex: digits, size: swatchSize) {
                result = result
                    + Text(Image(nsImage: swatch)).baselineOffset(-1)
                    + Text(tok)
                        .font(.system(size: hexFontSize, weight: .medium, design: .monospaced))
                        .foregroundColor(Color(nsColor: nsColor(fromHex: digits)!))
            } else {
                result = result + Text(tok)
            }
        }
        cursor = token.range.location + token.range.length
    }

    if cursor < ns.length {
        result = result + Text(ns.substring(from: cursor))
    }
    return result
}

/// Build rich `Text` from message content — headers, bullets, bold, code, and hex swatches.
func richText(from content: String, compact: Bool = false) -> Text {
    let lines = content.components(separatedBy: "\n")
    var result = Text("")

    let h1: CGFloat = compact ? 13 : 17
    let h2: CGFloat = compact ? 12 : 15
    let h3: CGFloat = compact ? 11 : 13

    for (i, line) in lines.enumerated() {
        if i > 0 { result = result + Text("\n") }

        // Detect markdown header prefix
        var headerLevel = 0
        var body = line
        if line.hasPrefix("### ")      { headerLevel = 3; body = String(line.dropFirst(4)) }
        else if line.hasPrefix("## ")   { headerLevel = 2; body = String(line.dropFirst(3)) }
        else if line.hasPrefix("# ")    { headerLevel = 1; body = String(line.dropFirst(2)) }

        // Detect bullet points: * or -
        var bullet = false
        if headerLevel == 0 {
            if body.hasPrefix("* ") || body.hasPrefix("- ") {
                bullet = true
                body = String(body.dropFirst(2))
            }
        }

        if bullet { result = result + Text("  \u{2022} ") }

        let parsed = parseInline(body, compact: compact)

        switch headerLevel {
        case 1:  result = result + parsed.font(.system(size: h1, weight: .bold))
        case 2:  result = result + parsed.font(.system(size: h2, weight: .bold))
        case 3:  result = result + parsed.font(.system(size: h3, weight: .semibold))
        default: result = result + parsed
        }
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
                        richText(from: message.content)
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
