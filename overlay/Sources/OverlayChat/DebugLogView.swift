import SwiftUI

struct DebugLogView: View {
    @ObservedObject var model: DebugLogModel

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView(.vertical, showsIndicators: false) {
                LazyVStack(alignment: .leading, spacing: 1) {
                    ForEach(model.entries) { entry in
                        HStack(alignment: .top, spacing: 4) {
                            Text(entry.timestamp)
                                .foregroundColor(.gray)
                            Text(levelTag(entry.level))
                                .foregroundColor(levelColor(entry.level))
                                .frame(width: 18, alignment: .center)
                            Text(entry.source)
                                .foregroundColor(Color.white.opacity(0.4))
                                .frame(width: 60, alignment: .leading)
                                .lineLimit(1)
                            Text(entry.message)
                                .foregroundColor(.white)
                                .lineLimit(3)
                        }
                        .font(.system(size: 10, design: .monospaced))
                        .id(entry.id)
                    }
                }
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
            }
            .onChange(of: model.entries.count) { _ in
                if let last = model.entries.last {
                    withAnimation(.none) {
                        proxy.scrollTo(last.id, anchor: .bottom)
                    }
                }
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color.black.opacity(0.88))
    }

    private func levelTag(_ level: String) -> String {
        switch level {
        case "DEBUG":   return "D"
        case "INFO":    return "I"
        case "WARNING": return "W"
        case "ERROR":   return "E"
        default:        return "?"
        }
    }

    private func levelColor(_ level: String) -> Color {
        switch level {
        case "DEBUG":   return .green
        case "INFO":    return .cyan
        case "WARNING": return .yellow
        case "ERROR":   return .red
        default:        return .white
        }
    }
}
