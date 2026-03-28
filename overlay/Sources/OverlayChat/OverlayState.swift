import SwiftUI
import Combine

final class OverlayState: ObservableObject {
    enum AnimationMode: Equatable {
        case idle           // slow calm pulse
        case suggestion     // gentle warm shimmer
        case excited        // quick double-pulse — "you're going to like this"
        case important      // sustained bright glow
        case warning        // sharp flash + hold
        case hidden         // deep focus — fade to almost nothing
    }

    @Published var mode: AnimationMode = .idle
    @Published var accentColor: Color = Color(hue: 0.6, saturation: 0.5, brightness: 0.9)
    @Published var currentSuggestion: String? = nil
    @Published var showChatPanel: Bool = false

    let voiceService = VoiceService()

    // Animation parameters derived from mode
    var baseIntensity: Double {
        switch mode {
        case .idle:       return 0.25
        case .suggestion: return 0.35
        case .excited:    return 0.4
        case .important:  return 0.6
        case .warning:    return 0.7
        case .hidden:     return 0.05
        }
    }

    var amplitude: Double {
        switch mode {
        case .idle:       return 0.15
        case .suggestion: return 0.2
        case .excited:    return 0.3
        case .important:  return 0.1
        case .warning:    return 0.15
        case .hidden:     return 0.03
        }
    }

    var frequency: Double {
        switch mode {
        case .idle:       return 0.8
        case .suggestion: return 1.2
        case .excited:    return 3.0
        case .important:  return 0.6
        case .warning:    return 2.0
        case .hidden:     return 0.4
        }
    }

    func intensity(at time: Double) -> Double {
        switch mode {
        case .excited:
            let cycle = time.truncatingRemainder(dividingBy: 2.0)
            if cycle < 0.3 {
                return baseIntensity + amplitude * sin(cycle / 0.3 * .pi)
            } else if cycle > 0.4 && cycle < 0.7 {
                return baseIntensity + amplitude * sin((cycle - 0.4) / 0.3 * .pi)
            }
            return baseIntensity
        case .warning:
            let cycle = time.truncatingRemainder(dividingBy: 3.0)
            if cycle < 0.2 {
                return baseIntensity + amplitude * (cycle / 0.2)
            }
            return baseIntensity + amplitude * 0.7
        default:
            return baseIntensity + amplitude * sin(time * frequency * .pi * 2)
        }
    }

    // MARK: - Bridge from WebSocket proposals

    func handleProposal(text: String, importance: ProactiveSuggestion.Importance) {
        currentSuggestion = text
        switch importance {
        case .low:    mode = .suggestion
        case .medium: mode = .suggestion
        case .high:   mode = .important
        }
        voiceService.speak(text)
    }

    func handleWarning(text: String) {
        currentSuggestion = text
        mode = .warning
        voiceService.speak(text)
    }

    func dismissSuggestion() {
        currentSuggestion = nil
        mode = .idle
        voiceService.stop()
    }

    func toggleChat() {
        showChatPanel.toggle()
    }
}
