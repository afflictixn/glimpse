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

    enum UIMode: Equatable {
        case ambient          // edge glow + suggestion pills only
        case floatingOverlay  // compact draggable chat bubble
        case fullPanel        // full right-side chat panel with settings
    }

    @Published var mode: AnimationMode = .idle
    @Published var accentColor: Color = Color(hue: 0.6, saturation: 0.5, brightness: 0.9)
    @Published var currentSuggestion: String? = nil
    @Published var uiMode: UIMode = .ambient
    /// Timestamp (timeIntervalSinceReferenceDate) of last mode change — used to animate fog slide in/out
    @Published var modeChangedAt: Double = 0

    let settings = Settings.shared
    let voiceService: VoiceService

    init() {
        self.voiceService = VoiceService(settings: Settings.shared)
    }

    // Animation parameters derived from mode
    var baseIntensity: Double {
        switch mode {
        case .idle:       return 0.02
        case .suggestion: return 0.30
        case .excited:    return 0.40
        case .important:  return 0.55
        case .warning:    return 0.65
        case .hidden:     return 0.0
        }
    }

    var amplitude: Double {
        switch mode {
        case .idle:       return 0.02
        case .suggestion: return 0.20
        case .excited:    return 0.30
        case .important:  return 0.10
        case .warning:    return 0.15
        case .hidden:     return 0.0
        }
    }

    var frequency: Double {
        switch mode {
        case .idle:       return 0.2
        case .suggestion: return 1.0
        case .excited:    return 3.0
        case .important:  return 0.5
        case .warning:    return 2.0
        case .hidden:     return 0.0
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
            // Flat — no pulsing. The shape breathing in EdgeGlowView
            // handles the "alive" feel; opacity stays constant.
            return baseIntensity
        }
    }

    // MARK: - Bridge from WebSocket proposals

    func handleProposal(text: String, importance: ProactiveSuggestion.Importance) {
        currentSuggestion = text
        modeChangedAt = Date().timeIntervalSinceReferenceDate
        switch importance {
        case .low:    mode = .suggestion
        case .medium: mode = .suggestion
        case .high:   mode = .important
        }
        voiceService.speak(text)
    }

    func handleWarning(text: String) {
        currentSuggestion = text
        modeChangedAt = Date().timeIntervalSinceReferenceDate
        mode = .warning
        voiceService.speak(text)
    }

    func dismissSuggestion() {
        currentSuggestion = nil
        modeChangedAt = Date().timeIntervalSinceReferenceDate
        mode = .idle
        voiceService.stop()
    }

    // MARK: - UI Mode Transitions

    func toggleChat() {
        // Cmd+Shift+O cycles: ambient → floating → full → ambient
        switch uiMode {
        case .ambient:        uiMode = .floatingOverlay
        case .floatingOverlay: uiMode = .fullPanel
        case .fullPanel:       uiMode = .ambient
        }
    }

    func openFloatingOverlay() {
        uiMode = .floatingOverlay
    }

    func expandToFullPanel() {
        uiMode = .fullPanel
    }

    func dismissToAmbient() {
        uiMode = .ambient
    }
}
