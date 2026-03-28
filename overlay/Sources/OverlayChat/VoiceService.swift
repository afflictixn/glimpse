import AVFoundation
import Combine

final class VoiceService {
    private let synthesizer = AVSpeechSynthesizer()
    private let settings: Settings
    private var cancellables = Set<AnyCancellable>()

    /// Currently resolved voice — updates when settings.selectedVoiceId changes
    private var resolvedVoice: AVSpeechSynthesisVoice?

    init(settings: Settings = .shared) {
        self.settings = settings
        resolvedVoice = Self.resolveVoice(id: settings.selectedVoiceId)

        settings.$selectedVoiceId
            .sink { [weak self] id in
                self?.resolvedVoice = Self.resolveVoice(id: id)
            }
            .store(in: &cancellables)
    }

    func speak(_ text: String) {
        guard settings.voiceEnabled else { return }

        if synthesizer.isSpeaking {
            synthesizer.stopSpeaking(at: .word)
        }

        let utterance = AVSpeechUtterance(string: text)
        utterance.voice = resolvedVoice
        utterance.rate = AVSpeechUtteranceDefaultSpeechRate * Float(settings.voiceRate * 2)
        utterance.volume = settings.voiceVolume
        utterance.pitchMultiplier = 1.0

        synthesizer.speak(utterance)
    }

    func stop() {
        if synthesizer.isSpeaking {
            synthesizer.stopSpeaking(at: .word)
        }
    }

    /// Available English voices for the settings picker
    static var availableVoices: [AVSpeechSynthesisVoice] {
        AVSpeechSynthesisVoice.speechVoices()
            .filter { $0.language.starts(with: "en") }
            .sorted { ($0.quality.rawValue, $0.name) > ($1.quality.rawValue, $1.name) }
    }

    private static func resolveVoice(id: String?) -> AVSpeechSynthesisVoice? {
        let voices = AVSpeechSynthesisVoice.speechVoices()
        if let id, let match = voices.first(where: { $0.identifier == id }) {
            return match
        }
        // Auto-select best available English voice
        return voices.first(where: {
            $0.language.starts(with: "en") && $0.quality == .enhanced
        }) ?? voices.first(where: {
            $0.language.starts(with: "en") && $0.quality == .premium
        }) ?? AVSpeechSynthesisVoice(language: "en-US")
    }
}
