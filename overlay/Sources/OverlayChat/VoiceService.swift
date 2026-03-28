import AVFoundation

final class VoiceService {
    private let synthesizer = AVSpeechSynthesizer()
    private var preferredVoice: AVSpeechSynthesisVoice?

    init() {
        // Try to find a good quality voice
        let voices = AVSpeechSynthesisVoice.speechVoices()
        // Prefer enhanced/premium voices
        preferredVoice = voices.first(where: {
            $0.language.starts(with: "en") && $0.quality == .enhanced
        }) ?? voices.first(where: {
            $0.language.starts(with: "en") && $0.quality == .premium
        }) ?? AVSpeechSynthesisVoice(language: "en-US")
    }

    func speak(_ text: String) {
        // Stop any current speech
        if synthesizer.isSpeaking {
            synthesizer.stopSpeaking(at: .word)
        }

        let utterance = AVSpeechUtterance(string: text)
        utterance.voice = preferredVoice
        utterance.rate = AVSpeechUtteranceDefaultSpeechRate * 0.95
        utterance.volume = 0.6
        utterance.pitchMultiplier = 1.0

        synthesizer.speak(utterance)
    }

    func stop() {
        if synthesizer.isSpeaking {
            synthesizer.stopSpeaking(at: .word)
        }
    }
}
