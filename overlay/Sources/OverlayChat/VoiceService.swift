import Foundation
import Combine

/// Calls the Python backend's /agent/speak endpoint which uses ElevenLabs TTS.
/// Audio playback happens server-side via afplay.
final class VoiceService {
    private let settings: Settings
    private let session = URLSession.shared
    private let backendBase: String

    init(settings: Settings = .shared, backendBase: String = "http://localhost:3030") {
        self.settings = settings
        self.backendBase = backendBase
    }

    func speak(_ text: String) {
        guard settings.voiceEnabled else { return }
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }

        guard let url = URL(string: "\(backendBase)/agent/speak") else { return }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        let body: [String: String] = ["text": text]
        request.httpBody = try? JSONSerialization.data(withJSONObject: body)

        session.dataTask(with: request) { _, response, error in
            if let error {
                print("[VoiceService] ElevenLabs speak failed: \(error.localizedDescription)")
            } else if let http = response as? HTTPURLResponse, http.statusCode != 200 {
                print("[VoiceService] ElevenLabs speak returned \(http.statusCode)")
            }
        }.resume()
    }

    func stop() {
        // Playback is server-side — nothing to stop locally
    }

    // Keep for settings picker compatibility (no-op list since we use ElevenLabs now)
    static var availableVoices: [String] { [] }
}
