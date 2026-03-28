import SwiftUI
import Combine

final class Settings: ObservableObject {
    static let shared = Settings()

    @Published var voiceEnabled: Bool {
        didSet {
            UserDefaults.standard.set(voiceEnabled, forKey: "voiceEnabled")
            writeVoiceFile(voiceEnabled)
            syncVoiceToBackend(voiceEnabled)
        }
    }

    private init() {
        let defaults = UserDefaults.standard
        defaults.register(defaults: [
            "voiceEnabled": true,
        ])
        self.voiceEnabled = defaults.bool(forKey: "voiceEnabled")
        writeVoiceFile(self.voiceEnabled)
        syncVoiceToBackend(self.voiceEnabled)
    }

    /// Write voice state to ~/.zexp/voice_enabled — the Python backend reads this
    /// file before every TTS call, so it works regardless of HTTP sync timing.
    private func writeVoiceFile(_ enabled: Bool) {
        let dir = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".zexp")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let file = dir.appendingPathComponent("voice_enabled")
        try? (enabled ? "true" : "false").write(to: file, atomically: true, encoding: .utf8)
    }

    /// Best-effort HTTP sync — belt-and-suspenders alongside the file.
    private func syncVoiceToBackend(_ enabled: Bool) {
        guard let url = URL(string: "http://localhost:3030/agent/voice") else { return }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try? JSONSerialization.data(withJSONObject: ["enabled": enabled])
        URLSession.shared.dataTask(with: request).resume()
    }
}
