import SwiftUI
import Combine

final class Settings: ObservableObject {
    static let shared = Settings()

    @Published var voiceEnabled: Bool {
        didSet {
            UserDefaults.standard.set(voiceEnabled, forKey: "voiceEnabled")
            syncVoiceToBackend(voiceEnabled)
        }
    }

    private init() {
        let defaults = UserDefaults.standard
        defaults.register(defaults: [
            "voiceEnabled": true,
        ])
        self.voiceEnabled = defaults.bool(forKey: "voiceEnabled")
        // Sync initial state to backend on launch
        syncVoiceToBackend(self.voiceEnabled)
    }

    private func syncVoiceToBackend(_ enabled: Bool) {
        guard let url = URL(string: "http://localhost:3030/agent/voice") else { return }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try? JSONSerialization.data(withJSONObject: ["enabled": enabled])
        URLSession.shared.dataTask(with: request).resume()
    }
}
