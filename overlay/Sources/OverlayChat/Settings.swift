import SwiftUI
import Combine

final class Settings: ObservableObject {
    static let shared = Settings()

    @Published var voiceEnabled: Bool {
        didSet { UserDefaults.standard.set(voiceEnabled, forKey: "voiceEnabled") }
    }
    @Published var voiceVolume: Float {
        didSet { UserDefaults.standard.set(voiceVolume, forKey: "voiceVolume") }
    }
    @Published var voiceRate: Float {
        didSet { UserDefaults.standard.set(voiceRate, forKey: "voiceRate") }
    }
    @Published var selectedVoiceId: String? {
        didSet { UserDefaults.standard.set(selectedVoiceId, forKey: "selectedVoiceId") }
    }

    private init() {
        let defaults = UserDefaults.standard

        // Register defaults on first launch
        defaults.register(defaults: [
            "voiceEnabled": true,
            "voiceVolume": Float(0.6),
            "voiceRate": Float(0.5),
        ])

        self.voiceEnabled = defaults.bool(forKey: "voiceEnabled")
        self.voiceVolume = defaults.float(forKey: "voiceVolume")
        self.voiceRate = defaults.float(forKey: "voiceRate")
        self.selectedVoiceId = defaults.string(forKey: "selectedVoiceId")
    }
}
