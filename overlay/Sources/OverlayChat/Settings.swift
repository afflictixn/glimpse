import SwiftUI
import Combine

final class Settings: ObservableObject {
    static let shared = Settings()

    @Published var voiceEnabled: Bool {
        didSet { UserDefaults.standard.set(voiceEnabled, forKey: "voiceEnabled") }
    }

    private init() {
        let defaults = UserDefaults.standard
        defaults.register(defaults: [
            "voiceEnabled": true,
        ])
        self.voiceEnabled = defaults.bool(forKey: "voiceEnabled")
    }
}
