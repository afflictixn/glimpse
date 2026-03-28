import SwiftUI

struct SettingsView: View {
    @ObservedObject var settings: Settings
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            header
            Divider().overlay(Color.white.opacity(0.15))

            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    voiceSection
                }
                .padding(16)
            }
        }
        .background(
            ZStack {
                Color.black.opacity(0.6)
                Color(hue: 0.6, saturation: 0.5, brightness: 0.9).opacity(0.05)
            }
        )
    }

    // MARK: - Header

    private var header: some View {
        HStack {
            Text("Settings")
                .font(.system(size: 13, weight: .semibold, design: .monospaced))
                .foregroundColor(.white)
            Spacer()
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
    }

    // MARK: - Voice

    private var voiceSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            sectionLabel("Voice (ElevenLabs)")

            Toggle("Enabled", isOn: $settings.voiceEnabled)
                .toggleStyle(.switch)
                .font(.system(size: 12))
                .foregroundColor(.white.opacity(0.85))

            if settings.voiceEnabled {
                Text("Using ElevenLabs TTS via backend")
                    .font(.system(size: 11))
                    .foregroundColor(.white.opacity(0.4))
            }
        }
    }

    // MARK: - Helpers

    private func sectionLabel(_ title: String) -> some View {
        Text(title.uppercased())
            .font(.system(size: 10, weight: .bold, design: .monospaced))
            .foregroundColor(.white.opacity(0.4))
            .tracking(1.2)
    }
}
