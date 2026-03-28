import SwiftUI
import AVFoundation

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
        .background(VisualEffectBackground(material: .hudWindow))
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
            sectionLabel("Voice")

            Toggle("Enabled", isOn: $settings.voiceEnabled)
                .toggleStyle(.switch)
                .font(.system(size: 12))
                .foregroundColor(.white.opacity(0.85))

            if settings.voiceEnabled {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Volume")
                        .font(.system(size: 11))
                        .foregroundColor(.white.opacity(0.5))
                    Slider(value: $settings.voiceVolume, in: 0...1)
                        .controlSize(.small)
                }

                VStack(alignment: .leading, spacing: 4) {
                    Text("Speed")
                        .font(.system(size: 11))
                        .foregroundColor(.white.opacity(0.5))
                    Slider(value: $settings.voiceRate, in: 0...1)
                        .controlSize(.small)
                }

                VStack(alignment: .leading, spacing: 4) {
                    Text("Voice")
                        .font(.system(size: 11))
                        .foregroundColor(.white.opacity(0.5))
                    voicePicker
                }
            }
        }
    }

    private var voicePicker: some View {
        Picker("", selection: Binding(
            get: { settings.selectedVoiceId ?? "" },
            set: { settings.selectedVoiceId = $0.isEmpty ? nil : $0 }
        )) {
            Text("Auto (best available)")
                .tag("")
            ForEach(VoiceService.availableVoices, id: \.identifier) { voice in
                Text("\(voice.name) (\(qualityLabel(voice.quality)))")
                    .tag(voice.identifier)
            }
        }
        .labelsHidden()
        .font(.system(size: 11))
    }

    // MARK: - Helpers

    private func sectionLabel(_ title: String) -> some View {
        Text(title.uppercased())
            .font(.system(size: 10, weight: .bold, design: .monospaced))
            .foregroundColor(.white.opacity(0.4))
            .tracking(1.2)
    }

    private func qualityLabel(_ quality: AVSpeechSynthesisVoiceQuality) -> String {
        switch quality {
        case .enhanced: return "enhanced"
        case .premium: return "premium"
        default: return "default"
        }
    }
}
