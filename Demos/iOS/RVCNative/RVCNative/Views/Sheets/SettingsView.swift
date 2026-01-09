import SwiftUI

/// App settings sheet with appearance and configuration options.
struct SettingsView: View {
    @Environment(\.dismiss) private var dismiss

    @AppStorage("accentTheme") private var accentTheme: AccentTheme = .system
    @AppStorage("isHighPrecision") private var isHighPrecision: Bool = true
    @AppStorage("isExperimental") private var isExperimental: Bool = false
    @AppStorage("autoCleanup") private var autoCleanup: Bool = false
    @AppStorage("showDebugLogs") private var showDebugLogs: Bool = true

    @Environment(\.colorScheme) private var colorScheme

    private var accentColor: Color {
        accentTheme.color(for: colorScheme)
    }

    var body: some View {
        NavigationStack {
            ZStack {
                backgroundGradient

                ScrollView {
                    VStack(spacing: 24) {
                        // Profile/About Section
                        VStack(spacing: 16) {
                            ZStack {
                                Circle()
                                    .fill(accentColor.opacity(0.15))
                                    .frame(width: 80, height: 80)
                                Image(systemName: "person.fill")
                                    .font(.system(size: 40))
                                    .foregroundStyle(accentColor)
                            }

                            VStack(spacing: 4) {
                                Text("RVC Native")
                                    .font(.title2.bold())
                                    .foregroundStyle(.primary)
                                Text("Version 1.0.0 (Build 2390)")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                        }
                        .padding(.top, 40)

                        // Theme Selection
                        settingsGroup(title: "Appearance") {
                            VStack(alignment: .leading, spacing: 16) {
                                Text("Accent Color")
                                    .font(.subheadline.bold())
                                    .foregroundStyle(.primary)

                                ScrollView(.horizontal, showsIndicators: false) {
                                    HStack(spacing: 16) {
                                        ForEach(AccentTheme.allCases) { theme in
                                            Button(action: { accentTheme = theme }) {
                                                VStack(spacing: 8) {
                                                    ZStack {
                                                        Circle()
                                                            .fill(theme.color(for: colorScheme))
                                                            .frame(width: 44, height: 44)

                                                        if accentTheme == theme {
                                                            Image(systemName: "checkmark")
                                                                .font(.system(size: 20, weight: .bold))
                                                                .foregroundStyle(.white)
                                                        }
                                                    }
                                                    Text(theme.rawValue)
                                                        .font(.caption2)
                                                        .foregroundStyle(accentTheme == theme ? .primary : .secondary)
                                                }
                                            }
                                        }
                                    }
                                    .padding(.horizontal, 4)
                                }
                            }
                            .padding()
                        }

                        // Settings Groups
                        VStack(spacing: 16) {
                            settingsGroup(title: "Inference Engine") {
                                settingsToggle(icon: "cpu", title: "High Precision (FP32)", isOn: $isHighPrecision)
                                settingsToggle(icon: "bolt.horizontal.fill", title: "Experimental Optimizations", isOn: $isExperimental)
                                if isExperimental {
                                    settingsRow(icon: "waveform.path.ecg", title: "Pitch Method", detail: "RMVPE")
                                }
                            }

                            settingsGroup(title: "Storage & Logs") {
                                settingsToggle(icon: "clock.arrow.2.circlepath", title: "Auto-Cleanup History", isOn: $autoCleanup)
                                settingsToggle(icon: "terminal", title: "Show System Logs", isOn: $showDebugLogs)
                            }

                            settingsGroup(title: "Links") {
                                Link(destination: URL(string: "https://github.com/Acelogic/Retrieval-based-Voice-Conversion-MLX/blob/main/Demos/iOS/RVCNative/PRIVACY_POLICY.md")!) {
                                    settingsRow(icon: "doc.text", title: "Privacy Policy", detail: "")
                                }

                                Link(destination: URL(string: "https://github.com/Acelogic/Retrieval-based-Voice-Conversion-MLX")!) {
                                    settingsRow(icon: "link", title: "Source Code", detail: "GitHub")
                                }
                            }
                        }
                        .padding(.horizontal)
                        .padding(.bottom, 40)
                    }
                }
            }
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") {
                        dismiss()
                    }
                    .bold()
                    .foregroundStyle(accentColor)
                }
            }
        }
    }

    private func settingsGroup<Content: View>(title: String, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(title.uppercased())
                .font(.caption2.bold())
                .tracking(1)
                .foregroundStyle(accentColor.opacity(0.8))
                .padding(.leading, 8)

            VStack(spacing: 0) {
                content()
            }
            .glassEffect(in: RoundedRectangle(cornerRadius: 16))
        }
    }

    private func settingsRow(icon: String, title: String, detail: String, color: Color? = nil) -> some View {
        HStack {
            Image(systemName: icon)
                .foregroundStyle(color ?? accentColor)
                .frame(width: 24)
            Text(title)
                .foregroundStyle(.primary)
            Spacer()
            Text(detail)
                .font(.caption)
                .foregroundStyle(.secondary)
            Image(systemName: "chevron.right")
                .font(.caption2)
                .foregroundStyle(.primary.opacity(0.5))
        }
        .padding()
        .contentShape(Rectangle())
    }

    private func settingsToggle(icon: String, title: String, isOn: Binding<Bool>) -> some View {
        HStack {
            Image(systemName: icon)
                .foregroundStyle(accentColor)
                .frame(width: 24)
            Toggle(title, isOn: isOn)
                .foregroundStyle(.primary)
                .tint(accentColor)
        }
        .padding()
    }

    // Use the same background gradient as the main view
    private var backgroundGradient: some View {
        ZStack {
            Color(uiColor: .systemBackground)

            LinearGradient(
                colors: [
                    accentColor.opacity(0.1),
                    Color.blue.opacity(0.15),
                    Color.purple.opacity(0.1)
                ],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
        }
        .ignoresSafeArea()
    }
}
