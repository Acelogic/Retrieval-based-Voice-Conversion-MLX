import SwiftUI
import RVCNativeFeature

/// Lab tab for audio input, recording, and voice conversion.
struct LabTabView: View {
    // Audio state bindings
    @Binding var inputURL: URL?
    @Binding var activeImport: ContentView.ImportType?

    // Advanced settings bindings
    @Binding var volumeEnvelope: Float
    @Binding var selectedF0Method: String
    @Binding var pitchShift: Double
    @Binding var featureRatio: Double
    @Binding var showAdvancedSettings: Bool

    // Recording state bindings
    @Binding var recordingTime: TimeInterval
    @Binding var recordingTimer: Timer?
    @Binding var justFinishedRecording: Bool

    // Conversion state bindings
    @Binding var isProcessing: Bool
    @Binding var isConverting: Bool
    @Binding var conversionProgress: Double
    @Binding var statusMessage: String
    @Binding var isModelLoaded: Bool

    // Animation state
    @Binding var phase: Double

    // Audio players and recorder
    @ObservedObject var audioRecorder: AudioRecorder
    @ObservedObject var inputPlayer: AudioPlayer

    // Stock audio URL
    let stockAudioURL: URL?

    // Callbacks
    var onStartInference: () async -> Void

    // Theme
    @AppStorage("accentTheme") private var accentTheme: AccentTheme = .system
    @Environment(\.colorScheme) private var colorScheme

    private var accentColor: Color {
        accentTheme.color(for: colorScheme)
    }

    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                headerView
                volumeControlView
                advancedSettingsView
                audioInputControlView
                conversionProgressView
                inferenceButtonView
            }
            .padding()
        }
        .background(backgroundGradient)
    }

    // MARK: - Header

    private var headerView: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text("Audio Lab")
                    .font(.title2)
                    .bold()
                Text("Convert your voice")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }
            Spacer()
        }
    }

    // MARK: - Volume Control

    private var volumeControlView: some View {
        VStack(alignment: .leading, spacing: 5) {
            HStack {
                Text("Volume Envelope (Reduce Noise)")
                    .font(.caption)
                    .foregroundStyle(.primary.opacity(0.8))
                Spacer()
                Text(String(format: "%.2f", volumeEnvelope))
                    .font(.caption)
                    .monospacedDigit()
                    .foregroundStyle(.primary)
            }
            Slider(value: $volumeEnvelope, in: 0.0...1.0, step: 0.01)
                .tint(accentColor)
        }
        .padding()
        .glassEffect(in: RoundedRectangle(cornerRadius: 16))
    }

    // MARK: - Advanced Settings

    private var advancedSettingsView: some View {
        VStack(spacing: 0) {
            DisclosureGroup(isExpanded: $showAdvancedSettings) {
                VStack(spacing: 16) {
                    Divider()
                        .overlay(.primary.opacity(0.1))
                        .padding(.vertical, 8)

                    // F0 Method
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Pitch Extraction")
                            .font(.caption2.bold())
                            .foregroundStyle(.secondary)

                        ScrollView(.horizontal, showsIndicators: false) {
                            HStack(spacing: 8) {
                                ForEach(["rmvpe", "dio", "pm", "harvest", "crepe", "crepe-tiny"], id: \.self) { method in
                                    Button(action: { selectedF0Method = method }) {
                                        Text(method.uppercased())
                                            .font(.caption2.bold())
                                            .foregroundStyle(selectedF0Method == method ? .white : .primary.opacity(0.6))
                                            .padding(.horizontal, 12)
                                            .padding(.vertical, 6)
                                            .background(selectedF0Method == method ? accentColor : Color.primary.opacity(0.08))
                                            .clipShape(Capsule())
                                    }
                                    .buttonStyle(.plain)
                                }
                            }
                        }
                    }

                    // Pitch Shift
                    VStack(spacing: 4) {
                        HStack {
                            Text("Pitch Shift")
                                .font(.caption2.bold())
                                .foregroundStyle(.secondary)
                            Spacer()
                            Text(pitchShift >= 0 ? "+\(Int(pitchShift))" : "\(Int(pitchShift))")
                                .font(.caption2)
                                .monospacedDigit()
                                .foregroundStyle(.primary.opacity(0.7))
                        }
                        Slider(value: $pitchShift, in: -12...12, step: 1.0)
                            .tint(accentColor)
                    }

                    // Feature Ratio
                    VStack(spacing: 4) {
                        HStack {
                            Text("Feature Retrieval")
                                .font(.caption2.bold())
                                .foregroundStyle(.secondary)
                            Spacer()
                            Text(String(format: "%.2f", featureRatio))
                                .font(.caption2)
                                .monospacedDigit()
                                .foregroundStyle(.primary.opacity(0.7))
                        }
                        Slider(value: $featureRatio, in: 0.0...1.0, step: 0.01)
                            .tint(accentColor)
                    }
                }
                .padding(.top, 4)
            } label: {
                HStack {
                    Image(systemName: "slider.horizontal.3")
                        .foregroundStyle(accentColor)
                    Text("Advanced Settings")
                        .font(.caption)
                        .foregroundStyle(.primary.opacity(0.8))
                    Spacer()
                }
            }
        }
        .padding()
        .glassEffect(in: RoundedRectangle(cornerRadius: 16))
    }

    // MARK: - Audio Input Controls

    @ViewBuilder
    private var audioInputControlView: some View {
        Group {
            // Select Audio File Button
            Button(action: {
                activeImport = .audio
            }) {
                HStack {
                    Image(systemName: "waveform.circle")
                    Text(inputURL == nil ? "Select Audio File" : "Selected: \(inputURL?.lastPathComponent ?? "File")")
                        .lineLimit(1)
                }
                .font(.headline)
                .frame(maxWidth: .infinity)
                .padding()
                .glassEffect(in: Capsule())
                .foregroundColor(accentColor)
                .overlay(
                    Capsule()
                        .stroke(accentColor.opacity(0.3), lineWidth: 1)
                )
            }

            // Audio Loaded Indicator
            if let input = inputURL {
                HStack {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundColor(.green)
                    Text("Ready to convert: \(input.lastPathComponent)")
                        .font(.caption)
                        .foregroundColor(.white.opacity(0.7))
                }
            } else if let stock = stockAudioURL {
                Text("Using stock audio: \(stock.lastPathComponent)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            // Play Original
            if let url = inputURL {
                Button(action: {
                    if inputPlayer.isPlaying {
                        inputPlayer.stop()
                    } else {
                        inputPlayer.play(url: url)
                    }
                }) {
                    HStack {
                        Image(systemName: inputPlayer.isPlaying ? "stop.fill" : "play.fill")
                        Text(inputPlayer.isPlaying ? "Stop Original" : "Play Original")
                    }
                    .font(.subheadline)
                    .foregroundColor(.primary)
                    .padding(.vertical, 4)
                    .padding(.horizontal, 12)
                    .glassEffect(in: Capsule())
                }
            }

            // Audio Recording UI
            audioRecordingView
        }
    }

    // MARK: - Audio Recording View

    @ViewBuilder
    private var audioRecordingView: some View {
        if audioRecorder.isRecording {
            // While Recording - Animated
            VStack(spacing: 16) {
                // Pulsing Red Circle with Timer
                ZStack {
                    Circle()
                        .fill(Color.red.opacity(0.2))
                        .frame(width: 120, height: 120)
                        .scaleEffect(phase)
                        .opacity(2 - phase)

                    Circle()
                        .fill(Color.red.opacity(0.3))
                        .frame(width: 90, height: 90)
                        .scaleEffect(1 + (phase * 0.5))

                    Circle()
                        .fill(Color.red)
                        .frame(width: 70, height: 70)

                    Image(systemName: "waveform")
                        .font(.system(size: 28))
                        .foregroundStyle(.white)
                }
                .onAppear {
                    withAnimation(.easeInOut(duration: 1.5).repeatForever(autoreverses: true)) {
                        phase = 1.5
                    }
                }

                // Recording Time
                Text(formatTime(recordingTime))
                    .font(.system(size: 40, weight: .thin, design: .rounded))
                    .monospacedDigit()
                    .foregroundStyle(.primary)

                Text("Recording...")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .opacity(phase > 1.2 ? 1.0 : 0.5)

                // Stop Button
                Button(action: {
                    audioRecorder.stopRecording()
                    recordingTimer?.invalidate()
                    justFinishedRecording = true
                    DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
                        justFinishedRecording = false
                    }
                }) {
                    HStack {
                        Image(systemName: "stop.circle.fill")
                        Text("Stop Recording")
                    }
                    .font(.headline)
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.red.opacity(0.5))
                    .glassEffect(in: Capsule())
                    .foregroundColor(.primary)
                }
            }
            .padding()
            .onAppear {
                recordingTime = 0
                recordingTimer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { _ in
                    recordingTime += 0.1
                }
            }
            .transition(.opacity.combined(with: .scale))
        } else if justFinishedRecording {
            // Just Finished Recording - Success Animation
            VStack(spacing: 16) {
                ZStack {
                    Circle()
                        .fill(Color.green.opacity(0.2))
                        .frame(width: 100, height: 100)

                    Circle()
                        .fill(Color.green)
                        .frame(width: 70, height: 70)

                    Image(systemName: "checkmark")
                        .font(.system(size: 32, weight: .bold))
                        .foregroundStyle(.white)
                }
                .transition(.scale.combined(with: .opacity))

                Text("Recording Complete!")
                    .font(.headline)
                    .foregroundStyle(.primary)

                Text(formatTime(recordingTime))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .padding()
            .transition(.opacity.combined(with: .move(edge: .bottom)))
        } else {
            // Not Recording - Record Button
            Button(action: {
                audioRecorder.startRecording()
            }) {
                HStack {
                    Image(systemName: "mic.circle.fill")
                    Text("Record Audio")
                }
                .font(.headline)
                .frame(maxWidth: .infinity)
                .padding()
                .glassEffect(in: Capsule())
                .foregroundColor(.red.opacity(0.8))
            }
        }
    }

    // MARK: - Conversion Progress

    private var conversionProgressView: some View {
        Group {
            if isConverting {
                VStack(spacing: 8) {
                    ProgressView(value: conversionProgress, total: 1.0)
                        .progressViewStyle(LinearProgressViewStyle())
                        .tint(accentColor)
                    Text("\(Int(conversionProgress * 100))% - \(statusMessage)")
                        .font(.caption)
                        .foregroundStyle(.primary.opacity(0.7))
                }
                .padding()
                .glassEffect(in: RoundedRectangle(cornerRadius: 8))
            }
        }
    }

    // MARK: - Inference Button

    private var inferenceButtonView: some View {
        Button(action: {
            Task {
                await onStartInference()
            }
        }) {
            HStack {
                if isProcessing {
                    ProgressView()
                        .progressViewStyle(CircularProgressViewStyle(tint: .white))
                } else {
                    Image(systemName: "bolt.fill")
                }
                Text(isProcessing ? "Inferencing..." : "Convert Voice")
            }
            .font(.headline)
            .frame(maxWidth: .infinity)
            .padding()
            .background(isProcessing || !isModelLoaded ? Color.gray.opacity(0.3) : Color.green.opacity(0.6))
            .glassEffect(in: Capsule())
            .foregroundColor(.primary)
            .shadow(color: .green.opacity(0.4), radius: 10, x: 0, y: 5)
        }
        .disabled(isProcessing || inputURL == nil || !isModelLoaded)
    }

    // MARK: - Helpers

    private func formatTime(_ time: TimeInterval) -> String {
        let minutes = Int(time) / 60
        let seconds = Int(time) % 60
        let tenths = Int((time * 10).truncatingRemainder(dividingBy: 10))
        return String(format: "%02d:%02d.%d", minutes, seconds, tenths)
    }

    // MARK: - Background

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
