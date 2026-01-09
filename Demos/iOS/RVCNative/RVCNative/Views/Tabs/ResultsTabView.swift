import SwiftUI
import UIKit
import RVCNativeFeature

/// Results tab displaying conversion history and detailed result view.
struct ResultsTabView: View {
    // State bindings
    @Binding var conversionHistory: [ConversionResult]
    @Binding var selectedResult: ConversionResult?
    @Binding var showClearAllAlert: Bool

    // Audio player for playback
    @ObservedObject var outputPlayer: AudioPlayer

    // Callbacks
    var onDeleteResult: (ConversionResult) -> Void

    // Theme
    @AppStorage("accentTheme") private var accentTheme: AccentTheme = .system
    @Environment(\.colorScheme) private var colorScheme

    private var accentColor: Color {
        accentTheme.color(for: colorScheme)
    }

    var body: some View {
        VStack(spacing: 0) {
            headerView.padding(.horizontal)

            if conversionHistory.isEmpty {
                emptyStateView
            } else {
                historyListView
            }
        }
        .background(backgroundGradient)
    }

    // MARK: - Header

    private var headerView: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text("Results")
                    .font(.title2)
                    .bold()
                Text("Your conversion history")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }
            Spacer()
        }
        .padding(.vertical, 12)
    }

    // MARK: - Empty State

    private var emptyStateView: some View {
        VStack(spacing: 20) {
            Spacer()
            Image(systemName: "waveform.badge.magnifyingglass")
                .font(.system(size: 60))
                .foregroundStyle(.primary.opacity(0.3))
            Text("No conversions yet")
                .font(.headline)
                .foregroundStyle(.primary.opacity(0.5))
            Text("Go to the Lab to start your first RVC conversion.")
                .font(.subheadline)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 40)
                .foregroundStyle(.primary.opacity(0.3))
            Spacer()
        }
    }

    // MARK: - History List

    private var historyListView: some View {
        ScrollView {
            VStack(spacing: 24) {
                if let selected = selectedResult {
                    detailedResultView(for: selected)
                        .transition(.move(edge: .top).combined(with: .opacity))
                }

                Divider().overlay(.primary.opacity(0.1))

                VStack(alignment: .leading, spacing: 16) {
                    HStack {
                        Text("History")
                            .font(.subheadline)
                            .bold()
                            .foregroundStyle(.primary.opacity(0.8))
                        Spacer()
                        Button(action: { showClearAllAlert = true }) {
                            Text("Clear All")
                                .font(.caption)
                                .bold()
                                .foregroundStyle(.red.opacity(0.8))
                        }
                    }

                    ForEach(conversionHistory) { result in
                        HistoryRow(result: result, isSelected: selectedResult?.id == result.id) {
                            withAnimation {
                                selectedResult = result
                            }
                        } onDelete: {
                            onDeleteResult(result)
                        }
                    }
                }
            }
            .padding()
        }
    }

    // MARK: - Detailed Result View

    private func detailedResultView(for result: ConversionResult) -> some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                VStack(alignment: .leading) {
                    Text("Selected Conversion")
                        .font(.caption2)
                        .bold()
                        .foregroundStyle(accentColor)
                    HStack(spacing: 4) {
                        Text(result.sourceAudioName)
                            .lineLimit(1)
                        Image(systemName: "arrow.right")
                        Text(result.modelName)
                            .lineLimit(1)
                    }
                    .font(.title3)
                    .bold()
                    .foregroundStyle(.primary)
                }
                Spacer()
                Text(result.date, style: .time)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            // Comparison Waveform
            WaveformComparisonView(
                originalSamples: result.originalWaveform,
                convertedSamples: result.convertedWaveform,
                progress: (outputPlayer.isPlaying && outputPlayer.duration > 0) ? outputPlayer.currentTime / outputPlayer.duration : 0,
                onSeek: { percentage in
                    outputPlayer.seek(to: percentage * outputPlayer.duration)
                }
            )
            .glassEffect(in: RoundedRectangle(cornerRadius: 12))
            .frame(height: 180)

            HStack(spacing: 12) {
                Button(action: {
                    if outputPlayer.isPlaying {
                        outputPlayer.stop()
                    } else {
                        outputPlayer.play(url: result.outputURL)
                    }
                }) {
                    HStack {
                        Image(systemName: outputPlayer.isPlaying ? "stop.fill" : "play.fill")
                        Text(outputPlayer.isPlaying ? "Stop" : "Play Result")
                    }
                    .font(.headline)
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.green.opacity(0.3))
                    .glassEffect(in: RoundedRectangle(cornerRadius: 12))
                    .foregroundColor(.green)
                }

                Button(action: {
                    // Share logic
                    let activityVC = UIActivityViewController(activityItems: [result.outputURL], applicationActivities: nil)
                    if let windowScene = UIApplication.shared.connectedScenes.first as? UIWindowScene,
                       let rootVC = windowScene.windows.first?.rootViewController {
                        rootVC.present(activityVC, animated: true)
                    }
                }) {
                    Image(systemName: "square.and.arrow.up")
                        .font(.headline)
                        .padding()
                        .background(Color.white.opacity(0.1))
                        .glassEffect(in: RoundedRectangle(cornerRadius: 12))
                        .foregroundColor(.primary)
                }
            }
        }
        .padding()
        .glassEffect(in: RoundedRectangle(cornerRadius: 16))
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
