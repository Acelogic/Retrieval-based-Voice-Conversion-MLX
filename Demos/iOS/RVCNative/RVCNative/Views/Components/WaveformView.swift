import SwiftUI

/// Displays an audio waveform visualization with optional playback progress indicator.
struct WaveformView: View {
    let samples: [Float]
    let color: Color
    let label: String
    var progress: Double? = nil // 0.0 to 1.0

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(label)
                    .font(.caption)
                    .foregroundColor(.secondary)
                Spacer()
                if let p = progress {
                    Text("\(Int(p * 100))%")
                        .font(.caption2)
                        .monospacedDigit()
                        .foregroundColor(.cyan.opacity(0.8))
                }
            }

            GeometryReader { geometry in
                let width = geometry.size.width
                let height = geometry.size.height
                let midY = height / 2

                ZStack(alignment: .leading) {
                    Path { path in
                        guard !samples.isEmpty else { return }

                        let pointCount = Int(width)
                        let samplesPerPoint = max(1, samples.count / pointCount)

                        for i in 0..<pointCount {
                            let startIdx = i * samplesPerPoint
                            let endIdx = min(startIdx + samplesPerPoint, samples.count)

                            guard startIdx < samples.count else { break }

                            let segment = samples[startIdx..<endIdx]
                            let minVal = segment.min() ?? 0
                            let maxVal = segment.max() ?? 0

                            let x = CGFloat(i)
                            let topY = midY - CGFloat(maxVal) * midY * 0.9
                            let bottomY = midY - CGFloat(minVal) * midY * 0.9

                            path.move(to: CGPoint(x: x, y: topY))
                            path.addLine(to: CGPoint(x: x, y: bottomY))
                        }
                    }
                    .stroke(color, lineWidth: 1)

                    // Center line
                    Path { path in
                        path.move(to: CGPoint(x: 0, y: midY))
                        path.addLine(to: CGPoint(x: width, y: midY))
                    }
                    .stroke(color.opacity(0.3), lineWidth: 0.5)

                    // Scrubber Line
                    if let p = progress {
                        Rectangle()
                            .fill(LinearGradient(colors: [.cyan, .cyan.opacity(0)], startPoint: .top, endPoint: .bottom))
                            .frame(width: 2)
                            .offset(x: CGFloat(p) * width - 1)
                            .shadow(color: .cyan.opacity(0.5), radius: 2)
                    }
                }
            }
        }
        .frame(height: 60)
    }
}

/// Side-by-side comparison of original and converted waveforms with seek support.
struct WaveformComparisonView: View {
    let originalSamples: [Float]
    let convertedSamples: [Float]
    var progress: Double? = nil
    var onSeek: ((Double) -> Void)? = nil

    var body: some View {
        VStack(spacing: 12) {
            Text("Waveform Comparison")
                .font(.headline)
                .foregroundStyle(.primary.opacity(0.9))

            GeometryReader { geometry in
                VStack(spacing: 8) {
                    WaveformView(samples: originalSamples, color: .blue, label: "Original", progress: progress)
                    WaveformView(samples: convertedSamples, color: .green, label: "Converted", progress: progress)
                }
                .contentShape(Rectangle())
                .gesture(
                    DragGesture(minimumDistance: 0)
                        .onChanged { value in
                            let percentage = max(0, min(1, value.location.x / geometry.size.width))
                            onSeek?(Double(percentage))
                        }
                )
            }
            .frame(height: 128) // Fixed height for the waveforms + labels
        }
        .padding()
        .background(.primary.opacity(0.05))
        .cornerRadius(12)
    }
}
