import Foundation
import AVFoundation

/// Extracts waveform sample data from audio files for visualization.
struct AudioWaveformExtractor {
    /// Extract downsampled audio samples from a URL for waveform visualization.
    /// - Parameters:
    ///   - url: The audio file URL to extract samples from
    ///   - maxSamples: Maximum number of samples to return (default: 4000)
    /// - Returns: Array of float samples normalized to [-1, 1] range
    static func extractSamples(from url: URL, maxSamples: Int = 4000) -> [Float] {
        guard let file = try? AVAudioFile(forReading: url) else {
            return []
        }

        let format = file.processingFormat
        let frameCount = AVAudioFrameCount(file.length)

        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
            return []
        }

        do {
            try file.read(into: buffer)
        } catch {
            return []
        }

        guard let channelData = buffer.floatChannelData else {
            return []
        }

        let samples = Array(UnsafeBufferPointer(start: channelData[0], count: Int(buffer.frameLength)))

        // Downsample if too many samples
        if samples.count <= maxSamples {
            return samples
        }

        let step = samples.count / maxSamples
        return stride(from: 0, to: samples.count, by: step).map { samples[$0] }
    }
}
