import Foundation
import Accelerate
import MLX

// MARK: - DSP-Based Pitch Extraction

/// DSP-based pitch extraction methods (DIO, PM, HARVEST approximations).
///
/// These are simplified implementations of traditional DSP pitch estimation
/// algorithms. For best results, use RMVPE (neural network method).
///
/// Methods:
/// - DIO: Autocorrelation-based pitch estimation (simplified)
/// - PM: DIO with post-processing refinement
/// - HARVEST: Higher accuracy autocorrelation (slower)
public class DSPPitchExtractor {

    private let sampleRate: Int
    private let hopSize: Int
    private let frameSize: Int

    /// Initialize the DSP pitch extractor.
    ///
    /// - Parameters:
    ///   - sampleRate: Audio sample rate (default: 16000 Hz)
    ///   - hopSize: Hop size in samples (default: 160 = 10ms at 16kHz)
    ///   - frameSize: Analysis frame size (default: 1024 = 64ms at 16kHz)
    public init(sampleRate: Int = 16000, hopSize: Int = 160, frameSize: Int = 1024) {
        self.sampleRate = sampleRate
        self.hopSize = hopSize
        self.frameSize = frameSize
    }

    // MARK: - Public Methods

    /// DIO-style pitch estimation using autocorrelation.
    ///
    /// - Parameters:
    ///   - audio: Audio signal as MLXArray (1D)
    ///   - f0Min: Minimum F0 to detect (Hz)
    ///   - f0Max: Maximum F0 to detect (Hz)
    /// - Returns: F0 contour with shape (1, T, 1) in Hz
    public func dio(audio: MLXArray, f0Min: Float = 50.0, f0Max: Float = 1100.0) -> MLXArray {
        // Convert to Swift array
        MLX.eval(audio)
        let audioData = audio.asArray(Float.self)

        // Extract F0 using autocorrelation
        let f0 = extractF0Autocorrelation(
            audio: audioData,
            f0Min: f0Min,
            f0Max: f0Max,
            refinement: false
        )

        // Convert to MLXArray with shape (1, T, 1)
        let f0Array = MLXArray(f0)
        return f0Array.reshaped([1, f0.count, 1])
    }

    /// PM-style pitch estimation (DIO + refinement).
    ///
    /// - Parameters:
    ///   - audio: Audio signal as MLXArray (1D)
    ///   - f0Min: Minimum F0 to detect (Hz)
    ///   - f0Max: Maximum F0 to detect (Hz)
    /// - Returns: F0 contour with shape (1, T, 1) in Hz
    public func pm(audio: MLXArray, f0Min: Float = 50.0, f0Max: Float = 1100.0) -> MLXArray {
        // Convert to Swift array
        MLX.eval(audio)
        let audioData = audio.asArray(Float.self)

        // Extract F0 with refinement
        var f0 = extractF0Autocorrelation(
            audio: audioData,
            f0Min: f0Min,
            f0Max: f0Max,
            refinement: true
        )

        // Apply median filtering for smoothing
        f0 = medianFilter(f0, windowSize: 3)

        // Convert to MLXArray with shape (1, T, 1)
        let f0Array = MLXArray(f0)
        return f0Array.reshaped([1, f0.count, 1])
    }

    /// HARVEST-style pitch estimation (higher quality, slower).
    ///
    /// - Parameters:
    ///   - audio: Audio signal as MLXArray (1D)
    ///   - f0Min: Minimum F0 to detect (Hz)
    ///   - f0Max: Maximum F0 to detect (Hz)
    /// - Returns: F0 contour with shape (1, T, 1) in Hz
    public func harvest(audio: MLXArray, f0Min: Float = 50.0, f0Max: Float = 1100.0) -> MLXArray {
        // Convert to Swift array
        MLX.eval(audio)
        let audioData = audio.asArray(Float.self)

        // Extract F0 with refinement and larger frame
        var f0 = extractF0Autocorrelation(
            audio: audioData,
            f0Min: f0Min,
            f0Max: f0Max,
            refinement: true,
            largerFrame: true
        )

        // Apply stronger smoothing
        f0 = medianFilter(f0, windowSize: 5)
        f0 = meanFilter(f0, windowSize: 3)

        // Convert to MLXArray with shape (1, T, 1)
        let f0Array = MLXArray(f0)
        return f0Array.reshaped([1, f0.count, 1])
    }

    // MARK: - Core Autocorrelation Algorithm

    /// Extract F0 using autocorrelation-based method.
    private func extractF0Autocorrelation(
        audio: [Float],
        f0Min: Float,
        f0Max: Float,
        refinement: Bool,
        largerFrame: Bool = false
    ) -> [Float] {
        let actualFrameSize = largerFrame ? frameSize * 2 : frameSize

        // Calculate lag range from F0 range
        let maxLag = Int(Float(sampleRate) / f0Min)  // Longest period
        let minLag = Int(Float(sampleRate) / f0Max)  // Shortest period

        // Number of frames
        let numFrames = max(1, (audio.count - actualFrameSize) / hopSize + 1)
        var f0Values = [Float](repeating: 0, count: numFrames)

        // Process each frame
        for frameIdx in 0..<numFrames {
            let startSample = frameIdx * hopSize
            let endSample = min(startSample + actualFrameSize, audio.count)

            guard endSample - startSample >= minLag * 2 else { continue }

            // Extract frame
            var frame = Array(audio[startSample..<endSample])

            // Apply Hanning window
            applyHanningWindow(&frame)

            // Compute normalized autocorrelation
            let (bestLag, confidence) = findBestLag(
                frame: frame,
                minLag: minLag,
                maxLag: min(maxLag, frame.count / 2)
            )

            // Convert lag to frequency
            if bestLag > 0 && confidence > 0.2 {
                var f0 = Float(sampleRate) / Float(bestLag)

                // Parabolic interpolation for refinement
                if refinement && bestLag > minLag && bestLag < min(maxLag, frame.count / 2) - 1 {
                    f0 = refineF0Parabolic(frame: frame, lag: bestLag)
                }

                // Clamp to valid range
                if f0 >= f0Min && f0 <= f0Max {
                    f0Values[frameIdx] = f0
                }
            }
        }

        return f0Values
    }

    /// Find the best lag using normalized autocorrelation.
    private func findBestLag(frame: [Float], minLag: Int, maxLag: Int) -> (lag: Int, confidence: Float) {
        var autocorr = [Float](repeating: 0, count: maxLag + 1)

        // Compute autocorrelation using vDSP
        for lag in minLag...maxLag {
            var sum: Float = 0
            var energy1: Float = 0
            var energy2: Float = 0

            let n = frame.count - lag
            guard n > 0 else { continue }

            // Get lagged frame
            let frameLagged = Array(frame.suffix(from: lag))

            // Dot product
            vDSP_dotpr(frame, 1, frameLagged, 1, &sum, vDSP_Length(n))

            // Energy normalization
            vDSP_svesq(frame, 1, &energy1, vDSP_Length(n))
            vDSP_svesq(frameLagged, 1, &energy2, vDSP_Length(n))

            let normFactor = sqrt(energy1 * energy2)
            if normFactor > 1e-10 {
                autocorr[lag] = sum / normFactor
            }
        }

        // Find peak
        var bestLag = 0
        var bestValue: Float = 0

        for lag in minLag...maxLag {
            if autocorr[lag] > bestValue {
                bestValue = autocorr[lag]
                bestLag = lag
            }
        }

        return (bestLag, bestValue)
    }

    /// Refine F0 estimate using parabolic interpolation.
    private func refineF0Parabolic(frame: [Float], lag: Int) -> Float {
        // Get autocorrelation values around the peak
        var r0: Float = 0, r1: Float = 0, r2: Float = 0
        let n = frame.count - lag - 1

        guard n > 0 else { return Float(sampleRate) / Float(lag) }

        let frameLagMinus1 = Array(frame.suffix(from: lag - 1))
        let frameLag = Array(frame.suffix(from: lag))
        let frameLagPlus1 = Array(frame.suffix(from: lag + 1))

        vDSP_dotpr(frame, 1, frameLagMinus1, 1, &r0, vDSP_Length(n))
        vDSP_dotpr(frame, 1, frameLag, 1, &r1, vDSP_Length(n))
        vDSP_dotpr(frame, 1, frameLagPlus1, 1, &r2, vDSP_Length(n))

        // Parabolic interpolation
        let denom = r0 - 2 * r1 + r2
        if abs(denom) > 1e-10 {
            let delta = 0.5 * (r0 - r2) / denom
            let refinedLag = Float(lag) + delta
            return Float(sampleRate) / refinedLag
        }

        return Float(sampleRate) / Float(lag)
    }

    // MARK: - Signal Processing Utilities

    /// Apply Hanning window to frame.
    private func applyHanningWindow(_ frame: inout [Float]) {
        let n = frame.count
        for i in 0..<n {
            let w = 0.5 * (1.0 - cos(2.0 * .pi * Float(i) / Float(n - 1)))
            frame[i] *= w
        }
    }

    /// Apply median filter for smoothing.
    private func medianFilter(_ input: [Float], windowSize: Int) -> [Float] {
        guard input.count > 0 else { return input }

        var output = input
        let halfWindow = windowSize / 2

        for i in 0..<input.count {
            let start = max(0, i - halfWindow)
            let end = min(input.count, i + halfWindow + 1)

            var window = Array(input[start..<end]).filter { $0 > 0 }  // Only non-zero values
            if !window.isEmpty {
                window.sort()
                output[i] = window[window.count / 2]
            }
        }

        return output
    }

    /// Apply mean filter for smoothing.
    private func meanFilter(_ input: [Float], windowSize: Int) -> [Float] {
        guard input.count > 0 else { return input }

        var output = input
        let halfWindow = windowSize / 2

        for i in 0..<input.count {
            let start = max(0, i - halfWindow)
            let end = min(input.count, i + halfWindow + 1)

            let window = Array(input[start..<end]).filter { $0 > 0 }
            if !window.isEmpty {
                output[i] = window.reduce(0, +) / Float(window.count)
            }
        }

        return output
    }
}
