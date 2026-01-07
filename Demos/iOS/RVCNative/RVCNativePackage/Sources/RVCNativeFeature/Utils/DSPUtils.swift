import Foundation
import MLX
import MLXRandom

// MARK: - Digital Signal Processing Utilities
// Replicating librosa features for benchmarking

public struct DSPUtils {
    // Shared instance removed in favor of static methods to avoid Sendable issues
    
    // Cache removed to avoid detailed concurrency handling for MLXArray (ref type).
    // Recomputing Mel Matrix is cheap enough for this benchmark suite.
    
    // MARK: - STFT
    
    /// Compute Short-Time Fourier Transform power spectrum
    /// Matches librosa.stft behavior (centered, hann window)
    /// Returns: Power Spectrogram [1, Frames, Freqs] (Magnitude Squared)
    public static func stft(audio: MLXArray, n_fft: Int = 1024, hop_length: Int = 256) -> MLXArray {
        // 1. Pad audio (Reflection padding to center frames)
        // Librosa centered=True pads by n_fft // 2
        let pad = n_fft / 2
        
        // Manual reflection padding since MLX.padded only supports .edge/.constant
        // audio: [S]
        // left: audio[1...pad].reversed()
        // right: audio[S-pad-1...S-2].reversed()
        let leftIdx = MLXArray(stride(from: pad, to: 0, by: -1))
        let rightIdx = MLXArray(stride(from: audio.size - 2, to: audio.size - pad - 2, by: -1))
        
        let padded = MLX.concatenated([audio[leftIdx], audio, audio[rightIdx]], axis: 0)
        
        // 2. Window (Hann)
        // Create hann window: 0.5 - 0.5 * cos(2*pi*n / N)
        // Librosa uses N (n_fft) for the window length and periodic=True by default
        let indices = MLXArray(0..<n_fft).asType(Float.self)
        let window = 0.5 - 0.5 * MLX.cos((2.0 * Float.pi * indices) / Float(n_fft))
        
        // 3. Frame the signal
        let frames = (padded.size - n_fft) / hop_length + 1
        
        // Gather indices: [0, 1, ... n_fft-1] + [0, hop, 2*hop ... ]
        let windowRange = MLXArray(0..<n_fft)
        let hops = MLXArray(stride(from: 0, to: frames * hop_length, by: hop_length))
        // This broadcast add creates a huge index matrix [Frames, N_FFT]
        // indices = windowRange [1, N] + hops [F, 1]
        let idx = windowRange.expandedDimensions(axis: 0) + hops.expandedDimensions(axis: 1)
        
        // Gather frames
        let framed = padded[idx] // [Frames, N_FFT]
        
        // Apply window
        let windowed = framed * window // Broadcast window [N_FFT]
        
        // 4. FFT
        // MLX FFT operates on last axis by default
        let fft = MLX.rfft(windowed) 
        
        // Power spectrum = magnitude^2
        let power = MLX.square(MLX.abs(fft))
        
        // Return [1, Frames, Freqs] - let the caller handle transposition if needed
        // This avoids confusion between [T, F] and [F, T]
        return power.expandedDimensions(axis: 0)
    }
    
    // MARK: - Mel Spectrogram
    
    /// Compute Log-Mel Spectrogram
    /// Matches librosa.feature.melspectrogram + power_to_db
    public static func melSpectrogram(audio: MLXArray, sampleRate: Int = 16000, n_fft: Int = 1024, hop_length: Int = 256, n_mels: Int = 128) -> MLXArray {
        // 1. Compute Power Spectrogram [1, T, F]
        let powerSpec = stft(audio: audio, n_fft: n_fft, hop_length: hop_length) // [1, T, F]
        
        // 2. Get Mel Filterbank [M, F]
        let melFilter = getMelFilter(sr: sampleRate, n_fft: n_fft, n_mels: n_mels) // [M, F]
        
        // 3. Apply Filterbank
        // Spec: [1, T, F] or [1, F, T]. Filter: [M, F].
        // Matmul wants (M, F) * (F, T)
        let spec2d = powerSpec.squeezed(axis: 0)
        let melSpec: MLXArray
        if spec2d.shape[0] == melFilter.shape[1] {
            // Already [F, T]
            melSpec = matmul(melFilter, spec2d)
        } else {
            // Likely [T, F], so transpose
            melSpec = matmul(melFilter, spec2d.T)
        }
        
        // 4. Power to DB
        // ref=max
        let top = melSpec.max().item(Float.self)
        let logMel = 10.0 * MLX.log10(MLX.maximum(melSpec, 1e-10))
        let peak = 10.0 * log10(max(top, 1e-10))
        let db = MLX.maximum(logMel, peak - 80.0) // Clip to top - 80dB range
        
        return db.expandedDimensions(axis: 0) // [1, M, T]
    }
    
    private static func getMelFilter(sr: Int, n_fft: Int, n_mels: Int) -> MLXArray {
        // Recompute filter (uncached)
        let n_freqs = n_fft / 2 + 1
        
        // 1. FFT Frequencies
        let fftFreqs = MLX.linspace(0, Float(sr) / 2.0, count: n_freqs)
        
        // 2. Mel points (in Hz)
        // Use Slaney-style mel scale (Librosa default)
        let minMel = hzToMel(0)
        let maxMel = hzToMel(Float(sr) / 2.0)
        
        // Linear Mel scale points
        let melPoints = MLX.linspace(minMel, maxMel, count: n_mels + 2)
        
        // Convert back to Hz [n_mels + 2]
        var melFreqsHz: [Float] = []
        let melPointsArray = melPoints.asArray(Float.self)
        for m in melPointsArray {
            melFreqsHz.append(melToHz(m))
        }
        
        // 3. Create Triangle Filters
        var weights = [Float](repeating: 0.0, count: n_mels * n_freqs)
        let fftFreqsArray = fftFreqs.asArray(Float.self)
        
        for i in 0..<n_mels {
            let left = melFreqsHz[i]
            let center = melFreqsHz[i+1]
            let right = melFreqsHz[i+2]
            
            // Slaney normalization
            let normFactor = 2.0 / (right - left)
            
            for j in 0..<n_freqs {
                let f = fftFreqsArray[j]
                if f > left && f < center {
                    weights[i * n_freqs + j] = (f - left) / (center - left) * normFactor
                } else if f >= center && f < right {
                    weights[i * n_freqs + j] = (right - f) / (right - center) * normFactor
                }
            }
        }
        
        return MLXArray(weights).reshaped([n_mels, n_freqs])
    }
    
    // Helpers
    private static func hzToMel(_ f: Float) -> Float {
        return 2595.0 * log10(1.0 + f / 700.0)
    }
    
    private static func melToHz(_ mel: Float) -> Float {
        return 700.0 * (pow(10.0, mel / 2595.0) - 1.0)
    }
}
