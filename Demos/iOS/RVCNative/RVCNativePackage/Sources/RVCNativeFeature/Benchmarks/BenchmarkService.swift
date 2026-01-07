import Foundation
import MLX
import MLXRandom

// MARK: - Benchmark Service

public actor BenchmarkService {
    public static let shared = BenchmarkService()
    
    public struct BenchmarkResult: Sendable, CustomStringConvertible {
        public let audioDuration: Double
        public let rtf: Double
        public let rms: Float
        public let peak: Float
        public let spectralCorrelation: Float?
        public let rmse: Float?
        
        public var description: String {
            var s = "Benchmark Report:\n"
            s += String(format: "- Duration: %.4fs\n", audioDuration)
            s += String(format: "- RTF: %.4f (Speed: %.2fx)\n", rtf, 1.0/rtf)
            s += String(format: "- RMS: %.4f\n", rms)
            s += String(format: "- Peak: %.4f\n", peak)
            if let sc = spectralCorrelation {
                s += String(format: "- Spec Corr: %.4f\n", sc)
            }
            if let rmse = rmse {
                s += String(format: "- RMSE: %.4f\n", rmse)
            }
            return s
        }
    }
    
    private init() {}
    
    public func computeMetrics(candidate: MLXArray, reference: MLXArray?, processingTime: Double, sampleRate: Int) -> BenchmarkResult {
        // 1. Basic Stats
        let rms = sqrt(MLX.square(candidate).mean()).item(Float.self)
        let peak = MLX.max(MLX.abs(candidate)).item(Float.self)
        let durationSec = Double(candidate.size) / Double(sampleRate)
        let rtfVal = processingTime / durationSec
        
        var specCorr: Float? = nil
        var rmseVal: Float? = nil
        
        if let ref = reference {
            // 2. RMSE
            // Ensure same length
            let minLen = min(candidate.size, ref.size)
            let cSlice = candidate[0..<minLen]
            let rSlice = ref[0..<minLen]
            
            let diff = cSlice - rSlice
            let diffSq = MLX.square(diff)
            rmseVal = sqrt(diffSq.mean()).item(Float.self)
            
            // 3. Spectral Correlation
            // Compute Log-Mel Spectrograms
            let melC = DSPUtils.melSpectrogram(audio: cSlice, sampleRate: sampleRate)
            let melR = DSPUtils.melSpectrogram(audio: rSlice, sampleRate: sampleRate)
            
            // Pearson Correlation
            // corr = cov(x, y) / (std(x) * std(y))
            // Flatten to 1D for correlation
            let flatC = melC.reshaped([-1])
            let flatR = melR.reshaped([-1])
            
            let meanC = flatC.mean()
            let meanR = flatR.mean()
            
            // Explicitly subtract MLXArrays (result is MLXArray)
            let subC: MLXArray = flatC - meanC
            let subR: MLXArray = flatR - meanR
            
            let num: MLXArray = (subC * subR).mean()
            let den: MLXArray = MLX.std(flatC) * MLX.std(flatR)
            
            let corr = num / den
            specCorr = corr.item(Float.self)
        }
        
        return BenchmarkResult(
            audioDuration: durationSec,
            rtf: rtfVal,
            rms: rms,
            peak: peak,
            spectralCorrelation: specCorr,
            rmse: rmseVal
        )
    }
}
