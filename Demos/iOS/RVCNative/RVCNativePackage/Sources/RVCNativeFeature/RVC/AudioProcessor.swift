import Foundation
import AVFoundation
import Accelerate
import MLX
import MLXNN

final class AudioProcessor: @unchecked Sendable {
    static let shared = AudioProcessor()
    
    // Load audio file to MLXArray (1D float32)
    func loadAudio(url: URL, targetSampleRate: Double = 16000) throws -> (MLXArray, Double) {
        print("[AudioProcessor] Loading: \(url.path)")
        let file = try AVAudioFile(forReading: url)
        print("[AudioProcessor] Format: \(file.fileFormat)")
        print("[AudioProcessor] Length: \(file.length)")
        
        // Read in the file's native processing format (handling stereo/mono)
        let format = file.processingFormat
        let frameLength = AVAudioFrameCount(file.length)
        
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameLength) else {
            throw NSError(domain: "AudioProcessor", code: 2, userInfo: [NSLocalizedDescriptionKey: "Buffer allocation failed"])
        }
        
        try file.read(into: buffer)
        
        // Convert to Float array (Mono)
        let frames = Int(buffer.frameLength)
        var audioSamples = [Float](repeating: 0.0, count: frames)
        
        if let floatChannelData = buffer.floatChannelData {
            let channels = Int(format.channelCount)
            for c in 0..<channels {
                let data = UnsafeBufferPointer(start: floatChannelData[c], count: frames)
                for i in 0..<frames {
                    audioSamples[i] += data[i]
                }
            }
            
            // Average if multiple channels
            if channels > 1 {
                let scale = 1.0 / Float(channels)
                for i in 0..<frames {
                    audioSamples[i] *= scale
                }
            }
        }
        
        // Resample if necessary
        if file.fileFormat.sampleRate != targetSampleRate {
            // Simple linear interpolation or Accelerate vDSP for demo
            // For robust resale, we should use vDSP_desamp or AVAudioConverter
            // Using AVAudioConverter here for quality
            audioSamples = try resample(buffer: buffer, from: file.fileFormat.sampleRate, to: targetSampleRate)
        }
        
        // Create MLXArray
        let mlxArray = MLXArray(audioSamples)
        return (mlxArray, targetSampleRate)
    }
    
    // Save MLXArray to WAV
    func saveAudio(array: MLXArray, url: URL, sampleRate: Double = 40000) throws {
        let count = array.size
        let samples = array.asArray(Float.self)
        
        // 1. Define File Format: Int16 (Standard WAV)
        // This ensures compatibility with all players
        let fileSettings: [String: Any] = [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVSampleRateKey: sampleRate,
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 16,
            AVLinearPCMIsFloatKey: false,
            AVLinearPCMIsBigEndianKey: false,
            AVLinearPCMIsNonInterleaved: false
        ]
        
        // 2. Define Buffer Format: Float32
        // AVAudioFile processes in Float32 by default
        let bufferFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: sampleRate, channels: 1, interleaved: false)!
        let buffer = AVAudioPCMBuffer(pcmFormat: bufferFormat, frameCapacity: AVAudioFrameCount(count))!
        buffer.frameLength = AVAudioFrameCount(count)
        
        // 3. Fill Buffer (Clamp checks are handled by CoreAudio conversion usually, but safe to clamp)
        if let channelData = buffer.floatChannelData?[0] {
             for i in 0..<count {
                 // conversion to Int16 implies clamping, but since we are keeping it as float
                 // we just ensure it's in -1...1 range so conversion doesn't overflow weirdly
                 channelData[i] = min(max(samples[i], -1.0), 1.0)
             }
        }
        
        // 4. Write
        try? FileManager.default.removeItem(at: url)
        
        // Initialize file with Int16 settings, but tell it we will provide Float32 data
        let file = try AVAudioFile(forWriting: url, settings: fileSettings, commonFormat: .pcmFormatFloat32, interleaved: false)
        try file.write(from: buffer)
    }
    
    private func resample(buffer: AVAudioPCMBuffer, from sourceRate: Double, to targetRate: Double) throws -> [Float] {
        let sourceFormat = buffer.format
        let targetFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: targetRate, channels: 1, interleaved: false)!
        
        guard let converter = AVAudioConverter(from: sourceFormat, to: targetFormat) else {
            throw NSError(domain: "AudioProcessor", code: 4, userInfo: [NSLocalizedDescriptionKey: "Converter init failed"])
        }
        
        let ratio = targetRate / sourceRate
        let targetFrameCount = AVAudioFrameCount(Double(buffer.frameLength) * ratio)
        
        // Restore declarations
        let targetBuffer = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: targetFrameCount)!
        var error: NSError? = nil
        
        struct BufferWrapper: @unchecked Sendable {
            let buffer: AVAudioPCMBuffer
        }
        let wrapper = BufferWrapper(buffer: buffer)
        
        // AVAudioConverterInputBlock
        let inputBlock: AVAudioConverterInputBlock = { inNumPackets, outStatus in
            outStatus.pointee = .haveData
            return wrapper.buffer
        }
        
        converter.convert(to: targetBuffer, error: &error, withInputFrom: inputBlock)
        
        if let error = error { throw error }
        
        guard let floatData = targetBuffer.floatChannelData?[0] else { return [] }
        return Array(UnsafeBufferPointer(start: floatData, count: Int(targetBuffer.frameLength)))
    }
    // MARK: - Volume Envelope (RMS)
    
    /// Calculate RMS using Conv1d to match Librosa/Python implementation
    private func calculateRMS(audio: MLXArray, sampleRate: Int) -> MLXArray {
        // Python pipeline logic:
        // Use standard window sizes (e.g. 50ms window, 12.5ms hop)
        // Previous logic using sampleRate/2 was too large (0.5s)
        let frameLength = Int(Double(sampleRate) * 0.05) // 50ms
        let hopLength = Int(Double(sampleRate) * 0.0125) // 12.5ms
        
        // 1. Square signal
        let squared = MLX.square(audio) // [T]
        
        // 2. Pad (Reflect/Edge)
        // Librosa uses centered padding (frameLength // 2 on both sides)
        // MLX.pad expects [(before, after)] per dimension
        let padAmount = frameLength / 2
        // Fix: Use tuple for IntOrPair((pad, pad))
        let padResult = MLX.padded(squared, widths: [IntOrPair((padAmount, padAmount))], mode: .edge)
        
        // 3. Conv1d for Moving Average (Sum then divide)
        // Input needs to be [N, T, C] -> [1, T_padded, 1]
        let input = padResult.reshaped([1, -1, 1])
        
        // Use MLXNN.Conv1d layer instead of functional default to ensure import availability
        let conv = MLXNN.Conv1d(
            inputChannels: 1,
            outputChannels: 1,
            kernelSize: frameLength,
            stride: hopLength,
            padding: 0,
            bias: false
        )
        
        // Set weights: 1/N
        let weightData = [Float](repeating: 1.0 / Float(frameLength), count: frameLength)
        // MLX Conv1d weight: [Out, K, In]
        let weight = MLXArray(weightData).reshaped([1, frameLength, 1])
        conv.update(parameters: ModuleParameters.unflattened(["weight": weight]))
        
        let output = conv(input)
        
        // 4. Sqrt
        var rms = MLX.sqrt(output)
        
        // Shape [1, Frames, 1] -> [1, Frames]
        rms = rms.squeezed(axis: 2)
        
        return rms
    }
    
    /// Linear Interpolation for RMS curve
    private func interpolateRMS(rms: MLXArray, targetLength: Int) -> MLXArray {
        // rms: [1, Frames]
        let frames = rms.shape[1]
        
        if frames == targetLength { return rms }
        if frames < 2 { return MLX.broadcast(rms, to: [1, targetLength]) }
        
        // Generate indices for target grid mapping to source grid
        // equivalent to np.interp(linspace(0,1,tgt), linspace(0,1,src), y)
        // This maps 0 -> 0 and (tgt-1) -> (src-1)
        
        let range = MLXArray(0..<targetLength).asType(Float.self)
        let scale = Float(frames - 1) / Float(targetLength - 1)
        let indices = range * scale
        
        let lower = MLX.floor(indices).asType(Int32.self)
        let upper = MLX.ceil(indices).asType(Int32.self)
        let weights = indices - lower.asType(Float.self) // Fractional part
        
        // Clamp indices
        let maxIdx = Int32(frames - 1)
        let lowerClamped = MLX.minimum(MLX.maximum(lower, 0), maxIdx)
        let upperClamped = MLX.minimum(MLX.maximum(upper, 0), maxIdx)
        
        // Gather values: rms[0, lower]
        // Note: indexing with MLXArray returns a new array
        let valLower = rms[0, lowerClamped] // [TargetLength]
        let valUpper = rms[0, upperClamped] // [TargetLength]
        
        // Lerp
        let result = valLower * (1 - weights) + valUpper * weights
        return result
    }

    /// Apply Volume Envelope mixing (RMS matching)
    /// rate: 1.0 = No change (use Model output volume), 0.0 = Match Input Volume fully.
    /// Usually rate < 1.0 (e.g. 0.25) to mitigate noise in silence.
    public func changeRMS(sourceAudio: MLXArray, sourceRate: Int, targetAudio: MLXArray, targetRate: Int, rate: Float) -> MLXArray {
        // RMS calculation
        let rms1 = calculateRMS(audio: sourceAudio, sampleRate: sourceRate) // [1, Frames1]
        let rms2 = calculateRMS(audio: targetAudio, sampleRate: targetRate) // [1, Frames2]
        
        let targetLength = targetAudio.size
        
        // Interpolate to match target audio sample count
        let rms1Interp = interpolateRMS(rms: rms1, targetLength: targetLength)
        let rms2Interp = interpolateRMS(rms: rms2, targetLength: targetLength)
        
        // Avoid div by zero / neg
        let rms2Safe = MLX.maximum(rms2Interp, 1e-6)
        
        // Formula: target * (rms1^(1-rate) * rms2^(rate-1))
        // If rate=1: target * 1 * 1 = target (Identity)
        // If rate=0: target * rms1 / rms2 (Match Source)
        // If rate=0.25: target * rms1^0.75 * rms2^-0.75 = target * (rms1/rms2)^0.75
        
        let p1 = MLX.pow(rms1Interp, 1.0 - rate)
        let p2 = MLX.pow(rms2Safe, rate - 1.0)
        let factor = p1 * p2
        
        return targetAudio * factor
    }
    
    // MARK: - Audio Preprocessing
    
    /// Apply 5th order Butterworth high-pass filter to remove DC offset and low-frequency noise
    /// Matches Python's scipy.signal.butter(5, 48, btype="high", fs=16000) + filtfilt
    public func highPassFilter(audio: MLXArray, sampleRate: Int = 16000, cutoff: Double = 48.0) -> MLXArray {
        // Convert MLXArray to Float array for vDSP processing
        let audioData = audio.asArray(Float.self)
        let count = audioData.count
        
        // For 5th order, we use 3 cascaded sections (1st + 2nd + 2nd order)
        // Using exact scipy.signal.butter(5, 48, btype="high", fs=16000, output='sos') coefficients
        
        // Section 0 (first order)
        let b0_0: Double = 0.9699606451838447
        let b1_0: Double = -0.9699606451838447
        let a1_0: Double = -0.9813258904926881
        
        // Section 1 (biquad)
        let b0_1: Double = 1.0
        let b1_1: Double = -2.0
        let b2_1: Double = 1.0
        let a1_1: Double = -1.9696106864374547
        let a2_1: Double = 0.9699606452559844
        
        // Section 2 (biquad)
        let b0_2: Double = 1.0
        let b1_2: Double = -2.0
        let b2_2: Double = 1.0
        let a1_2: Double = -1.9880652422382208
        let a2_2: Double = 0.9884184800471566
        
        var output = audioData
        
        // Apply cascaded sections (forward pass)
        output = applyFirstOrder(output, b0: b0_0, b1: b1_0, a1: a1_0)
        output = applyBiquad(output, b0: b0_1, b1: b1_1, b2: b2_1, a1: a1_1, a2: a2_1)
        output = applyBiquad(output, b0: b0_2, b1: b1_2, b2: b2_2, a1: a1_2, a2: a2_2)
        
        // Reverse pass (filtfilt for zero-phase)
        output.reverse()
        output = applyFirstOrder(output, b0: b0_0, b1: b1_0, a1: a1_0)
        output = applyBiquad(output, b0: b0_1, b1: b1_1, b2: b2_1, a1: a1_1, a2: a2_1)
        output = applyBiquad(output, b0: b0_2, b1: b1_2, b2: b2_2, a1: a1_2, a2: a2_2)
        output.reverse()
        
        return MLXArray(output)
    }
    
    /// Apply biquad IIR filter section
    private func applyBiquad(_ input: [Float], b0: Double, b1: Double, b2: Double, a1: Double, a2: Double) -> [Float] {
        var output = [Float](repeating: 0, count: input.count)
        var w1: Double = 0.0
        var w2: Double = 0.0
        
        // Direct Form II implementation
        for i in 0..<input.count {
            let w0 = Double(input[i]) - a1 * w1 - a2 * w2
            output[i] = Float(b0 * w0 + b1 * w1 + b2 * w2)
            w2 = w1
            w1 = w0
        }
        
        return output
    }
    
    /// Apply first-order IIR filter section
    private func applyFirstOrder(_ input: [Float], b0: Double, b1: Double, a1: Double) -> [Float] {
        var output = [Float](repeating: 0, count: input.count)
        var w1: Double = 0.0
        
        for i in 0..<input.count {
            let w0 = Double(input[i]) - a1 * w1
            output[i] = Float(b0 * w0 + b1 * w1)
            w1 = w0
        }
        
        return output
    }
    
    /// Normalize audio to prevent clipping (matches Python's max/0.99 normalization)
    public func normalizeAudio(_ audio: MLXArray, maxScale: Float = 0.99) -> MLXArray {
        let audioMax = MLX.max(MLX.abs(audio)).item(Float.self) / maxScale
        if audioMax > 1.0 {
            return audio / audioMax
        }
        return audio
    }
    
    // MARK: - Audio Analysis
    
    public struct AudioStats: CustomStringConvertible {
        public let rms: Float
        public let peak: Float
        public let minVal: Float
        public let maxVal: Float
        public let zeroCrossingRate: Float
        public let duration: Float
        
        public var description: String {
            return String(format: "RMS: %.4f | Peak: %.4f | Range: [%.4f, %.4f] | ZCR: %.4f | Dur: %.2fs", rms, peak, minVal, maxVal, zeroCrossingRate, duration)
        }
    }
    
    public func analyzeAudio(_ audio: MLXArray, sampleRate: Int) -> AudioStats {
        // Compute stats synchronously (eval needed)
        
        // 1. Min/Max/Peak
        let minVal = audio.min().item(Float.self)
        let maxVal = audio.max().item(Float.self)
        let peak = max(abs(minVal), abs(maxVal))
        
        // 2. RMS
        let squared = MLX.square(audio)
        let meanSquared = squared.mean().item(Float.self)
        let rms = sqrt(meanSquared)
        
        // 3. Zero Crossing Rate
        // Count sign changes
        let sign = MLX.sign(audio)
        let diff = sign[1...] - sign[0..<(audio.shape[0]-1)]
        let nonZeroDiffs = MLX.abs(diff) .> 0
        let zeroCrossings = nonZeroDiffs.sum().item(Int.self)
        let zcr = Float(zeroCrossings) / Float(audio.shape[0])
        
        // 4. Duration
        let duration = Float(audio.shape[0]) / Float(sampleRate)
        
        return AudioStats(
            rms: rms,
            peak: peak,
            minVal: minVal,
            maxVal: maxVal,
            zeroCrossingRate: zcr,
            duration: duration
        )
    }
}
