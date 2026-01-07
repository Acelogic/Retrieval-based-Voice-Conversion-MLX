import Foundation
import AVFoundation
import Accelerate
import MLX
import MLXNN

final class AudioProcessor: @unchecked Sendable {
    static let shared = AudioProcessor()
    
    // Load audio file to MLXArray (1D float32)
    func loadAudio(url: URL, targetSampleRate: Double = 16000) throws -> (MLXArray, Double) {
        let file = try AVAudioFile(forReading: url)
        
        guard let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: file.fileFormat.sampleRate, channels: 1, interleaved: false) else {
            throw NSError(domain: "AudioProcessor", code: 1, userInfo: [NSLocalizedDescriptionKey: "Invalid format"])
        }
        
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(file.length)) else {
            throw NSError(domain: "AudioProcessor", code: 2, userInfo: [NSLocalizedDescriptionKey: "Buffer allocation failed"])
        }
        
        try file.read(into: buffer)
        
        guard let floatData = buffer.floatChannelData?[0] else {
            throw NSError(domain: "AudioProcessor", code: 3, userInfo: [NSLocalizedDescriptionKey: "No float data"])
        }
        
        let frameLength = Int(buffer.frameLength)
        var audioSamples = Array(UnsafeBufferPointer(start: floatData, count: frameLength))
        
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
        // frame_length = source_rate // 2 * 2
        // hop_length = source_rate // 2
        let frameLength = (sampleRate / 2) * 2
        let hopLength = sampleRate / 2
        
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
}
