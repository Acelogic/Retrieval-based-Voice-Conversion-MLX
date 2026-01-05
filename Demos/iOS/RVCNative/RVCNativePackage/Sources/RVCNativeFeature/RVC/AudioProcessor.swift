import Foundation
import AVFoundation
import Accelerate
import MLX

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
        // Convert to Swift Array
        // Assuming array is 1D float32
        let samples = array.asArray(Float.self)
        
        let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: sampleRate, channels: 1, interleaved: false)!
        let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(count))!
        buffer.frameLength = AVAudioFrameCount(count)
        
        guard let channelData = buffer.floatChannelData?[0] else { return }
        channelData.initialize(from: samples, count: count)
        
        let file = try AVAudioFile(forWriting: url, settings: format.settings)
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
}
