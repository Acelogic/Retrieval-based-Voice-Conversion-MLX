import Foundation
import MLX
import MLXRandom
import MLXNN

@MainActor
public class RVCInference: ObservableObject {
    public static let bundle = Bundle.module
    @Published public var status: String = "Idle"
    var hubert: HubertModel?
    var generator: Generator?
    
    public init() {
        print("RVCInference: Initializing...")
        
        #if targetEnvironment(simulator)
        MLX.Device.setDefault(device: Device.cpu)
        print("RVCInference: Running on Simulator, forced CPU device to avoid Metal crash.")
        #endif
        
        // Initialize generator
        self.generator = Generator()
        print("RVCInference: Initialized Generator")
        
        // Initialize hubert
        self.hubert = HubertModel()
        print("RVCInference: Initialized Hubert")
    }
    
    public func loadWeights(url: URL, hubertURL: URL? = nil) throws {
        // Load Generator
        let arrays = try MLX.loadArrays(url: url)
        var generatorParams = [String: MLXArray]()
        for (key, value) in arrays {
             generatorParams[key] = value
        }
        let params = ModuleParameters.unflattened(generatorParams)
        self.generator?.update(parameters: params)
        if let hubertURL = hubertURL {
            DispatchQueue.main.async { self.status = "Loading Hubert weights..." }
            print("RVCInference: Loading Hubert weights from \(hubertURL.path)")
            let hubertWeights = try MLX.loadArrays(url: hubertURL)
            hubert?.update(parameters: ModuleParameters.unflattened(hubertWeights))
            print("RVCInference: Hubert weights loaded")
        }
        
        DispatchQueue.main.async { self.status = "Model Loaded" }
    }
    
    public func infer(audioURL: URL, outputURL: URL) async {
        DispatchQueue.main.async { self.status = "Loading Audio..." }
        print("native-rvc: Starting inference for \(audioURL.lastPathComponent)")
        
        do {
            let (audio, _) = try AudioProcessor.shared.loadAudio(url: audioURL, targetSampleRate: 16000)
            print("native-rvc: Audio loaded. Shape: \(audio.shape)")
            
            DispatchQueue.main.async { self.status = "Extracting Features..." }
            // 1. Hubert
            // Expand [T] -> [1, T]
            // Use reshaped instead of expandedDims
            let audioBatch = audio.reshaped(1, audio.shape[0])
            // Optional: Use encoder to extract features (if we had weights)
            // For now, let's assume we want to use the full hubert model
            guard let hubert = hubert else {
                throw NSError(domain: "RVCInference", code: 1, userInfo: [NSLocalizedDescriptionKey: "Hubert model not initialized"])
            }
            let features = hubert(audioBatch)
            print("RVCInference: Feature extraction complete, shape: \(features.shape)")
            
            // 2. F0 (Simulated for demo)
            // Need F0 [1, Frames, 1]
            let frames = features.shape[1]
            // Random F0 for demo proof-of-concept
            // Note: MLXRandom.uniform(low:high:shape:) is the correct API
            let f0 = MLXRandom.uniform(low: 0.0, high: 1.0, [1, frames, 1]) * 200 + 100
            _ = f0 // Suppress warning for demo
            
            DispatchQueue.main.async { self.status = "Generating Voice..." }
            print("native-rvc: Generating voice using Generator...")
            
            // Check features shape
            // Generator expects [1, 256, T] or similar depending on the model
            // Hubert output is usually [1, T, 768]
            var processedFeatures = features
            if processedFeatures.dim(2) == 768 {
                // If it's the full model output, we might need a projection or just transpose
                // Actually, the Generator in RVC usually takes 768-dim features (or 256 depending on version)
                // Let's transpose to [1, 768, T] as Generator expects channels in dim 1
                processedFeatures = processedFeatures.transposed(0, 2, 1)
            }
            
            DispatchQueue.main.async { self.status = "Generating audio..." }
            print("RVCInference: Starting generation")
            guard let generator = generator else {
                 throw NSError(domain: "RVCInference", code: 1, userInfo: [NSLocalizedDescriptionKey: "Generator model not initialized"])
            }
            let audioOut = generator(processedFeatures)
            print("native-rvc: Audio generated. Shape: \(audioOut.shape)")
            
            DispatchQueue.main.async { self.status = "Saving..." }
            
            // Save
            try AudioProcessor.shared.saveAudio(array: audioOut.squeezed(), url: outputURL)
            print("native-rvc: Audio saved to \(outputURL.path)")
            
            DispatchQueue.main.async { self.status = "Done!" }
            
        } catch {
            print("native-rvc: Error during inference: \(error)")
            DispatchQueue.main.async { self.status = "Error: \(error.localizedDescription)" }
        }
    }
}
