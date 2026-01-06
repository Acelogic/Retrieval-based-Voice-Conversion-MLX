
import Foundation
import MLX
import MLXRandom
import MLXNN

    @MainActor
    public class RVCInference: ObservableObject {
        public static let bundle = Bundle.module
        @Published public var status: String = "Idle"
        
        // Callback for logging to UI
        public var onLog: ((String) -> Void)?
        
        var hubertModel: HubertModel?
        var featureProjection: MLXNN.Linear?  // 768 -> 192
        var generator: Generator?
        var rmvpe: RMVPE?
        
        private func log(_ message: String) {
            print(message) // Keep console output
            DispatchQueue.main.async {
                self.onLog?(message)
            }
        }
        
        public init() {
            log("RVCInference: Initializing...")
            #if targetEnvironment(simulator)
            MLX.Device.setDefault(device: Device.cpu)
            log("RVCInference: Running on Simulator, forced CPU device.")
            #endif
        }
        
        public func loadWeights(hubertURL: URL, modelURL: URL, rmvpeURL: URL? = nil) async throws {
            DispatchQueue.main.async { self.status = "Loading models..." }
            
            // 1. Load Hubert
            log("RVCInference: Loading Hubert from \(hubertURL.lastPathComponent)")
            let hubertWeights = try MLX.loadArrays(url: hubertURL)
            
            self.hubertModel = HubertModel(config: HubertConfig())
            var newParams: [String: MLXArray] = [:]
            for (k, v) in hubertWeights {
                newParams[k] = v
            }
            self.hubertModel?.update(parameters: ModuleParameters.unflattened(newParams))
            
            // 2. Load Generator (simplified approach - just Generator + feature projection)
            log("RVCInference: Loading Generator from \(modelURL.lastPathComponent)")
            let modelWeights = try MLX.loadArrays(url: modelURL)
            
            // Auto-transpose Conv1d weights from PyTorch to MLX format
            var transposedWeights: [String: MLXArray] = [:]
            var transposedCount = 0
            for (key, value) in modelWeights {
                if key.contains(".weight") && value.ndim == 3 {
                    let shape = value.shape
                    if shape[2] < shape[1] {
                        transposedWeights[key] = value.transposed(0, 2, 1)
                        transposedCount += 1
                    } else {
                        transposedWeights[key] = value
                    }
                } else {
                    transposedWeights[key] = value
                }
            }
            log("RVCInference: Transposed \(transposedCount) Conv1d weights")
            
            // Create feature projection (768 -> 192 using enc_p.emb_phone weights)
            self.featureProjection = MLXNN.Linear(768, 192)
            if let weight = transposedWeights["enc_p.emb_phone.weight"],
               let bias = transposedWeights["enc_p.emb_phone.bias"] {
                self.featureProjection?.update(parameters: ModuleParameters.unflattened([
                    "weight": weight,
                    "bias": bias
                ]))
                log("RVCInference: Loaded feature projection (768->192)")
            }
            
            // Create Generator (192 input, with gin_channels for conditioning)
            self.generator = Generator(inputChannels: 192, ginChannels: 256)
            
            // Load Generator weights (dec.* prefix)
            var genParams: [String: MLXArray] = [:]
            for (k, v) in transposedWeights {
                if k.hasPrefix("dec.") {
                    let newK = String(k.dropFirst(4))
                    genParams[newK] = v
                }
            }
            self.generator?.update(parameters: ModuleParameters.unflattened(genParams))
            log("RVCInference: Loaded Generator with \(genParams.count) weight keys")
            
            // 3. Load RMVPE (Optional)
            if let rmvpeURL = rmvpeURL {
                do {
                    log("RVCInference: Loading RMVPE from \(rmvpeURL.lastPathComponent)")
                    let rmvpeWeights = try MLX.loadArrays(url: rmvpeURL)
                    self.rmvpe = RMVPE()
                    
                    self.rmvpe?.update(parameters: ModuleParameters.unflattened(rmvpeWeights))
                    log("RVCInference: Loaded RMVPE weights: \(rmvpeWeights.count) keys")
                } catch {
                    log("RVCInference: Failed to load RMVPE: \(error). Using fallback F0.")
                    self.rmvpe = nil
                }
            } else {
                log("RVCInference: No RMVPE URL provided. Using fallback F0.")
            }
            
            DispatchQueue.main.async { self.status = "Models Loaded" }
        }
        
        public func infer(audioURL: URL, outputURL: URL) async {
            do {
                DispatchQueue.main.async { self.status = "Loading Audio..." }
                
                // Load whole audio (Float array size is fine, just not tensors)
                let (audioArray, sampleRate) = try AudioProcessor.shared.loadAudio(url: audioURL)
                // audioArray: [TotalSamples]
                
                let totalSamples = audioArray.size
                // 3 seconds chunk size (Safe for mobile memory)
                // 16000 * 3 = 48000. 48000 % 320 == 0.
                let chunkSize = 16000 * 3
                let padSamples = 16000 // 1 second context padding
                
                // Pad input audio globally to handle edges (Simulate Reflection/Edge padding)
                // MLX.padded with .edge mode
                let audioArrayPadded = MLX.padded(audioArray, widths: [IntOrPair((padSamples, padSamples))], mode: .edge)
                
                log("RVCInference: Starting chunked inference. Total samples: \(totalSamples). Chunk size: \(chunkSize). Padding: \(padSamples)")
                
                var outputChunks: [MLXArray] = []
                var currentPos = 0
                var chunkIdx = 0
                
                while currentPos < totalSamples {
                    chunkIdx += 1
                    log("RVCInference: Processing chunk \(chunkIdx)... (Original: \(currentPos)-\(min(currentPos+chunkSize, totalSamples)))")
                    DispatchQueue.main.async { self.status = "Chunk \(chunkIdx)..." }
                    
                    // Calculate positions in the PADDED array
                    let startIndexPadded = currentPos // Already offset by padSamples due to prepending
                    let endIndexPadded = startIndexPadded + chunkSize + (2 * padSamples) // Include pre and post padding
                    
                    // Clamp end if necessary
                    let actualEndPadded = min(endIndexPadded, audioArrayPadded.size)
                    
                    // Extract padded chunk
                    let paddedChunk = audioArrayPadded[startIndexPadded..<actualEndPadded]
                    
                    // Process chunk
                    let processedPaddedChunk = try await inferChunk(chunk: paddedChunk)
                    
                    // Calculate crop indices.
                    // The processedPaddedChunk has audio for padded+core+padded samples input.
                    // Each input sample roughly maps to `targetSR / sourceSR` output samples.
                    // Here source is 16kHz hubert, target is 40kHz audio.
                    // Actually, the hop size relationship is complex. Simpler: samples_per_feature.
                    // Let's assume output length is proportional to input length.
                    // The output for the core `chunkSize` samples is in the middle.
                    
                    // Output conversion: 16kHz input, 40kHz output. Ratio = 2.5.
                    // No, the HuBERT processes at 16kHz, conv_pre hop is 320 = 20ms@16kHz.
                    // Generator upsamples by 400x from features (10*10*2*2 = 400).
                    // So if feature length is F, audio length is F * 400.
                    // Hubert stride is 320, so AudioLen / 320 ~= FeatureLen.
                    // Combined: OutputAudio = (InputAudio / 320) * 400 = InputAudio * 1.25
                    // 16000 samples -> 16000 * 1.25 = 20000? No that doesn't seem right for 16k to 40k conversion it should be 2.5x.
                    // The model is trained for 40k output. Hubert is at 16k.
                    // Input 16k samples -> Features. Features * 400 upsample -> Output samples @ 40k.
                    // Let's check: 1 second audio at 16kHz = 16000 samples. HuBERT stride 320 -> 50 features.
                    // Features upsampled 2x in pipeline, then generator x400. So 50 * 2 * 400 = 40000 output samples.
                    // That's 40k samples for 1 second, i.e., 40kHz output. Correct!
                    // So output length is input_samples * (40000/16000) = input * 2.5
                    
                    let outputRatio: Float = 40000.0 / 16000.0 // = 2.5
                    
                    // We need to crop padSamples worth of audio from each end.
                    let cropOutputSamples = Int(Float(padSamples) * outputRatio)
                    
                    let processedLen = processedPaddedChunk.shape[1] // Assuming [B, L, 1]
                    let coreStartIdx = cropOutputSamples
                    var coreEndIdx = processedLen - cropOutputSamples
                    
                    // Handle edge cases for last chunk
                    if coreEndIdx <= coreStartIdx {
                        // Chunk too small, take whatever we got. This is edge case.
                        coreEndIdx = processedLen
                    }
                    
                    let coreOutput = processedPaddedChunk[0..., coreStartIdx..<coreEndIdx, 0...]
                    outputChunks.append(coreOutput.squeezed(axes: [0, 2])) // [L]
                    
                    // Aggressive memory release each chunk
                    MLX.eval(coreOutput)
                    MLX.Memory.clearCache()
                    
                    currentPos += chunkSize
                }
                
                DispatchQueue.main.async { self.status = "Finalizing..." }
                
                // Concatenate all output chunks
                let finalOutput = MLX.concatenated(outputChunks, axis: 0) // [TotalOutputSamples]
                MLX.eval(finalOutput)
                
                let outputSampleRate: Double = 40000.0 // Standard RVC V2 output
                
                log("RVCInference: Saving output to \(outputURL.path)")
                try AudioProcessor.shared.saveAudio(array: finalOutput, url: outputURL, sampleRate: outputSampleRate)
                
                log("RVCInference: Done!")
                DispatchQueue.main.async { self.status = "Done!" }
                
            } catch {
                log("RVCInference Error: \(error)")
                DispatchQueue.main.async { self.status = "Error: \(error.localizedDescription)" }
            }
        }
        
        private func inferChunk(chunk: MLXArray) async throws -> MLXArray {
            // chunk: [T]
             
            // 1. Hubert Feature Extraction
            let audioInput = chunk.expandedDimensions(axis: 0) // [1, T]
            guard let hubertModel = hubertModel else {
                throw NSError(domain: "RVCInference", code: 1, userInfo: [NSLocalizedDescriptionKey: "Hubert model missing"])
            }
            let features = hubertModel(audioInput) // [1, Frames, 768]
            MLX.eval(features)
             
            // 2. Feature Projection (768 -> 192)
            guard let projection = featureProjection else {
                throw NSError(domain: "RVCInference", code: 1, userInfo: [NSLocalizedDescriptionKey: "Feature projection missing"])
            }
            var projectedFeatures = projection(features)  // [1, Frames, 192]
            projectedFeatures = leakyRelu(projectedFeatures, negativeSlope: 0.1)
            MLX.eval(projectedFeatures)
            
            // 3. F0 Estimation
            var f0: MLXArray
            if let rmvpe = rmvpe {
                f0 = rmvpe.infer(audio: chunk, thred: 0.03) // [1, Frames_rmvpe, 1]
            } else {
                // Fallback: constant F0
                let frames = features.shape[1] * 2
                f0 = MLX.full([1, frames, 1], values: MLXArray(200.0))
            }
            MLX.eval(f0)
            
            // 4. Upsample Features (2x) - simple repeat
            let N = projectedFeatures.shape[0]
            let L = projectedFeatures.shape[1]
            let C = projectedFeatures.shape[2]
            
            let expanded = projectedFeatures.expandedDimensions(axis: 2)
            let broadcasted = MLX.broadcast(expanded, to: [N, L, 2, C])
            var phone = broadcasted.reshaped([N, L * 2, C])  // [N, L*2, 192]
            
            // 5. Sync lengths
            let lenFeat = phone.shape[1]
            let lenF0 = f0.shape[1]
            let minLen = min(lenFeat, lenF0)
            
            phone = phone[0..., 0..<minLen, 0...]
            f0 = f0[0..., 0..<minLen, 0...]
            
            // 6. Generator
            guard let generator = generator else {
                throw NSError(domain: "RVCInference", code: 1, userInfo: [NSLocalizedDescriptionKey: "Generator missing"])
            }
            
            let audioOut = generator(phone, f0: f0)
            return audioOut
        }
    }
