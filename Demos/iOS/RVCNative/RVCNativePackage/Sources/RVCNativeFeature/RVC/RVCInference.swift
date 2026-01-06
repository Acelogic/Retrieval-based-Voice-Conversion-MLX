
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
        var synthesizer: Synthesizer?
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
            
            // 2. Load Synthesizer (TextEncoder + Flow + Generator)
            log("RVCInference: Loading Synthesizer from \(modelURL.lastPathComponent)")
            let modelWeights = try MLX.loadArrays(url: modelURL)

            // Note: The Python conversion script already transposes all weights to MLX format
            // No additional transposition needed here!
            log("RVCInference: Loaded \(modelWeights.count) weights (already in MLX format)")
            
            // Initialize Synthesizer (V2 defaults)
            self.synthesizer = Synthesizer(
                interChannels: 192,
                hiddenChannels: 192,
                filterChannels: 768,
                nHeads: 2,
                nLayers: 6,
                kernelSize: 3,
                pDropout: 0.0,
                embeddingDim: 768,  // Model weights expect 768-dim HuBERT features
                speakerEmbedDim: 256,
                ginChannels: 256,
                useF0: true
            )
            
            // Remap keys for Synthesizer
            // RVC V2 PyTorch Weights:
            // enc_p.* -> TextEncoder
            // dec.* -> Generator
            // flow.flows.0, 2, 4, 6 -> Flow (indices 0, 1, 2, 3)
            // emb_g.weight -> Speaker Embedding
            
            var synthParams: [String: MLXArray] = [:]
            for (k, v) in modelWeights {
                var newK = k
                if k.hasPrefix("flow.flows.") {
                    let parts = k.components(separatedBy: ".")
                    if parts.count >= 3, let oldIdx = Int(parts[2]) {
                        // PyTorch indices: 0, 2, 4, 6
                        // Swift indices: 0, 1, 2, 3
                        let newIdx = oldIdx / 2
                        newK = (["flow", "flows", String(newIdx)] + parts.dropFirst(3)).joined(separator: ".")
                    }
                }
                synthParams[newK] = v
            }
            
            self.synthesizer?.update(parameters: ModuleParameters.unflattened(synthParams))
            log("RVCInference: Loaded Synthesizer with \(synthParams.count) weight keys")
            
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
            // chunk: [T] - 16kHz
             
            // 1. Hubert Feature Extraction (16kHz -> 50fps)
            let audioInput = chunk.expandedDimensions(axis: 0) // [1, T]
            guard let hubertModel = hubertModel else {
                throw NSError(domain: "RVCInference", code: 1, userInfo: [NSLocalizedDescriptionKey: "Hubert model missing"])
            }
            let hubertFeatures = hubertModel(audioInput) // [1, Frames, 768]
            MLX.eval(hubertFeatures)
            log("DEBUG: HuBERT output shape: \(hubertFeatures.shape)")

            // 2. F0 Estimation (16kHz -> 100fps)
            var f0: MLXArray
            if let rmvpe = rmvpe {
                f0 = rmvpe.infer(audio: chunk, thred: 0.03) // [1, Frames_rmvpe, 1]
            } else {
                // Fallback: constant F0 (200Hz)
                // RMVPE produces 100fps, Hubert 50fps. So RMVPE has 2x frames.
                let frames = hubertFeatures.shape[1] * 2
                f0 = MLX.full([1, frames, 1], values: MLXArray(200.0))
            }
            MLX.eval(f0)
            
            // 3. Upsample Hubert Features to match F0 (100fps)
            let N = hubertFeatures.shape[0]
            let L = hubertFeatures.shape[1]
            let C = hubertFeatures.shape[2]
            
            // Simple repeat upsampling: [1, L, 768] -> [1, L, 2, 768] -> [1, L*2, 768]
            let expanded = hubertFeatures.expandedDimensions(axis: 2)
            let broadcasted = MLX.broadcast(expanded, to: [N, L, 2, C])
            var phone = broadcasted.reshaped([N, L * 2, C])
            log("DEBUG: Upsampled phone shape: \(phone.shape)")

            // 4. Coarse Pitch calculation (Hz -> Bucket 1-255)
            let f0Hz = f0.squeezed(axes: [2]) // [1, L_f0]
            let f0_min: Float = 50.0
            let f0_max: Float = 1100.0
            let f0_mel_min = 1127.0 * Darwin.log(1.0 + Double(f0_min) / 700.0)
            let f0_mel_max = 1127.0 * Darwin.log(1.0 + Double(f0_max) / 700.0)
            
            // MLX Mel calculation: 1127 * ln(1 + f/700)
            let f0_mel = 1127.0 * MLX.log(1.0 + f0Hz / 700.0)
            
            // Bucket quantization
            var pitch = (f0_mel - f0_mel_min) * (254.0 / (f0_mel_max - f0_mel_min)) + 1.0
            pitch = MLX.where(f0Hz .<= f0_min, MLXArray(1.0), pitch) 
            pitch = MLX.maximum(pitch, 1.0)
            pitch = MLX.minimum(pitch, 255.0)
            let pitchBuckets = pitch.asType(Int32.self)
            
            // 5. Sync lengths
            let p_len_val = min(phone.shape[1], f0Hz.shape[1])
            phone = phone[0..., 0..<p_len_val, 0...]
            let nsff0 = f0Hz[0..., 0..<p_len_val].expandedDimensions(axis: 2)
            let pitchFinal = pitchBuckets[0..., 0..<p_len_val]
            let phoneLengths = MLXArray([Int32(p_len_val)])
            
            // 6. Synthesizer Inference
            guard let synthesizer = synthesizer else {
                throw NSError(domain: "RVCInference", code: 1, userInfo: [NSLocalizedDescriptionKey: "Synthesizer missing"])
            }
            
            let sid = MLXArray([Int32(0)]) // Default speaker 0

            log("DEBUG: Before Synthesizer - phone: \(phone.shape), pitch: \(pitchFinal.shape), nsff0: \(nsff0.shape)")

            let audioOut = synthesizer.infer(
                phone: phone,
                phoneLengths: phoneLengths,
                pitch: pitchFinal,
                nsff0: nsff0,
                sid: sid
            )
            
        return audioOut // [1, T_out, 1]
        }
    }
