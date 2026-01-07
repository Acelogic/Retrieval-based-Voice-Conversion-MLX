
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

            // DEBUG: Print all dec.* keys to understand weight naming
            let decKeys = modelWeights.keys.filter { $0.hasPrefix("dec.") }.sorted()
            log("RVCInference: Generator weight keys: \(decKeys.prefix(20))...")  // Show first 20
            
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

            // Helper: Check if a key needs .conv inserted (for Conv1d wrapper classes)
            func needsConvInsertion(_ key: String) -> Bool {
                // Generator Conv1d wrappers (NOT ConvTranspose1d - now using native)
                if key.hasPrefix("dec.conv_pre.") || key.hasPrefix("dec.conv_post.") { return true }
                // dec.up_* uses native ConvTransposed1d, no .conv wrapper
                if key.contains("dec.noise_conv_") { return true }
                if key.contains("dec.resblock_") && (key.contains(".c1_") || key.contains(".c2_")) { return true }
                return false
            }

            // Helper: Check if this is a ConvTranspose1d weight that needs kernel flip
            // Handle both naming conventions: dec.up_0.weight OR dec.ups.0.weight
            func isConvTransposeWeight(_ key: String) -> Bool {
                let isUpWeight = (key.contains("dec.up_") || key.contains("dec.ups.")) && key.hasSuffix(".weight")
                return isUpWeight
            }

            var synthParams: [String: MLXArray] = [:]
            for (k, v) in modelWeights {
                var newK = k
                var newV = v

                // 1. Remap PyTorch-style Generator keys to match Swift structure
                // dec.ups.X -> dec.up_X
                if newK.contains("dec.ups.") {
                    newK = newK.replacingOccurrences(of: "dec.ups.", with: "dec.up_")
                }
                // dec.noise_convs.X -> dec.noise_conv_X
                if newK.contains("dec.noise_convs.") {
                    newK = newK.replacingOccurrences(of: "dec.noise_convs.", with: "dec.noise_conv_")
                }
                // dec.resblocks.X.convs1.Y -> dec.resblock_X.c1_Y
                if newK.contains("dec.resblocks.") {
                    newK = newK.replacingOccurrences(of: "dec.resblocks.", with: "dec.resblock_")
                    newK = newK.replacingOccurrences(of: ".convs1.", with: ".c1_")
                    newK = newK.replacingOccurrences(of: ".convs2.", with: ".c2_")
                }

                // 2. Flow index remapping
                if k.hasPrefix("flow.flows.") {
                    let parts = k.components(separatedBy: ".")
                    if parts.count >= 3, let oldIdx = Int(parts[2]) {
                        // PyTorch indices: 0, 2, 4, 6
                        // Swift indices: 0, 1, 2, 3
                        let newIdx = oldIdx / 2
                        newK = (["flow", "flows", String(newIdx)] + parts.dropFirst(3)).joined(separator: ".")
                    }
                }

                // 3. DON'T flip kernel - the conversion script already handled transposition
                // Testing: The manual ConvTranspose1d implementation might not need kernel flip
                // if isConvTransposeWeight(k) && newV.ndim == 3 {
                //     newV = newV[0..., .stride(by: -1), 0...]
                //     log("RVCInference: Flipped kernel for \(k)")
                // }

                // 4. Insert .conv for Conv1d wrapper classes in Generator
                if needsConvInsertion(newK) {
                    if newK.hasSuffix(".weight") {
                        newK = String(newK.dropLast(7)) + ".conv.weight"
                    } else if newK.hasSuffix(".bias") {
                        newK = String(newK.dropLast(5)) + ".conv.bias"
                    }
                }

                synthParams[newK] = newV
            }

            self.synthesizer?.update(parameters: ModuleParameters.unflattened(synthParams))
            self.synthesizer?.train(false)  // CRITICAL: Set to eval mode (disables Dropout, uses BatchNorm running stats)
            log("RVCInference: Loaded Synthesizer with \(synthParams.count) weight keys")
            
            // 3. Load RMVPE (Optional)
            if let rmvpeURL = rmvpeURL {
                do {
                    log("RVCInference: Loading RMVPE from \(rmvpeURL.lastPathComponent)")
                    let rmvpeWeights = try MLX.loadArrays(url: rmvpeURL)
                    self.rmvpe = RMVPE()
                    
                    self.rmvpe?.update(parameters: ModuleParameters.unflattened(rmvpeWeights))
                    self.rmvpe?.train(false)  // CRITICAL: Set to eval mode for correct inference
                    log("RVCInference: Loaded RMVPE weights: \(rmvpeWeights.count) keys")
                } catch {
                    log("RVCInference: Failed to load RMVPE: \(error). Using fallback F0.")
                    self.rmvpe = nil
                }
            } else {
                log("RVCInference: No RMVPE URL provided. Using fallback F0.")
            }
            
            // Also set HuBERT to eval mode
            self.hubertModel?.train(false)
            
            DispatchQueue.main.async { self.status = "Models Loaded" }
        }
        
        public func infer(audioURL: URL, outputURL: URL, volumeEnvelope: Float = 1.0) async {
            do {
                DispatchQueue.main.async { self.status = "Loading Audio..." }
                
                // Load whole audio (Float array size is fine, just not tensors)
                let (audioArray, sampleRate) = try AudioProcessor.shared.loadAudio(url: audioURL)
                // audioArray: [TotalSamples]
                
                let totalSamples = audioArray.size
                log("RVCInference: Processing \(totalSamples) samples (\(Float(totalSamples)/16000.0)s at 16kHz)")
                
                DispatchQueue.main.async { self.status = "Processing..." }
                
                // Process entire audio in one pass (simpler, avoids chunk boundary artifacts)
                // For iOS memory constraints, we limit to ~30s audio for now
                let maxSamples = 16000 * 30  // 30 seconds at 16kHz
                var audioToProcess = audioArray
                if totalSamples > maxSamples {
                    log("RVCInference: Audio too long (\(totalSamples) samples), truncating to \(maxSamples)")
                    audioToProcess = audioArray[0..<maxSamples]
                }
                
                // Add small edge padding for model context (0.1s each side)
                let padSamples = 1600  // 0.1 seconds
                let audioPadded = MLX.padded(audioToProcess, widths: [IntOrPair((padSamples, padSamples))], mode: .edge)
                
                // Run inference on full audio
                let outputPadded = try await inferChunk(chunk: audioPadded)
                MLX.eval(outputPadded)
                
                // Calculate crop to remove padding from output
                // Input is at 16kHz, output is at 40kHz (2.5x ratio)
                let outputRatio: Float = 40000.0 / 16000.0
                let cropSamples = Int(Float(padSamples) * outputRatio)
                
                let outputLen = outputPadded.shape[1]
                let coreStart = cropSamples
                let coreEnd = outputLen - cropSamples
                
                var finalOutput: MLXArray
                if coreEnd > coreStart {
                    finalOutput = outputPadded[0..., coreStart..<coreEnd, 0...].squeezed(axes: [0, 2])
                } else {
                    // Fallback if output is too short
                    finalOutput = outputPadded.squeezed(axes: [0, 2])
                }
                
                MLX.eval(finalOutput)
                
                let outputSampleRate: Double = 40000.0 // Standard RVC V2 output
                
                // Volume Envelope Mixing
                if volumeEnvelope != 1.0 {
                    log("RVCInference: Applying Volume Envelope (rate: \(volumeEnvelope))")
                    // sourceRate is 16000 (from loadAudio default)
                    // targetRate is 40000
                    finalOutput = AudioProcessor.shared.changeRMS(
                        sourceAudio: audioToProcess,
                        sourceRate: 16000,
                        targetAudio: finalOutput,
                        targetRate: Int(outputSampleRate),
                        rate: volumeEnvelope
                    )
                }
                
                log("RVCInference: Saving \(finalOutput.size) samples to \(outputURL.path)")
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
            MLX.Memory.clearCache()  // MEMORY FIX: Clear after HuBERT
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
            MLX.Memory.clearCache()  // MEMORY FIX: Clear after RMVPE
            
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
            
            // LOGGING FOR PARITY CHECK: Dump first 20 values of inputs
            if true { // Always log for debug
                log("DEBUG DATA DUMP:")
                
                // 1. F0 (Raw Hz)
                // 1. F0 (Raw Hz)
                // Fix: Index [0] for batch to get rank-1 array [T]
                let f0Data = nsff0.asType(Float.self)[0, 0..<min(20, p_len_val), 0]
                var f0Str = "F0 (First 20): ["
                for i in 0..<f0Data.shape[0] {
                     f0Str += String(format: "%.4f, ", f0Data[i].item(Float.self))
                }
                log(f0Str + "]")
                
                // 2. Pitch (Buckets)
                let pitchData = pitchFinal.asType(Int32.self)[0, 0..<min(20, p_len_val)]
                var pitchStr = "Pitch (First 20): ["
                 for i in 0..<pitchData.shape[0] {
                     pitchStr += String(format: "%d, ", pitchData[i].item(Int32.self))
                }
                log(pitchStr + "]")
                
                // 3. Phone (First feature of first 20 frames)
                let phoneData = phone[0, 0..<min(20, p_len_val), 0]
                var phoneStr = "Phone[0] (First 20): ["
                for i in 0..<phoneData.shape[0] {
                     phoneStr += String(format: "%.4f, ", phoneData[i].item(Float.self))
                }
                log(phoneStr + "]")
            }
            
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
