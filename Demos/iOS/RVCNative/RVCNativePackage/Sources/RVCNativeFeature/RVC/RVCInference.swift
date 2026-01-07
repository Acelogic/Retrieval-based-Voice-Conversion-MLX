
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
        var modelSampleRate: Int = 40000  // Detected from model config
        
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
                var newKey = k
                
                // Remap encoder.layers.X to encoder.lX (Swift model uses l0, l1, etc.)
                if newKey.hasPrefix("encoder.layers.") {
                    let parts = newKey.components(separatedBy: ".")
                    if parts.count >= 3, let idx = Int(parts[2]) {
                        newKey = "encoder.l\(idx)." + parts.dropFirst(3).joined(separator: ".")
                    }
                }
                
                // Remap feature_extractor.conv_layers.X to feature_extractor.lX
                if newKey.hasPrefix("feature_extractor.conv_layers.") {
                    let parts = newKey.components(separatedBy: ".")
                    if parts.count >= 3, let idx = Int(parts[2]) {
                        newKey = "feature_extractor.l\(idx)." + parts.dropFirst(3).joined(separator: ".")
                    }
                }
                
                newParams[newKey] = v
            }
            
            // Fix HuBERT PosConv Weight Norm
            let gKey = "encoder.pos_conv_embed.conv.weight_g"
            let vKey = "encoder.pos_conv_embed.conv.weight_v"
            let outKey = "encoder.pos_conv_embed.conv.weight"
            
            if let weight_g_raw = newParams[gKey], let weight_v_raw = newParams[vKey] {
                 // PyTorch weight_v: (Out, In/Groups, Kernel) e.g. (768, 48, 128)
                 // PyTorch weight_g: (1, 1, Kernel) e.g. (1, 1, 128) [Weight Norm dim=2]
                 
                 // Structurally transpose to MLX Conv1d layout (Out, Kernel, In/Groups)
                 let weight_v = weight_v_raw.transposed(axes: [0, 2, 1]) // (768, 128, 48)
                 let weight_g = weight_g_raw.transposed(axes: [0, 2, 1]) // (1, 128, 1)
                 
                 // Norm calculation (PyTorch dim=2 -> MLX axes [0, 2])
                 let v_sqr = weight_v * weight_v
                 let v_sum = v_sqr.sum(axes: [0, 2], keepDims: true) // Result (1, 128, 1)
                 let v_norm = sqrt(v_sum + 1e-12)
                 let weight_normalized = weight_v / v_norm
                 
                 // Fuse
                 let weight_fused = weight_g * weight_normalized
                 
                 newParams[outKey] = weight_fused
                 newParams.removeValue(forKey: gKey)
                 newParams.removeValue(forKey: vKey)
                 log("RVCInference: Fused HuBERT PosConv weights (transposed, dim=2)")
            }
            log("RVCInference: Loaded \(newParams.count) HuBERT weights")
            log("RVCInference: HuBERT sample keys: \(Array(newParams.keys.prefix(5)))")
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

            // Read model config for architecture parameters
            var detectedSR = self.modelSampleRate
            var detectedUpsRates = [10, 10, 2, 2]
            var detectedKernelSizes = [16, 16, 4, 4]

            let configURL = modelURL.deletingPathExtension().appendingPathExtension("json")
            if FileManager.default.fileExists(atPath: configURL.path) {
                log("RVCInference: Found config at \(configURL.lastPathComponent)")
                if let data = try? Data(contentsOf: configURL),
                   let json = try? JSONSerialization.jsonObject(with: data) as? [Any] {
                    if json.count > 17, let sr = json[17] as? Int {
                        detectedSR = sr
                        log("RVCInference: Detected Sample Rate \(detectedSR)Hz")
                    }
                    if json.count > 12, let uArr = json[12] as? [Any] {
                        let u = uArr.compactMap { $0 as? Int }
                        if !u.isEmpty {
                            detectedUpsRates = u
                            log("RVCInference: Detected Upsample Rates \(detectedUpsRates)")
                        }
                    }
                    if json.count > 14, let kArr = json[14] as? [Any] {
                        let k = kArr.compactMap { $0 as? Int }
                        if !k.isEmpty {
                            detectedKernelSizes = k
                            log("RVCInference: Detected Kernel Sizes \(detectedKernelSizes)")
                        }
                    }
                }
            }

            // Initialize Synthesizer with architecture from config
            self.synthesizer = Synthesizer(
                interChannels: 192,
                hiddenChannels: 192,
                filterChannels: 768,
                nHeads: 2,
                nLayers: 6,
                kernelSize: 3,
                pDropout: 0.0,
                embeddingDim: 768,
                speakerEmbedDim: 256,
                ginChannels: 256,
                useF0: true,
                upsampleRates: detectedUpsRates,
                upsampleKernelSizes: detectedKernelSizes,
                sampleRate: detectedSR
            )
            
            // Store sample rate for use during inference
            self.modelSampleRate = detectedSR
            
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

                // 1b. Remap fused resblock conv weights: dec.resblock_X.c1_Y.weight -> dec.resblock_X.c1_Y.conv.weight
                // Swift's Conv1d wrapper has a nested .conv property
                if newK.contains("dec.resblock_") && (newK.contains(".c1_") || newK.contains(".c2_")) {
                    let oldKey = newK
                    if newK.hasSuffix(".weight") && !newK.contains(".conv.") {
                        newK = newK.replacingOccurrences(of: ".weight", with: ".conv.weight")
                        if oldKey.contains("c1_0") && oldKey.contains("resblock_0") {
                            log("DEBUG: Remapped resblock key: \(oldKey) -> \(newK)")
                        }
                    }
                    if newK.hasSuffix(".bias") && !newK.contains(".conv.") {
                        newK = newK.replacingOccurrences(of: ".bias", with: ".conv.bias")
                    }
                }

                // 1c. Remap TextEncoder encoder keys: attn_X -> attn_layers.X, norm1_X -> norm_layers_1.X, etc.
                // Python: setattr(self, f"attn_{i}", l) creates enc_p.encoder.attn_0
                // Swift: uses arrays like enc_p.encoder.attn_layers.0
                if newK.contains("enc_p.encoder.") {
                    // Remap layer names: attn_X -> attn_layers.X, etc.
                    for i in 0..<6 {  // Assuming max 6 layers
                        newK = newK.replacingOccurrences(of: "encoder.attn_\(i).", with: "encoder.attn_layers.\(i).")
                        newK = newK.replacingOccurrences(of: "encoder.norm1_\(i).", with: "encoder.norm_layers_1.\(i).")
                        newK = newK.replacingOccurrences(of: "encoder.ffn_\(i).", with: "encoder.ffn_layers.\(i).")
                        newK = newK.replacingOccurrences(of: "encoder.norm2_\(i).", with: "encoder.norm_layers_2.\(i).")
                    }
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
                // IMPORTANT: Check !contains(".conv.") to prevent double insertion (resblocks handled in section 1b)
                if needsConvInsertion(newK) && !newK.contains(".conv.") {
                    if newK.hasSuffix(".weight") {
                        newK = String(newK.dropLast(7)) + ".conv.weight"
                    } else if newK.hasSuffix(".bias") {
                        newK = String(newK.dropLast(5)) + ".conv.bias"
                    }
                }

                if newK.contains("enc_p.emb_pitch") {
                    log("DEBUG: check - Found Pitch Embedding Key: \(newK)")
                }
                
                synthParams[newK] = newV
            }

            log("RVCInference: About to load \(synthParams.count) parameters into Synthesizer")

            // DEBUG: Show sample of remapped resblock keys
            let resblockKeys = synthParams.keys.filter { $0.contains("resblock_0.c1_0") }.sorted()
            log("DEBUG: Resblock_0.c1_0 keys after remapping: \(resblockKeys)")
            
            // 5. CRITICAL: Fuse weight normalization for ConvTransposed1d layers
            // PyTorch weight normalization: weight = g * (v / ||v||)
            // where g is weight_g and v is weight_v
            for i in 0..<4 {
                let gKey = "dec.up_\(i).weight_g"
                let vKey = "dec.up_\(i).weight_v"
                let outKey = "dec.up_\(i).weight"
                
                if let weight_g = synthParams[gKey], let weight_v = synthParams[vKey] {
                    // Compute L2 norm of weight_v along output channel axis (axis 0)
                    // weight_v shape: (out_channels, kernel_size, in_channels) in MLX format
                    let v_sqr = weight_v * weight_v
                    let v_sum = v_sqr.sum(axes: [1, 2], keepDims: true)  // Sum over kernel and in_channels
                    let v_norm = sqrt(v_sum + 1e-12)  // Add epsilon for numerical stability
                    
                    // Normalize and scale
                    let weight_normalized = weight_v / v_norm
                    let weight_fused = weight_g * weight_normalized
                    
                    synthParams[outKey] = weight_fused
                    log("RVCInference: Fused weight_g \(weight_g.shape) + weight_v \(weight_v.shape) -> \(outKey) \(weight_fused.shape)")
                    
                    // Remove the separate g/v keys
                    synthParams.removeValue(forKey: gKey)
                    synthParams.removeValue(forKey: vKey)
                }
            }
            
            // Also fuse weight normalization for resblock convolutions if present
            for i in 0..<12 {
                for (convPrefix, convCount) in [("c1_", 3), ("c2_", 3)] {
                    for j in 0..<convCount {
                        let base = "dec.resblock_\(i).\(convPrefix)\(j)"
                        let gKey = "\(base).weight_g"
                        let vKey = "\(base).weight_v"
                        let outKey = "\(base).conv.weight"
                        
                        if let weight_g = synthParams[gKey], let weight_v = synthParams[vKey] {
                            // For Conv1d: weight_v shape (out_channels, kernel, in_channels)
                            let v_sqr = weight_v * weight_v
                            let v_sum = v_sqr.sum(axes: [1, 2], keepDims: true)
                            let v_norm = sqrt(v_sum + 1e-12)
                            
                            let weight_normalized = weight_v / v_norm
                            let weight_fused = weight_g * weight_normalized
                            
                            synthParams[outKey] = weight_fused
                            // Remove the separate g/v keys
                            synthParams.removeValue(forKey: gKey)
                            synthParams.removeValue(forKey: vKey)
                        }
                    }
                }
            }
            
            log("RVCInference: After weight fusion: \(synthParams.count) parameters")
            log("RVCInference: Generator up_0 weight shape in file: \(synthParams["dec.up_0.weight"]?.shape ?? [])")

            // DEBUG: Check emb_phone weight in synthParams BEFORE loading
            if let embPhoneWeight = synthParams["enc_p.emb_phone.weight"] {
                MLX.eval(embPhoneWeight)
                log("DEBUG: synthParams[enc_p.emb_phone.weight] BEFORE update: shape=\(embPhoneWeight.shape), range=[\(embPhoneWeight.min().item(Float.self))...\(embPhoneWeight.max().item(Float.self))]")
            } else {
                log("DEBUG: synthParams[enc_p.emb_phone.weight] NOT FOUND!")
            }

            self.synthesizer?.update(parameters: ModuleParameters.unflattened(synthParams))
            self.synthesizer?.train(false)  // CRITICAL: Set to eval mode (disables Dropout, uses BatchNorm running stats)
            log("RVCInference: Successfully loaded Synthesizer with \(synthParams.count) weight keys")

            // DEBUG: Verify weights are loaded correctly
            if let synth = self.synthesizer {
                // Check TextEncoder weights
                let emb_phone = synth.enc_p.emb_phone.weight
                log("DEBUG: enc_p.emb_phone.weight: shape=\(emb_phone.shape), range=[\(emb_phone.min().item(Float.self))...\(emb_phone.max().item(Float.self))]")
                if let emb_pitch = synth.enc_p.emb_pitch {
                    let w = emb_pitch.weight
                    log("DEBUG: enc_p.emb_pitch.weight: shape=\(w.shape), range=[\(w.min().item(Float.self))...\(w.max().item(Float.self))]")
                }

                // Check Generator resblock weights
                let w0 = synth.dec.resblock_0.c1_0.conv.weight
                log("DEBUG: resblock_0.c1_0.conv.weight (kernel=3): shape=\(w0.shape), range=[\(w0.min().item(Float.self))...\(w0.max().item(Float.self))]")
            }
            
            // 3. Load RMVPE (Optional)
            if let rmvpeURL = rmvpeURL {
                do {
                    log("RVCInference: Loading RMVPE from \(rmvpeURL.lastPathComponent)")
                    let rmvpeWeights = try MLX.loadArrays(url: rmvpeURL)
                    self.rmvpe = RMVPE()
                    
                    // Remap RMVPE weight keys (Python MLX format -> Swift)
                    var remappedRMVPE: [String: MLXArray] = [:]
                    for (k, v) in rmvpeWeights {
                        var newKey = k
                        
                        // Skip batch tracking stats
                        if newKey.contains("num_batches_tracked") {
                            continue
                        }
                        
                        // Remap fc.bigru.forward_grus.0 -> bigru.fwd0
                        if newKey.hasPrefix("fc.bigru.forward_grus.0.") {
                            newKey = "bigru.fwd0." + String(newKey.dropFirst("fc.bigru.forward_grus.0.".count))
                        }
                        
                        // Remap fc.bigru.backward_grus.0 -> bigru.bwd0
                        if newKey.hasPrefix("fc.bigru.backward_grus.0.") {
                            newKey = "bigru.bwd0." + String(newKey.dropFirst("fc.bigru.backward_grus.0.".count))
                        }
                        
                        // Remap fc.linear -> linear
                        if newKey.hasPrefix("fc.linear.") {
                            newKey = "linear." + String(newKey.dropFirst("fc.linear.".count))
                        }
                        
                        // Remap blocks.X -> bX (for ResEncoder/Decoder blocks)
                        newKey = newKey.replacingOccurrences(of: ".blocks.0.", with: ".b0.")
                        newKey = newKey.replacingOccurrences(of: ".blocks.1.", with: ".b1.")
                        newKey = newKey.replacingOccurrences(of: ".blocks.2.", with: ".b2.")
                        newKey = newKey.replacingOccurrences(of: ".blocks.3.", with: ".b3.")
                        
                        // Remap unet.encoder.layers.X -> unet.encoder.lX
                        if newKey.contains(".layers.") {
                            newKey = newKey.replacingOccurrences(of: ".layers.0.", with: ".l0.")
                            newKey = newKey.replacingOccurrences(of: ".layers.1.", with: ".l1.")
                            newKey = newKey.replacingOccurrences(of: ".layers.2.", with: ".l2.")
                            newKey = newKey.replacingOccurrences(of: ".layers.3.", with: ".l3.")
                            newKey = newKey.replacingOccurrences(of: ".layers.4.", with: ".l4.")
                        }
                        
                        // Remap BatchNorm running stats (snake_case from PyTorch/SafeTensors -> camelCase for Swift MLX)
                        // This is CRITICAL: unmatched keys mean BN uses init stats (mean=0, var=1), destroying signal scale
                        if newKey.contains("running_mean") {
                            print("DEBUG: Remapping running_mean found: \(newKey)")
                        }
                        newKey = newKey.replacingOccurrences(of: ".running_mean", with: ".runningMean")
                        newKey = newKey.replacingOccurrences(of: ".running_var", with: ".runningVar")
                        
                        if newKey.contains("runningMean") {
                            print("DEBUG: Remapped to runningMean: \(newKey)")
                        }
                        
                        remappedRMVPE[newKey] = v
                    }
                    
                    log("RVCInference: RMVPE sample keys after remapping: \(Array(remappedRMVPE.keys.prefix(5)))")

                    // Check if running stats are in the remapped dict
                    let bnRunningKeys = remappedRMVPE.keys.filter { $0.contains("encoder.bn.running") }
                    log("RVCInference: BN running stat keys to load: \(bnRunningKeys)")
                    if let rmKey = remappedRMVPE["unet.encoder.bn.runningMean"] {
                        log("RVCInference: encoder.bn.runningMean value: \(rmKey.asArray(Float.self))")
                    }
                    if let rvKey = remappedRMVPE["unet.encoder.bn.runningVar"] {
                        log("RVCInference: encoder.bn.runningVar value: \(rvKey.asArray(Float.self))")
                    }

                    self.rmvpe?.update(parameters: ModuleParameters.unflattened(remappedRMVPE))
                    self.rmvpe?.setTrainingMode(false)  // CRITICAL: Set to eval mode for correct inference

                    log("RVCInference: ✅ RMVPE loaded with CustomBatchNorm (\(remappedRMVPE.count) keys)")
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
        
        public func infer(audioURL: URL, outputURL: URL) async {
            do {
                DispatchQueue.main.async { self.status = "Loading Audio..." }
                
                // Load whole audio (Float array size is fine, just not tensors)
                let (audioArray, _) = try AudioProcessor.shared.loadAudio(url: audioURL)
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
                
                // Apply high-pass filter (Butterworth 5th order, 48Hz)
                // Matches Python: signal.butter(5, 48, btype="high", fs=16000) + filtfilt
                audioToProcess = applyButterworthHighPass(audioToProcess)
                
                // Add padding for model context (1s each side, reflect mode)
                // Matches Python: np.pad(audio, (16000, 16000), mode="reflect")
                let padSamples = 1600
                let audioPadded = padReflect(audioToProcess, padding: padSamples)
                
                // Run inference on full audio
                let outputPadded = try await inferChunk(chunk: audioPadded)
                MLX.eval(outputPadded)
                
                // Calculate crop to remove padding from output
                // Input is at 16kHz, output is at model sample rate
                let outputRatio: Float = Float(self.modelSampleRate) / 16000.0
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
                
                let outputSampleRate: Double = Double(self.modelSampleRate)
                
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
            
            // DEBUG: Log audio input stats
            log("DEBUG: Audio input - shape: \(chunk.shape), min: \(chunk.min().item(Float.self)), max: \(chunk.max().item(Float.self)), mean: \(chunk.mean().item(Float.self))")
            
            // Log first 20 audio samples
            let audioSlice = chunk[0..<min(20, chunk.shape[0])].asType(Float.self)
            MLX.eval(audioSlice)
            let audioSamples = audioSlice.asArray(Float.self)
            log("DEBUG: Audio input (padded, filtered) first 20 samples: \(audioSamples)")
             
            // 1. Hubert Feature Extraction (16kHz -> 50fps)
            let audioInput = chunk.expandedDimensions(axis: 0) // [1, T]
            guard let hubertModel = hubertModel else {
                throw NSError(domain: "RVCInference", code: 1, userInfo: [NSLocalizedDescriptionKey: "Hubert model missing"])
            }
            let hubertFeatures = hubertModel(audioInput) // [1, Frames, 768]
            MLX.eval(hubertFeatures)
            MLX.Memory.clearCache()  // MEMORY FIX: Clear after HuBERT
            log("DEBUG: HuBERT output shape: \(hubertFeatures.shape)")
            
            // DEBUG: Log first HuBERT frame's first 5 features
            let hubertSlice = hubertFeatures[0, 0, 0..<5].asType(Float.self)
            MLX.eval(hubertSlice)
            let hubertSamples = hubertSlice.asArray(Float.self)
            log("DEBUG: HuBERT[0,0,:5]: \(hubertSamples)")


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
                
                // 1. F0 (Raw Hz) - use asArray to avoid .item() precondition failure
                let f0Slice = nsff0[0, 0..<min(20, p_len_val), 0].asType(Float.self)
                MLX.eval(f0Slice)
                let f0Values = f0Slice.asArray(Float.self)
                let f0Str = "F0 (First 20): [\(f0Values.map { String(format: "%.4f", $0) }.joined(separator: ", "))]"
                log(f0Str)
                
                // 2. Pitch (Buckets)
                let pitchSlice = pitchFinal[0, 0..<min(20, p_len_val)].asType(Int32.self)
                MLX.eval(pitchSlice)
                let pitchValues = pitchSlice.asArray(Int32.self)
                let pitchStr = "Pitch (First 20): [\(pitchValues.map { String($0) }.joined(separator: ", "))]"
                log(pitchStr)
                
                // 3. Phone (First feature of first 20 frames)
                let phoneSlice = phone[0, 0..<min(20, p_len_val), 0].asType(Float.self)
                MLX.eval(phoneSlice)
                let phoneValues = phoneSlice.asArray(Float.self)
                let phoneStr = "Phone[0] (First 20): [\(phoneValues.map { String(format: "%.4f", $0) }.joined(separator: ", "))]"
                log(phoneStr)
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

        /// Run benchmark: perform inference and compare with reference audio
        public func runBenchmark(audioURL: URL, referenceURL: URL?, outputURL: URL) async throws -> String {
            // Run inference
            await infer(audioURL: audioURL, outputURL: outputURL)

            // If reference provided, compare using Python script
            guard let refURL = referenceURL else {
                return "Inference completed. No reference provided for comparison."
            }

            let task = Process()
            task.executableURL = URL(fileURLWithPath: "/usr/bin/python3")
            task.arguments = [
                "tools/compare_wavs.py",
                refURL.path,
                outputURL.path
            ]

            let pipe = Pipe()
            task.standardOutput = pipe
            task.standardError = pipe

            try task.run()
            task.waitUntilExit()

            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            guard let output = String(data: data, encoding: .utf8) else {
                return "Failed to read comparison output"
            }

            return output
        }
        
        /// Apply high-pass filter: Butterworth 5th order, 48Hz cut-off, 16kHz sample rate
        /// Matches scipy.signal.butter(5, 48, btype='high', fs=16000)
        /// Uses forward-backward filtering (filtfilt) for zero phase shift
        private func applyButterworthHighPass(_ audio: MLXArray) -> MLXArray {
            // Coefficients for butter(5, 48, fs=16000)
            let b: [Double] = [0.9699606451838447, -4.849803225919223, 9.699606451838447, -9.699606451838447, 4.849803225919223, -0.9699606451838447]
            let a: [Double] = [1.0, -4.939001819168364, 9.757863526739543, -9.639544849413458, 4.761506797356209, -0.9408236532054606]
            
            // Extract samples to CPU and convert to Double for precision
            MLX.eval(audio)
            let samplesFloat = audio.asArray(Float.self)
            let samples = samplesFloat.map { Double($0) }
            
            // Helper for simple Direct Form II filtering
            func filter(_ x: [Double]) -> [Double] {
                var y = [Double](repeating: 0, count: x.count)
                
                // Direct Form I difference equation:
                // y[n] = b0*x[n] + ... + b5*x[n-5] - a1*y[n-1] - ... - a5*y[n-5]
                // Assumes a[0] = 1.0
                
                for n in 0..<x.count {
                    // Feedforward
                    var val = b[0] * x[n]
                    if n > 0 { val += b[1] * x[n-1] }
                    if n > 1 { val += b[2] * x[n-2] }
                    if n > 2 { val += b[3] * x[n-3] }
                    if n > 3 { val += b[4] * x[n-4] }
                    if n > 4 { val += b[5] * x[n-5] }
                    
                    // Feedback
                    if n > 0 { val -= a[1] * y[n-1] }
                    if n > 1 { val -= a[2] * y[n-2] }
                    if n > 2 { val -= a[3] * y[n-3] }
                    if n > 3 { val -= a[4] * y[n-4] }
                    if n > 4 { val -= a[5] * y[n-5] }
                    
                    y[n] = val
                }
                return y
            }
            
            // Forward pass
            let y_fwd = filter(samples)
            
            // Backward pass (reverse, filter, reverse)
            let y_rev = Array(y_fwd.reversed())
            let y_back = filter(y_rev)
            let y_final = Array(y_back.reversed())
            
            // Convert back to Float
            let resultFloat = y_final.map { Float($0) }
            return MLXArray(resultFloat)
        }
        
        /// Manual reflect padding matching numpy.pad(mode='reflect')
        /// Pads with the reflection of the vector mirrored on the first and last values of the vector
        private func padReflect(_ audio: MLXArray, padding: Int) -> MLXArray {
            MLX.eval(audio)
            let samples = audio.asArray(Float.self)
            let n = samples.count
            
            guard n > 1 else { return audio }
            
            // Left pad: reverse of samples[1...padding]
            // If padding > n, this simple logic fails, but we assume padding < n (16000 < 30s audio)
            // Python: pad=2, [0,1,2] -> [2,1, 0,1,2]
            
            var leftPad: [Float] = []
            if padding > 0 {
                // indices: 1 to padding
                let start = 1
                let end = min(padding, n - 1)
                if end >= start {
                    leftPad = Array(samples[start...end].reversed())
                }
            }
            
            var rightPad: [Float] = []
            if padding > 0 {
                // indices: (n-1-padding) to (n-2)
                // Python: pad=2, [0,1,2] -> [0,1,2, 1,0]
                // indices reflected around last element (2): 1, 0
                // indices: n-2 down to n-1-padding
                
                let start = max(0, n - 1 - padding)
                let end = n - 2
                if end >= start {
                    rightPad = Array(samples[start...end].reversed())
                }
            }
            
            let result = leftPad + samples + rightPad
            return MLXArray(result)
        }
    }
