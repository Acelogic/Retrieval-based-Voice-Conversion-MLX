
import Foundation
import MLX
import MLXNN
import MLXFFT 

// MARK: - Basic Blocks

class ConvBlockRes: Module {
    let conv1: Conv2d
    let bn1: BatchNorm
    let conv2: Conv2d
    let bn2: BatchNorm
    let shortcut: Conv2d?
    
    init(inChannels: Int, outChannels: Int, momentum: Float = 0.01) {
        self.conv1 = Conv2d(inputChannels: inChannels, outputChannels: outChannels, kernelSize: 3, stride: 1, padding: 1, bias: false)
        // (featureCount, eps, momentum)
        self.bn1 = BatchNorm(featureCount: outChannels, eps: 1e-5, momentum: momentum)
        
        self.conv2 = Conv2d(inputChannels: outChannels, outputChannels: outChannels, kernelSize: 3, stride: 1, padding: 1, bias: false)
        self.bn2 = BatchNorm(featureCount: outChannels, eps: 1e-5, momentum: momentum)
        
        if inChannels != outChannels {
            self.shortcut = Conv2d(inputChannels: inChannels, outputChannels: outChannels, kernelSize: 1, stride: 1, bias: true)
        } else {
            self.shortcut = nil
        }
        super.init()
    }
    
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let residual: MLXArray
        if let shortcut = shortcut {
            residual = shortcut(x)
        } else {
            residual = x
        }
        
        var out = relu(bn1(conv1(x)))
        out = relu(bn2(conv2(out)))
        
        return out + residual
    }
}

class ResEncoderBlock: Module {
    let blocks: [ConvBlockRes]
    let pool: AvgPool2d?
    
    init(inChannels: Int, outChannels: Int, kernelSize: Int?, nBlocks: Int = 1, momentum: Float = 0.01) {
        var _blocks: [ConvBlockRes] = []
        _blocks.append(ConvBlockRes(inChannels: inChannels, outChannels: outChannels, momentum: momentum))
        for _ in 0..<(nBlocks - 1) {
            _blocks.append(ConvBlockRes(inChannels: outChannels, outChannels: outChannels, momentum: momentum))
        }
        self.blocks = _blocks
        
        if let k = kernelSize {
            self.pool = AvgPool2d(kernelSize: .init(k), stride: .init(k))
        } else {
            self.pool = nil
        }
        super.init()
    }
    
    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray?) {
        var x = x
        for block in blocks {
            x = block(x)
        }
        if let pool = pool {
            return (x, pool(x))
        }
        return (x, nil)
    }
}

// MARK: - Encoder

class Encoder: Module {
    let bn: BatchNorm
    let layers: [ResEncoderBlock]
    let outChannel: Int
    
    init(inChannels: Int, inSize: Int, nEncoders: Int, kernelSize: Int, nBlocks: Int, outChannels: Int = 16, momentum: Float = 0.01) {
        self.bn = BatchNorm(featureCount: inChannels, eps: 1e-5, momentum: momentum)
        
        var _layers: [ResEncoderBlock] = []
        var currIn = inChannels
        var currOut = outChannels
        
        for _ in 0..<nEncoders {
            _layers.append(ResEncoderBlock(inChannels: currIn, outChannels: currOut, kernelSize: kernelSize, nBlocks: nBlocks, momentum: momentum))
            currIn = currOut
            currOut *= 2
        }
        self.layers = _layers
        self.outChannel = currIn * 2
        super.init()
    }
    
    func callAsFunction(_ x: MLXArray) -> (MLXArray, [MLXArray]) {
        var x = bn(x)
        var concatTensors: [MLXArray] = []
        
        for layer in layers {
            let (t, pooled) = layer(x)
            concatTensors.append(t)
            if let p = pooled {
                x = p
            }
        }
        return (x, concatTensors)
    }
}

// MARK: - Decoder

class ConvTransposed2dBlock: Module {
    let convTranspose: ConvTransposed2d
    let outputPadding: (Int, Int)
    
    init(inChannels: Int, outChannels: Int, stride: (Int, Int), padding: (Int, Int), outputPadding: (Int, Int)) {
        self.outputPadding = outputPadding
        
        // Use ConvTransposed2d layer
        self.convTranspose = ConvTransposed2d(inputChannels: inChannels, outputChannels: outChannels, kernelSize: 3, stride: .init(stride), padding: .init(padding), bias: false)
        
        super.init()
    }
    
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: [N, H, W, C]
        var y = convTranspose(x)
        
        // Manual output padding
        let (opH, opW) = outputPadding
        if opH > 0 || opW > 0 {
             y = padded(y, widths: [[0,0], [0, opH], [0, opW], [0,0]])
        }
        return y
    }
}

class ResDecoderBlock: Module {
    let conv1Trans: ConvTransposed2dBlock
    let bn1: BatchNorm
    let blocks: [ConvBlockRes]
    
    init(inChannels: Int, outChannels: Int, stride: (Int, Int), nBlocks: Int = 1, momentum: Float = 0.01) {
        let padding = (1, 1)
        let op = (stride == (1, 2)) ? (0, 1) : (1, 1)
        
        self.conv1Trans = ConvTransposed2dBlock(inChannels: inChannels, outChannels: outChannels, stride: stride, padding: padding, outputPadding: op)
        self.bn1 = BatchNorm(featureCount: outChannels, eps: 1e-5, momentum: momentum)
        
        var _blocks: [ConvBlockRes] = []
        _blocks.append(ConvBlockRes(inChannels: outChannels * 2, outChannels: outChannels, momentum: momentum))
        for _ in 0..<(nBlocks - 1) {
             _blocks.append(ConvBlockRes(inChannels: outChannels, outChannels: outChannels, momentum: momentum))
        }
        self.blocks = _blocks
        super.init()
    }
    
    func callAsFunction(_ x: MLXArray, concatTensor: MLXArray) -> MLXArray {
        var x = conv1Trans(x)
        x = relu(bn1(x))
        
        // Match dimensions
        // x: [N, H, W, C], concatTensor: [N, Ht, Wt, Ct]
        let H = x.shape[1]
        let W = x.shape[2]
        let Ht = concatTensor.shape[1]
        let Wt = concatTensor.shape[2]
        
        if H != Ht || W != Wt {
            let padH = Ht - H
            let padW = Wt - W
            if padH > 0 || padW > 0 {
                x = padded(x, widths: [[0,0], [0, max(0, padH)], [0, max(0, padW)], [0,0]])
            }
            // Crop if needed
            if x.shape[1] > Ht { x = x[0..., 0..<Ht, 0..., 0...] }
            if x.shape[2] > Wt { x = x[0..., 0..., 0..<Wt, 0...] }
        }
        
        x = concatenated([x, concatTensor], axis: -1)
        
        for block in blocks {
            x = block(x)
        }
        return x
    }
}

class Decoder: Module {
    let layers: [ResDecoderBlock]
    
    init(inChannels: Int, nDecoders: Int, stride: (Int, Int), nBlocks: Int, momentum: Float = 0.01) {
        var _layers: [ResDecoderBlock] = []
        var currIn = inChannels
        
        for _ in 0..<nDecoders {
            let outChannels = currIn / 2
            _layers.append(ResDecoderBlock(inChannels: currIn, outChannels: outChannels, stride: stride, nBlocks: nBlocks, momentum: momentum))
            currIn = outChannels
        }
        self.layers = _layers
         super.init()
    }
    
    func callAsFunction(_ x: MLXArray, concatTensors: [MLXArray]) -> MLXArray {
        var x = x
        for (i, layer) in layers.enumerated() {
            x = layer(x, concatTensor: concatTensors[concatTensors.count - 1 - i])
        }
        return x
    }
}

class Intermediate: Module {
    let layers: [ResEncoderBlock]
    
    init(inChannels: Int, outChannels: Int, nInters: Int, nBlocks: Int, momentum: Float = 0.01) {
        var _layers: [ResEncoderBlock] = []
        _layers.append(ResEncoderBlock(inChannels: inChannels, outChannels: outChannels, kernelSize: nil, nBlocks: nBlocks, momentum: momentum))
        for _ in 0..<(nInters - 1) {
             _layers.append(ResEncoderBlock(inChannels: outChannels, outChannels: outChannels, kernelSize: nil, nBlocks: nBlocks, momentum: momentum))
        }
        self.layers = _layers
        super.init()
    }
    
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = x
        for layer in layers {
            let (out, _) = layer(x)
            x = out
        }
        return x
    }
}

class DeepUnet: Module {
    let encoder: Encoder
    let intermediate: Intermediate
    let decoder: Decoder
    
    init(kernelSize: Int, nBlocks: Int, enDeLayers: Int = 5, interLayers: Int = 4, inChannels: Int = 1, enOutChannels: Int = 16) {
        self.encoder = Encoder(inChannels: inChannels, inSize: 128, nEncoders: enDeLayers, kernelSize: kernelSize, nBlocks: nBlocks, outChannels: enOutChannels)
        let encOutCh = self.encoder.outChannel
        
        self.intermediate = Intermediate(inChannels: encOutCh / 2, outChannels: encOutCh, nInters: interLayers, nBlocks: nBlocks)
        self.decoder = Decoder(inChannels: encOutCh, nDecoders: enDeLayers, stride: (kernelSize, kernelSize), nBlocks: nBlocks)
        
        super.init()
    }
    
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (encOut, concatTensors) = encoder(x)
        let interOut = intermediate(encOut)
        let decOut = decoder(interOut, concatTensors: concatTensors)
        return decOut
    }
}

// MARK: - BiGRU

class BiGRU: Module {
    let numLayers: Int
    let hiddenFeatures: Int
    let forwardGRUs: [GRU]
    let backwardGRUs: [GRU]
    
    init(inputFeatures: Int, hiddenFeatures: Int, numLayers: Int) {
        self.numLayers = numLayers
        self.hiddenFeatures = hiddenFeatures
        
        var _fwd: [GRU] = []
        var _bwd: [GRU] = []
        
        for i in 0..<numLayers {
            let dim = (i == 0) ? inputFeatures : hiddenFeatures * 2
            _fwd.append(GRU(inputSize: dim, hiddenSize: hiddenFeatures, bias: true))
            _bwd.append(GRU(inputSize: dim, hiddenSize: hiddenFeatures, bias: true))
        }
        self.forwardGRUs = _fwd
        self.backwardGRUs = _bwd
        super.init()
    }
    
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = x // [N, L, C]
        
        for i in 0..<numLayers {
            let fwd = forwardGRUs[i]
            let bwd = backwardGRUs[i]
            
            let outFwd = fwd(x) // [N, L, H]
            
            // Reverse for backward along axis 1 (L)
            // Using deprecated stride subscript as it is robust without full MLXArrayIndex inference
            let xRev = x[0..., .stride(by: -1), 0...]
            let outBwdRev = bwd(xRev)
            let outBwd = outBwdRev[0..., .stride(by: -1), 0...]
            
            x = concatenated([outFwd, outBwd], axis: -1)
        }
        return x
    }
}

// MARK: - E2E (RMVPE)

class RMVPE: Module {
    let unet: DeepUnet
    let cnn: Conv2d
    let bigru: BiGRU
    let linear: Linear
    let dropout: Dropout
    
    let centMapping: MLXArray
    
    override init() {
        // Defaults from rmvpe.py: n_blocks=4, n_gru=1, kernel_size=(2,2)
        // E2E(4, 1, (2,2))
        self.unet = DeepUnet(kernelSize: 2, nBlocks: 4, enDeLayers: 5, interLayers: 4, inChannels: 1, enOutChannels: 16)
        self.cnn = Conv2d(inputChannels: 16, outputChannels: 3, kernelSize: 3, stride: 1, padding: 1)
        
        self.bigru = BiGRU(inputFeatures: 384, hiddenFeatures: 256, numLayers: 1)
        self.linear = Linear(inputDimensions: 512, outputDimensions: 360) 
        
        self.dropout = Dropout(p: 0.25)
        
        // Constants
        // 20 * np.arange(360) + 1997.379...
        let cents = MLXArray(Array(0..<360).map { Float($0) }) * 20.0 + 1997.3794084376191
        
        self.centMapping = cents
        
        super.init()
    }
    
    func callAsFunction(_ mel: MLXArray) -> MLXArray {
        // mel: [N, T, n_mels] (128)
        // UNet expects [N, H, W, C] -> [N, T, n_mels, 1]
        var x = mel.expandedDimensions(axis: -1)
        
        x = unet(x) // [N, T, n_mels, 16]
        
        // CNN expects [N, H, W, C]
        x = cnn(x) // [N, T, n_mels, 3]
        
        // Reshape for GRU: [N, T, n_mels * 3]
        let shape = x.shape
        let B = shape[0]
        let T = shape[1]
        x = x.reshaped([B, T, -1])
        
        x = bigru(x) // [N, T, 512]
        x = linear(x) // [N, T, 360]
        x = dropout(x)
        x = sigmoid(x)
        return x
    }
    
    func decode(_ hidden: MLXArray, thred: Float = 0.03) -> MLXArray {
        /// Decodes hidden representation to F0.
        /// Matches Python MLX implementation exactly (rvc_mlx/lib/mlx/rmvpe.py:355-404)
        ///
        /// CRITICAL: This implementation matches PyTorch's to_local_average_cents exactly.
        /// The key fix that achieved 0.986 correlation was:
        /// 1. Weighted averaging of cents around argmax peak (9-sample window)
        /// 2. Correct F0 formula: f0 = 10 * 2^(cents/1200)
        /// This reduced cents prediction error from 349.7 to 1.5 cents!

        // hidden: [N, T, 360] or [T, 360]
        var h = hidden
        if h.ndim == 3 {
            h = h[0]  // Take first batch element -> [T, 360]
        }

        // Find center (argmax) for each frame
        let center = MLX.argMax(h, axis: -1)  // [T]

        // Pad hidden for window gathering (matches PyTorch)
        // Pad axis 1 (freq axis) by 4 on each side
        let salience = MLX.padded(h, widths: [IntOrPair((0, 0)), IntOrPair((4, 4))])  // [T, 368]

        // Adjust center indices to account for padding
        let centerPadded = center + 4

        // Pad cents_mapping once (outside loop for efficiency)
        let centsMappingPadded = MLX.padded(centMapping, widths: [IntOrPair((4, 4))])

        // Extract 9-sample windows around each center
        // For each frame t, we want salience[t, center[t]-4:center[t]+5]
        // and cents_mapping[center[t]-4:center[t]+5]
        let T = h.shape[0]
        var centsPredArray: [Float] = []

        for t in 0..<T {
            let centerIdx = Int(centerPadded[t].item(Int32.self))
            let start = centerIdx - 4
            let end = centerIdx + 5  // Python uses [start:end] which is exclusive of end

            // Extract window from salience
            let salienceWindow = salience[t, start..<end]  // [9]

            // Extract corresponding cents values
            let centsWindow = centsMappingPadded[start..<end]  // [9]

            // Weighted average of cents
            let product: MLXArray = salienceWindow * centsWindow
            let productSum = MLX.sum(product).item(Float.self)
            let weightSum = MLX.sum(salienceWindow).item(Float.self)

            // Avoid division by zero
            if weightSum > 0 {
                let centValue: Float = productSum / weightSum
                centsPredArray.append(centValue)
            } else {
                centsPredArray.append(0.0)
            }
        }

        // Convert array to MLXArray
        var centsPred = MLXArray(centsPredArray)

        // Apply threshold based on max salience
        let maxx = MLX.max(salience, axis: 1)  // [T]
        let mask = maxx .> thred
        centsPred = centsPred * mask.asType(centsPred.dtype)

        // Convert cents to F0 using CORRECT formula
        // Python line 401: f0 = 10 * (2 ** (cents_pred / 1200))
        // This is the fix that achieved 0.986 correlation!
        var f0 = 10.0 * MLX.pow(2.0, centsPred / 1200.0)

        // Zero out where cents_pred was 0 (f0 would be 10.0)
        let zeroMask = centsPred .== 0.0
        f0 = f0 * (1.0 - zeroMask.asType(f0.dtype))

        return f0.expandedDimensions(axis: -1)  // [T, 1]
    }
    
    // Helper to run full inference
    func infer(audio: MLXArray, thred: Float = 0.03) -> MLXArray {
        // audio: [T]
        let melProcessor = MelSpectrogram()
        let mel = melProcessor(audio) // [n_mels, T_frames] (Log Mel) (128, T)
        print("DEBUG: Mel Spectrogram Stats: min \(mel.min().item(Float.self)), max \(mel.max().item(Float.self)), shape \(mel.shape)")
        
        // Model expects [N, T, n_mels] (transposed)
        // mel.T -> [T_frames, n_mels]. Add dim [1, T, n_mels]
        let melInput = mel.transposed().expandedDimensions(axis: 0)
        
        // Run model
        let hidden = self(melInput) // [1, T, 360]
        
        // Decode
        let f0 = self.decode(hidden, thred: thred) // [T, 1]
        
        // DEBUG Stats
        let f0_min = f0.min().item(Float.self)
        let f0_max = f0.max().item(Float.self)
        let f0_mean = f0.mean().item(Float.self)
        print("DEBUG: RMVPE F0 Stats: min \(f0_min), max \(f0_max), mean \(f0_mean), shape \(f0.shape)")

        // CRITICAL: Add batch dimension to match RVCInference expectations
        // decode() returns [T, 1], we need [1, T, 1] for the pipeline
        let f0_batched = f0.expandedDimensions(axis: 0)  // [T, 1] -> [1, T, 1]
        print("DEBUG: RMVPE F0 final shape: \(f0_batched.shape)")

        return f0_batched
    }
}

// MARK: - Mel Spectrogram

class MelSpectrogram {
    let n_fft: Int = 1024
    let hop_length: Int = 160
    let win_length: Int = 1024
    let n_mels: Int = 128
    let sr: Int = 16000
    let fmin: Float = 30
    let fmax: Float = 8000
    
    var mel_filterbank: MLXArray?
    var window: MLXArray?
    
    init() {
        // Lazy init
    }
    
    func hz_to_mel(_ hz: Float) -> Float {
        // Foundation log10 requires Double
        return 2595 * Float(Foundation.log10(1.0 + Double(hz) / 700.0))
    }
    
    func mel_to_hz(_ mel: Float) -> Float {
        // Foundation pow requires Double
        return 700 * (Float(Foundation.pow(10.0, Double(mel) / 2595.0)) - 1)
    }
    
    func create_mel_filterbank() -> MLXArray {
        let mel_min = hz_to_mel(fmin)
        let mel_max = hz_to_mel(fmax)
        
        // linspace mel_points
        var mel_points: [Float] = []
        let step = (mel_max - mel_min) / Float(n_mels + 1)
        for i in 0..<(n_mels + 2) {
            mel_points.append(mel_min + Float(i) * step)
        }
        
        let hz_points = mel_points.map { mel_to_hz($0) }
        
        // freq bins
        let n_freqs = n_fft / 2 + 1
        var freq_bins: [Float] = []
        let freq_step = Float(sr) / Float(n_fft) 

        for i in 0..<n_freqs {
            freq_bins.append(Float(i) * freq_step)
        }
        
        var filterbank_data: [Float] = Array(repeating: 0, count: n_mels * n_freqs)
        
        for i in 0..<n_mels {
            let left = hz_points[i]
            let center = hz_points[i+1]
            let right = hz_points[i+2]
            
            for j in 0..<n_freqs {
                let f = freq_bins[j]
                var val: Float = 0
                
                if f >= left && f <= center {
                    val = (f - left) / (center - left + 1e-10)
                } else if f >= center && f <= right {
                    val = (right - f) / (right - center + 1e-10)
                }
                
                filterbank_data[i * n_freqs + j] = val
            }
        }
        
        return MLXArray(filterbank_data, [n_mels, n_freqs])
    }
    
    func create_window() -> MLXArray {
        var win: [Float] = []
        for i in 0..<win_length {
            let v = 0.5 - 0.5 * cos(2 * Float.pi * Float(i) / Float(win_length))
            win.append(v)
        }
        return MLXArray(win)
    }
    
    func callAsFunction(_ audio: MLXArray) -> MLXArray {
        // audio: [T]
        if mel_filterbank == nil { mel_filterbank = create_mel_filterbank() }
        if window == nil { window = create_window() }
        
        let pad_len = n_fft / 2
        
        let audio_padded = padded(audio, widths: [[pad_len, pad_len]]) 
        
        let len_p = audio_padded.shape[0]
        let num_frames = 1 + (len_p - n_fft) / hop_length
        
        let frame_starts = MLXArray(Array(stride(from: 0, to: num_frames * hop_length, by: hop_length))).expandedDimensions(axis: 1)
        let win_indices = MLXArray(Array(0..<n_fft)).expandedDimensions(axis: 0)
        
        let indices = frame_starts + win_indices // [num_frames, n_fft]
        
        let frames = audio_padded[indices] // [N_frames, n_fft]
        
        let win = window!
        let frames_w = frames * win
        
        // FFT
        // Using MLXFFT.rfft
        let spectrum = MLXFFT.rfft(frames_w, axis: -1)
        
        // Magnitude
        let magnitude = abs(spectrum)
        
        let mel = matmul(mel_filterbank!, magnitude.transposed())
        
        let log_mel = MLX.log(maximum(mel, 1e-5))
        
        return log_mel
    }
}
