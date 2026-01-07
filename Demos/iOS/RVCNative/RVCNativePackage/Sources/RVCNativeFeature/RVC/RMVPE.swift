
import Foundation
import MLX
import MLXNN
import MLXFFT

// MARK: - Custom BatchNorm with Running Stats Support

/// Custom BatchNorm that properly supports loading running_mean and running_var
/// MLX Swift's default BatchNorm doesn't expose running stats via parameters()
class CustomBatchNorm: Module {
    var weight: MLXArray
    var bias: MLXArray
    var runningMean: MLXArray
    var runningVar: MLXArray
    let eps: Float
    let momentum: Float
    var isTraining: Bool = true

    init(featureCount: Int, eps: Float = 1e-5, momentum: Float = 0.1) {
        self.eps = eps
        self.momentum = momentum

        // Initialize parameters (trainable)
        self.weight = MLXArray.ones([featureCount])
        self.bias = MLXArray.zeros([featureCount])

        // Initialize running stats (non-trainable but loadable)
        self.runningMean = MLXArray.zeros([featureCount])
        self.runningVar = MLXArray.ones([featureCount])

        super.init()
    }

    func setTrainingMode(_ mode: Bool) {
        self.isTraining = mode
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x shape: [N, H, W, C] for Conv2d
        // Normalize over the channel dimension (last axis)

        if isTraining {
            // Training mode: use batch statistics
            let axes = Array(0..<(x.ndim - 1))
            let mean = x.mean(axes: axes, keepDims: true)
            let variance = ((x - mean) * (x - mean)).mean(axes: axes, keepDims: true)

            // Update running stats
            let squeezedMean = mean.squeezed()
            let squeezedVar = variance.squeezed()
            runningMean = runningMean * (1 - momentum) + squeezedMean * momentum
            runningVar = runningVar * (1 - momentum) + squeezedVar * momentum

            // Normalize
            let normalized = (x - mean) / sqrt(variance + eps)
            return normalized * weight + bias
        } else {
            // Eval mode: use running statistics
            // Reshape running stats to match input [1, 1, 1, C] for broadcasting
            let shape = [Int](repeating: 1, count: x.ndim - 1) + [x.shape.last!]
            let mean = runningMean.reshaped(shape)
            let variance = runningVar.reshaped(shape)

            // Normalize
            let normalized = (x - mean) / sqrt(variance + eps)

            // Reshape weight and bias for broadcasting
            let w = weight.reshaped(shape)
            let b = bias.reshaped(shape)

            return normalized * w + b
        }
    }
}

// MARK: - Basic Blocks

class ConvBlockRes: Module {
    let conv1: Conv2d
    let bn1: CustomBatchNorm
    let conv2: Conv2d
    let bn2: CustomBatchNorm
    let shortcut: Conv2d?

    init(inChannels: Int, outChannels: Int, momentum: Float = 0.01) {
        self.conv1 = Conv2d(inputChannels: inChannels, outputChannels: outChannels, kernelSize: 3, stride: 1, padding: 1, bias: false)
        // (featureCount, eps, momentum)
        self.bn1 = CustomBatchNorm(featureCount: outChannels, eps: 1e-5, momentum: momentum)

        self.conv2 = Conv2d(inputChannels: outChannels, outputChannels: outChannels, kernelSize: 3, stride: 1, padding: 1, bias: false)
        self.bn2 = CustomBatchNorm(featureCount: outChannels, eps: 1e-5, momentum: momentum)
        
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
    let b0, b1, b2, b3: ConvBlockRes?
    let pool: AvgPool2d?
    let nBlocks: Int
    
    init(inChannels: Int, outChannels: Int, kernelSize: Int?, nBlocks: Int = 1, momentum: Float = 0.01) {
        self.nBlocks = nBlocks
        
        self.b0 = ConvBlockRes(inChannels: inChannels, outChannels: outChannels, momentum: momentum)
        self.b1 = nBlocks > 1 ? ConvBlockRes(inChannels: outChannels, outChannels: outChannels, momentum: momentum) : nil
        self.b2 = nBlocks > 2 ? ConvBlockRes(inChannels: outChannels, outChannels: outChannels, momentum: momentum) : nil
        self.b3 = nBlocks > 3 ? ConvBlockRes(inChannels: outChannels, outChannels: outChannels, momentum: momentum) : nil
        
        if let k = kernelSize {
            self.pool = AvgPool2d(kernelSize: .init(k), stride: .init(k))
        } else {
            self.pool = nil
        }
        super.init()
    }
    
    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray?) {
        var x = x
        if let b = b0 { x = b(x) }
        if let b = b1 { x = b(x) }
        if let b = b2 { x = b(x) }
        if let b = b3 { x = b(x) }
        
        if let pool = pool {
            return (x, pool(x))
        }
        return (x, nil)
    }
}

// MARK: - Encoder

class Encoder: Module {
    let bn: CustomBatchNorm
    let l0, l1, l2, l3, l4: ResEncoderBlock
    let outChannel: Int

    init(inChannels: Int, inSize: Int, nEncoders: Int, kernelSize: Int, nBlocks: Int, outChannels: Int = 16, momentum: Float = 0.01) {
        // CRITICAL FIX: Match Python eps=1e-5 (was 1e-3)
        self.bn = CustomBatchNorm(featureCount: inChannels, eps: 1e-5, momentum: momentum)

        var cIn = inChannels
        var cOut = outChannels
        
        self.l0 = ResEncoderBlock(inChannels: cIn, outChannels: cOut, kernelSize: kernelSize, nBlocks: nBlocks, momentum: momentum)
        cIn = cOut; cOut *= 2
        
        self.l1 = ResEncoderBlock(inChannels: cIn, outChannels: cOut, kernelSize: kernelSize, nBlocks: nBlocks, momentum: momentum)
        cIn = cOut; cOut *= 2
        
        self.l2 = ResEncoderBlock(inChannels: cIn, outChannels: cOut, kernelSize: kernelSize, nBlocks: nBlocks, momentum: momentum)
        cIn = cOut; cOut *= 2
        
        self.l3 = ResEncoderBlock(inChannels: cIn, outChannels: cOut, kernelSize: kernelSize, nBlocks: nBlocks, momentum: momentum)
        cIn = cOut; cOut *= 2
        
        self.l4 = ResEncoderBlock(inChannels: cIn, outChannels: cOut, kernelSize: kernelSize, nBlocks: nBlocks, momentum: momentum)
        cIn = cOut; cOut *= 2
        
        self.outChannel = cIn
        super.init()
    }
    
    func callAsFunction(_ x: MLXArray) -> (MLXArray, [MLXArray]) {
        var x = bn(x)
        print("DEBUG RMVPE Encoder: After input BN - min \(x.min().item(Float.self)), max \(x.max().item(Float.self)), mean \(x.mean().item(Float.self))")
        var concatTensors: [MLXArray] = []

        let layers = [l0, l1, l2, l3, l4]
        for (idx, layer) in layers.enumerated() {
            let (t, pooled) = layer(x)
            print("DEBUG RMVPE Encoder: Layer \(idx) output - min \(t.min().item(Float.self)), max \(t.max().item(Float.self)), mean \(t.mean().item(Float.self))")
            concatTensors.append(t)
            if let p = pooled {
                x = p
                print("DEBUG RMVPE Encoder: Layer \(idx) pooled - min \(p.min().item(Float.self)), max \(p.max().item(Float.self)), mean \(p.mean().item(Float.self))")
            } else {
                x = t
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
    let bn1: CustomBatchNorm
    let b0, b1, b2, b3: ConvBlockRes?
    let nBlocks: Int

    init(inChannels: Int, outChannels: Int, stride: (Int, Int), nBlocks: Int = 1, momentum: Float = 0.01) {
        let padding = (1, 1)
        let op = (stride == (1, 2)) ? (0, 1) : (1, 1)

        self.conv1Trans = ConvTransposed2dBlock(inChannels: inChannels, outChannels: outChannels, stride: stride, padding: padding, outputPadding: op)
        self.bn1 = CustomBatchNorm(featureCount: outChannels, eps: 1e-5, momentum: momentum)
        
        self.nBlocks = nBlocks
        self.b0 = ConvBlockRes(inChannels: outChannels * 2, outChannels: outChannels, momentum: momentum)
        self.b1 = nBlocks > 1 ? ConvBlockRes(inChannels: outChannels, outChannels: outChannels, momentum: momentum) : nil
        self.b2 = nBlocks > 2 ? ConvBlockRes(inChannels: outChannels, outChannels: outChannels, momentum: momentum) : nil
        self.b3 = nBlocks > 3 ? ConvBlockRes(inChannels: outChannels, outChannels: outChannels, momentum: momentum) : nil
        
        super.init()
    }
    
    func callAsFunction(_ x: MLXArray, concatTensor: MLXArray) -> MLXArray {
        var x = conv1Trans(x)
        x = relu(bn1(x))
        
        // Match dimensions
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
            if x.shape[1] > Ht { x = x[0..., 0..<Ht, 0..., 0...] }
            if x.shape[2] > Wt { x = x[0..., 0..., 0..<Wt, 0...] }
        }
        
        x = concatenated([x, concatTensor], axis: -1)
        
        if let b = b0 { x = b(x) }
        if let b = b1 { x = b(x) }
        if let b = b2 { x = b(x) }
        if let b = b3 { x = b(x) }
        
        return x
    }
}

class Decoder: Module {
    let l0, l1, l2, l3, l4: ResDecoderBlock
    
    init(inChannels: Int, nDecoders: Int, stride: (Int, Int), nBlocks: Int, momentum: Float = 0.01) {
        var cIn = inChannels
        var cOut = cIn / 2
        
        self.l0 = ResDecoderBlock(inChannels: cIn, outChannels: cOut, stride: stride, nBlocks: nBlocks, momentum: momentum)
        cIn = cOut; cOut = cIn / 2
        
        self.l1 = ResDecoderBlock(inChannels: cIn, outChannels: cOut, stride: stride, nBlocks: nBlocks, momentum: momentum)
        cIn = cOut; cOut = cIn / 2
        
        self.l2 = ResDecoderBlock(inChannels: cIn, outChannels: cOut, stride: stride, nBlocks: nBlocks, momentum: momentum)
        cIn = cOut; cOut = cIn / 2
        
        self.l3 = ResDecoderBlock(inChannels: cIn, outChannels: cOut, stride: stride, nBlocks: nBlocks, momentum: momentum)
        cIn = cOut; cOut = cIn / 2
        
        self.l4 = ResDecoderBlock(inChannels: cIn, outChannels: cOut, stride: stride, nBlocks: nBlocks, momentum: momentum)
        
        super.init()
    }
    
    func callAsFunction(_ x: MLXArray, concatTensors: [MLXArray]) -> MLXArray {
        var x = x
        x = l0(x, concatTensor: concatTensors[4])
        x = l1(x, concatTensor: concatTensors[3])
        x = l2(x, concatTensor: concatTensors[2])
        x = l3(x, concatTensor: concatTensors[1])
        x = l4(x, concatTensor: concatTensors[0])
        return x
    }
}

class Intermediate: Module {
    let l0, l1, l2, l3: ResEncoderBlock
    
    init(inChannels: Int, outChannels: Int, nInters: Int, nBlocks: Int, momentum: Float = 0.01) {
        self.l0 = ResEncoderBlock(inChannels: inChannels, outChannels: outChannels, kernelSize: nil, nBlocks: nBlocks, momentum: momentum)
        self.l1 = ResEncoderBlock(inChannels: outChannels, outChannels: outChannels, kernelSize: nil, nBlocks: nBlocks, momentum: momentum)
        self.l2 = ResEncoderBlock(inChannels: outChannels, outChannels: outChannels, kernelSize: nil, nBlocks: nBlocks, momentum: momentum)
        self.l3 = ResEncoderBlock(inChannels: outChannels, outChannels: outChannels, kernelSize: nil, nBlocks: nBlocks, momentum: momentum)
        super.init()
    }
    
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = x
        x = l0(x).0
        x = l1(x).0
        x = l2(x).0
        x = l3(x).0
        return x
    }
}

class DeepUnet: Module {
    let encoder: Encoder
    let intermediate: Intermediate
    let decoder: Decoder
    
    init(kernelSize: Int, nBlocks: Int, enDeLayers: Int = 5, interLayers: Int = 4, inChannels: Int = 1, enOutChannels: Int = 16) {
        self.encoder = Encoder(inChannels: inChannels, inSize: 128, nEncoders: enDeLayers, kernelSize: kernelSize, nBlocks: nBlocks, outChannels: enOutChannels)
        let encOutCh = self.encoder.outChannel // 256
        
        // Python doubles channels here: Intermediate(256, 512)
        self.intermediate = Intermediate(inChannels: encOutCh, outChannels: encOutCh * 2, nInters: interLayers, nBlocks: nBlocks)
        // Decoder starts with the doubled channel count: Decoder(512)
        self.decoder = Decoder(inChannels: encOutCh * 2, nDecoders: enDeLayers, stride: (kernelSize, kernelSize), nBlocks: nBlocks)
        
        super.init()
    }
    
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        print("DEBUG RMVPE: DeepUnet input shape \(x.shape)")
        let (encOut, concatTensors) = encoder(x)
        print("DEBUG RMVPE: Encoder output shape \(encOut.shape), concats: \(concatTensors.map { $0.shape })")
        let interOut = intermediate(encOut)
        print("DEBUG RMVPE: Intermediate output shape \(interOut.shape)")
        let decOut = decoder(interOut, concatTensors: concatTensors)
        print("DEBUG RMVPE: Decoder output shape \(decOut.shape)")
        return decOut
    }
}

// MARK: - PyTorchGRU (matches PyTorch's exact formula)

/// GRU implementation that exactly matches PyTorch's GRU formula:
/// r_t = σ(W_ir @ x_t + b_ir + W_hr @ h_{t-1} + b_hr)
/// z_t = σ(W_iz @ x_t + b_iz + W_hz @ h_{t-1} + b_hz)
/// n_t = tanh(W_in @ x_t + b_in + r_t * (W_hn @ h_{t-1} + b_hn))
/// h_t = (1 - z_t) * n_t + z_t * h_{t-1}
class PyTorchGRU: Module {
    let inputSize: Int
    let hiddenSize: Int
    
    // Parameters - must be var or let property for MLX registration
    var weight_ih: MLXArray
    var weight_hh: MLXArray
    var bias_ih: MLXArray?
    var bias_hh: MLXArray?
    
    init(inputSize: Int, hiddenSize: Int, bias: Bool = true) {
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        
        let scale = 1.0 / sqrt(Float(hiddenSize))
        self.weight_ih = MLXRandom.uniform(low: -scale, high: scale, [3 * hiddenSize, inputSize])
        self.weight_hh = MLXRandom.uniform(low: -scale, high: scale, [3 * hiddenSize, hiddenSize])
        
        if bias {
            self.bias_ih = MLX.zeros([3 * hiddenSize])
            self.bias_hh = MLX.zeros([3 * hiddenSize])
        } else {
            self.bias_ih = nil
            self.bias_hh = nil
        }
        
        super.init()
    }
    
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: [B, L, D]
        let B = x.shape[0]
        let L = x.shape[1]
        let H = hiddenSize
        
        // Initialize hidden state
        var h = MLX.zeros([B, H])
        
        var outputs: [MLXArray] = []
        
        for t in 0..<L {
            let x_t = x[0..., t, 0...]  // [B, D]
            
            // Input projection: [B, 3*H]
            var gi = matmul(x_t, weight_ih.T)
            if let bias_ih = bias_ih {
                gi = gi + bias_ih
            }
            
            // Hidden projection: [B, 3*H]
            var gh = matmul(h, weight_hh.T)
            if let bias_hh = bias_hh {
                gh = gh + bias_hh
            }
            
            // Split into gates (r, z, n order matches PyTorch)
            let i_r = gi[0..., 0..<H]
            let i_z = gi[0..., H..<(2*H)]
            let i_n = gi[0..., (2*H)...]
            
            let h_r = gh[0..., 0..<H]
            let h_z = gh[0..., H..<(2*H)]
            let h_n = gh[0..., (2*H)...]
            
            // Reset gate
            let r_t = sigmoid(i_r + h_r)
            
            // Update gate
            let z_t = sigmoid(i_z + h_z)
            
            // New gate
            let n_t = tanh(i_n + r_t * h_n)
            
            // New hidden state
            h = (1 - z_t) * n_t + z_t * h
            
            outputs.append(h)
        }
        
        // Stack outputs: [B, L, H]
        return stacked(outputs, axis: 1)
    }
}

// MARK: - BiGRU (using PyTorchGRU)

class BiGRU: Module {
    let numLayers: Int
    let hiddenFeatures: Int
    
    // Explicit properties for 1 layer (standard for RMVPE)
    let fwd0: PyTorchGRU
    let bwd0: PyTorchGRU
    
    init(inputFeatures: Int, hiddenFeatures: Int, numLayers: Int) {
        self.numLayers = numLayers
        self.hiddenFeatures = hiddenFeatures
        
        self.fwd0 = PyTorchGRU(inputSize: inputFeatures, hiddenSize: hiddenFeatures, bias: true)
        self.bwd0 = PyTorchGRU(inputSize: inputFeatures, hiddenSize: hiddenFeatures, bias: true)
        
        super.init()
    }
    
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = x // [N, L, C]
        
        let outFwd = fwd0(x) // [N, L, H]
        
        // Reverse for backward along axis 1 (L)
        let xRev = x[0..., .stride(by: -1), 0...]
        let outBwdRev = bwd0(xRev)
        let outBwd = outBwdRev[0..., .stride(by: -1), 0...]
        
        x = concatenated([outFwd, outBwd], axis: -1)
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

    /// Set training mode including custom BatchNorm layers
    func setTrainingMode(_ mode: Bool) {
        self.train(mode)  // Call parent's train()

        // Manually set training mode on all CustomBatchNorm instances in the UNet
        // Since recursive traversal of NestedItem is complex, we access known paths directly
        unet.encoder.bn.setTrainingMode(mode)

        // Set training mode for all layers in encoder/intermediate/decoder
        setLayerTrainingMode(mode, layers: [
            unet.encoder.l0, unet.encoder.l1, unet.encoder.l2, unet.encoder.l3, unet.encoder.l4
        ])
        setLayerTrainingMode(mode, layers: [
            unet.intermediate.l0, unet.intermediate.l1, unet.intermediate.l2, unet.intermediate.l3
        ])
        setLayerTrainingMode(mode, layers: [
            unet.decoder.l0, unet.decoder.l1, unet.decoder.l2, unet.decoder.l3, unet.decoder.l4
        ])
    }

    private func setLayerTrainingMode(_ mode: Bool, layers: [Module]) {
        for layer in layers {
            if let encLayer = layer as? ResEncoderBlock {
                setResEncoderBlockTrainingMode(mode, block: encLayer)
            } else if let decLayer = layer as? ResDecoderBlock {
                setResDecoderBlockTrainingMode(mode, block: decLayer)
            }
        }
    }

    private func setResEncoderBlockTrainingMode(_ mode: Bool, block: ResEncoderBlock) {
        if let b0 = block.b0 { setConvBlockResTrainingMode(mode, block: b0) }
        if let b1 = block.b1 { setConvBlockResTrainingMode(mode, block: b1) }
        if let b2 = block.b2 { setConvBlockResTrainingMode(mode, block: b2) }
        if let b3 = block.b3 { setConvBlockResTrainingMode(mode, block: b3) }
    }

    private func setResDecoderBlockTrainingMode(_ mode: Bool, block: ResDecoderBlock) {
        block.bn1.setTrainingMode(mode)
        if let b0 = block.b0 { setConvBlockResTrainingMode(mode, block: b0) }
        if let b1 = block.b1 { setConvBlockResTrainingMode(mode, block: b1) }
        if let b2 = block.b2 { setConvBlockResTrainingMode(mode, block: b2) }
        if let b3 = block.b3 { setConvBlockResTrainingMode(mode, block: b3) }
    }

    private func setConvBlockResTrainingMode(_ mode: Bool, block: ConvBlockRes) {
        block.bn1.setTrainingMode(mode)
        block.bn2.setTrainingMode(mode)
    }

    func callAsFunction(_ mel: MLXArray) -> MLXArray {
        // mel: [1, T, n_mels, 1] (from infer())
        var x = mel
        
        x = unet(x) // [N, T, n_mels, 16]
        print("DEBUG: RMVPE UNet output stats: min \(x.min().item(Float.self)), max \(x.max().item(Float.self)), mean \(x.mean().item(Float.self))")
        
        // CNN expects [N, H, W, C]
        x = cnn(x) // [N, T, n_mels, 3]
        print("DEBUG: RMVPE CNN output stats: min \(x.min().item(Float.self)), max \(x.max().item(Float.self)), mean \(x.mean().item(Float.self))")
        
        // Reshape for GRU: [N, T, n_mels * 3]
        // CRITICAL: Transpose to [N, T, 3, n_mels] BEFORE flattening to match Python!
        x = x.transposed(axes: [0, 1, 3, 2])
        
        let shape = x.shape
        let B = shape[0]
        let T = shape[1]
        x = x.reshaped([B, T, -1])
        
        x = bigru(x) // [N, T, 512]
        print("DEBUG: RMVPE BiGRU output stats: min \(x.min().item(Float.self)), max \(x.max().item(Float.self)), mean \(x.mean().item(Float.self))")
        
        x = linear(x) // [N, T, 360]
        print("DEBUG: RMVPE Linear output stats: min \(x.min().item(Float.self)), max \(x.max().item(Float.self)), mean \(x.mean().item(Float.self))")
        
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
        // Python: cents_pred[maxx <= thred] = 0
        // This zeros out cents_pred for unvoiced segments (low salience)
        let maxx = MLX.max(salience, axis: 1)  // [T]
        
        // DEBUG: Check maxx values
        let maxxMin = maxx.min().item(Float.self)
        let maxxMax = maxx.max().item(Float.self)
        let maxxMean = maxx.mean().item(Float.self)
        print("DEBUG RMVPE: maxx stats: min=\(maxxMin), max=\(maxxMax), mean=\(maxxMean), thred=\(thred)")
        
        let voicedMask = maxx .> thred  // True where voiced (high salience)
        let numVoiced = MLX.sum(voicedMask.asType(Float.self)).item(Float.self)
        print("DEBUG RMVPE: voiced frames: \(numVoiced) / \(T) (\(100*numVoiced/Float(T))%)")
        
        centsPred = centsPred * voicedMask.asType(centsPred.dtype)

        // Convert cents to F0 using CORRECT formula
        // Python line 401: f0 = 10 * (2 ** (cents_pred / 1200))
        var f0 = 10.0 * MLX.pow(2.0, centsPred / 1200.0)

        // Zero out unvoiced frames (where f0 would be ~10 Hz)
        // Python line 402: f0[f0 == 10] = 0
        // Use voicedMask directly since that's more reliable than floating point comparison
        f0 = f0 * voicedMask.asType(f0.dtype)

        return f0.expandedDimensions(axis: -1)  // [T, 1]
    }
    
    // Helper to run full inference
    func infer(audio: MLXArray, thred: Float = 0.03) -> MLXArray {
        // audio: [T]
        let melProcessor = MelSpectrogram()
        melProcessor.debug_mel_filterbank()
        let mel = melProcessor(audio) // [n_mels, T_frames] (Log Mel) (128, T)
        print("DEBUG: Mel Spectrogram Stats: min \(mel.min().item(Float.self)), max \(mel.max().item(Float.self)), shape \(mel.shape)")
        
        // Model expects [1, T, n_mels, 1]
        // mel: [128, T]
        let nFrames = mel.shape[1]
        let padCurr = 32 * ((nFrames - 1) / 32 + 1) - nFrames
        
        var melPadded = mel
        if padCurr > 0 {
            // Reflect padding on the time axis (axis 1)
            // Python: mel_padded = np.pad(mel, ((0, 0), (0, pad_curr)), mode='reflect')
            // Using our manual reflection logic for axis 1
            let n = nFrames
            let indices = Array(0..<padCurr).map { n - 2 - ($0 % (n - 1)) }
            let padIdx = MLXArray(indices.map { Int32($0) })
            let reflection = mel[0..., padIdx]
            melPadded = concatenated([mel, reflection], axis: 1)
        }
        
        // [128, T_padded] -> [T_padded, 128] -> [1, T_padded, 128, 1]
        let melInput = melPadded.transposed().expandedDimensions(axis: 0).expandedDimensions(axis: -1)
        print("DEBUG: RMVPE melInput stats: min \(melInput.min().item(Float.self)), max \(melInput.max().item(Float.self)), shape \(melInput.shape)")
        
        // Run model
        let hidden = self(melInput) // [1, T_padded, 360]
        print("DEBUG: RMVPE hidden stats: min \(hidden.min().item(Float.self)), max \(hidden.max().item(Float.self)), mean \(hidden.mean().item(Float.self))")
        
        // Take only original n_frames
        let hiddenTrimmed = hidden[0..., 0..<nFrames, 0...]
        
        // Decode
        let f0 = self.decode(hiddenTrimmed, thred: thred) // [T, 1]
        
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
            
            // Slaney normalization: 2.0 / (frequencies[i+2] - frequencies[i])
            let norm = 2.0 / (right - left)
            
            for j in 0..<n_freqs {
                let f = freq_bins[j]
                var val: Float = 0
                
                if f >= left && f <= center {
                    val = (f - left) / (center - left + 1e-10)
                } else if f >= center && f <= right {
                    val = (right - f) / (right - center + 1e-10)
                }
                
                // Apply normalization
                filterbank_data[i * n_freqs + j] = val * norm
            }
        }
        
        return MLXArray(filterbank_data, [n_mels, n_freqs])
    }
    
    func debug_mel_filterbank() {
        if mel_filterbank == nil { mel_filterbank = create_mel_filterbank() }
        let fb = mel_filterbank!
        print("DEBUG: Mel Filterbank Stats - shape: \(fb.shape), min: \(fb.min().item(Float.self)), max: \(fb.max().item(Float.self)), mean: \(fb.mean().item(Float.self))")
        // Log first 10 values of first filter
        let slice = fb[0, 0..<10].asType(Float.self)
        MLX.eval(slice)
        print("DEBUG: Mel Filterbank [0, :10]: \(slice.asArray(Float.self))")
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
        print("DEBUG: MelSpectrogram input audio: min \(audio.min().item(Float.self)), max \(audio.max().item(Float.self))")
        // audio: [T]
        if mel_filterbank == nil { mel_filterbank = create_mel_filterbank() }
        if window == nil { window = create_window() }
        
        let pad_len = n_fft / 2
        let n = audio.size
        
        // Manual reflection padding using MLX native slicing
        // Left pad: reverse of the start
        let leftIdx = MLXArray(stride(from: pad_len, to: 0, by: -1))
        let leftPad = audio[leftIdx]
        
        // Right pad: reverse of the end
        let rightIdx = MLXArray(stride(from: n - 2, to: n - pad_len - 2, by: -1))
        let rightPad = audio[rightIdx]
        
        let audio_padded = concatenated([leftPad, audio, rightPad], axis: 0)
        
        print("DEBUG: RMVPE Audio Padded [0...20]: \(audio_padded[0..<20].asArray(Float.self))")
        print("DEBUG: RMVPE Audio Padded Center [512-10...512+10]: \(audio_padded[502..<522].asArray(Float.self))")
        
        let len_p = audio_padded.shape[0]
        let num_frames = 1 + (len_p - n_fft) / hop_length
        
        let frame_starts = MLXArray(Array(stride(from: 0, to: num_frames * hop_length, by: hop_length))).expandedDimensions(axis: 1)
        let win_indices = MLXArray(Array(0..<n_fft)).expandedDimensions(axis: 0)
        
        let indices = frame_starts + win_indices // [num_frames, n_fft]
        
        let frames = audio_padded[indices] // [N_frames, n_fft]
        
        let win = window!
        let frames_w = frames * win
        print("DEBUG: RMVPE Window stats: min \(win.min().item(Float.self)), max \(win.max().item(Float.self)), sum \(win.sum().item(Float.self))")
        print("DEBUG: RMVPE frames_w stats: min \(frames_w.min().item(Float.self)), max \(frames_w.max().item(Float.self)), mean \(frames_w.mean().item(Float.self))")
        
        // FFT
        // Using MLXFFT.rfft
        let spectrum = MLXFFT.rfft(frames_w, axis: -1)
        
        // Magnitude
        let magnitude = abs(spectrum)
        print("DEBUG: RMVPE Magnitude stats: min \(magnitude.min().item(Float.self)), max \(magnitude.max().item(Float.self)), mean \(magnitude.mean().item(Float.self))")
        
        let mel = matmul(mel_filterbank!, magnitude.transposed())
        print("DEBUG: RMVPE Mel (pre-log) stats: min \(mel.min().item(Float.self)), max \(mel.max().item(Float.self)), mean \(mel.mean().item(Float.self))")
        
        let log_mel = MLX.log(maximum(mel, 1e-5))
        
        let melMin = log_mel.min().item(Float.self)
        let melMax = log_mel.max().item(Float.self)
        let melMean = log_mel.mean().item(Float.self)
        print("DEBUG: Mel Spectrogram Stats: min \(melMin), max \(melMax), mean \(melMean), shape \(log_mel.shape)")
        
        // Log first frame (column 0), first 10 bins (rows 0-9)
        // log_mel shape is [128, T]
        let slice = log_mel[0..<10, 0].asType(Float.self)
        MLX.eval(slice)
        let sliceArr = slice.asArray(Float.self)
        print("DEBUG: Mel[0, :10]: \(sliceArr)")
        
        return log_mel
    }
}
