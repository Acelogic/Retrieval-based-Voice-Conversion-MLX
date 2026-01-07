import Foundation
import MLX
import MLXNN
import MLXRandom

/// RVC Synthesizer Implementation for iOS
///
/// This is a Swift/MLX port of the Python MLX implementation in:
/// - rvc_mlx/lib/mlx/encoders.py (TextEncoder, Encoder)
/// - rvc_mlx/lib/mlx/generators.py (Generator)
/// - rvc_mlx/lib/mlx/attentions.py (MultiHeadAttention, FFN)
///
/// CRITICAL FIX APPLIED:
/// - TextEncoder output format: Fixed dimension mismatch (B,C,T vs B,T,C)
///   - Now correctly transposes stats before splitting
///   - Returns (m, logs) in (B, C, T) format matching Python
///   - Returns xMask in (B, 1, T) format matching Python
///   - This was the "Known issue" in commit df081a66
///
/// Architecture matches Python exactly:
/// - TextEncoder: LeakyReLU(0.1), not GELU
/// - FFN: ReLU activation (default when activation=None in Python)
/// - Generator: LeakyReLU with LRELU_SLOPE
///
/// Reference: rvc_mlx/lib/mlx/encoders.py, generators.py

// MARK: - Utility Functions

func sequenceMask(lengths: MLXArray, maxLength: Int) -> MLXArray {
    // Creates a mask of shape (B, maxLength) where mask[b, i] = 1 if i < lengths[b]
    let x = MLXArray(0..<maxLength).asType(lengths.dtype)
    return (x.expandedDimensions(axis: 0) .< lengths.expandedDimensions(axis: 1)).asType(Float32.self)
}

// MARK: - Multi-Head Attention

class MultiHeadAttention: Module {
    let channels: Int
    let outChannels: Int
    let nHeads: Int
    let kChannels: Int
    let windowSize: Int?
    
    let conv_q: MLXNN.Conv1d
    let conv_k: MLXNN.Conv1d
    let conv_v: MLXNN.Conv1d
    let conv_o: MLXNN.Conv1d
    let drop: MLXNN.Dropout
    
    init(channels: Int, outChannels: Int, nHeads: Int, pDropout: Float = 0.0, windowSize: Int? = nil) {
        self.channels = channels
        self.outChannels = outChannels
        self.nHeads = nHeads
        self.kChannels = channels / nHeads
        self.windowSize = windowSize
        
        self.conv_q = MLXNN.Conv1d(inputChannels: channels, outputChannels: channels, kernelSize: 1)
        self.conv_k = MLXNN.Conv1d(inputChannels: channels, outputChannels: channels, kernelSize: 1)
        self.conv_v = MLXNN.Conv1d(inputChannels: channels, outputChannels: channels, kernelSize: 1)
        self.conv_o = MLXNN.Conv1d(inputChannels: channels, outputChannels: outChannels, kernelSize: 1)
        self.drop = MLXNN.Dropout(p: pDropout)
        super.init()
    }
    
    func callAsFunction(_ x: MLXArray, c: MLXArray, attnMask: MLXArray? = nil) -> MLXArray {
        let q = conv_q(x)
        let k = conv_k(c)
        let v = conv_v(c)
        let (out, _) = attention(query: q, key: k, value: v, mask: attnMask)
        return conv_o(out)
    }
    
    func attention(query: MLXArray, key: MLXArray, value: MLXArray, mask: MLXArray?) -> (MLXArray, MLXArray) {
        let b = key.shape[0]
        let t_s = key.shape[1]
        let t_t = query.shape[1]
        
        // Reshape for multi-head: (B, T, C) -> (B, T, Heads, HeadDim) -> (B, Heads, T, HeadDim)
        var q = query.reshaped([b, t_t, nHeads, kChannels]).transposed(0, 2, 1, 3)
        var k = key.reshaped([b, t_s, nHeads, kChannels]).transposed(0, 2, 1, 3)
        var v = value.reshaped([b, t_s, nHeads, kChannels]).transposed(0, 2, 1, 3)
        
        // Scaled dot-product attention
        var scores = MLX.matmul(q / sqrt(Float(kChannels)), k.transposed(0, 1, 3, 2))
        
        if let mask = mask {
            scores = MLX.where(mask .== 0, MLXArray(-1e4), scores)
        }
        
        let pAttn = drop(softmax(scores, axis: -1))
        var output = MLX.matmul(pAttn, v)
        
        // Reshape back to (B, T_t, C)
        output = output.transposed(0, 2, 1, 3).reshaped([b, t_t, -1])
        return (output, pAttn)
    }
}

// MARK: - Feed Forward Network

class FFN: Module {
    let conv_1: MLXNN.Conv1d
    let conv_2: MLXNN.Conv1d
    let drop: MLXNN.Dropout
    let kernel_size: Int
    
    init(inChannels: Int, outChannels: Int, filterChannels: Int, kernelSize: Int, pDropout: Float = 0.0) {
        self.conv_1 = MLXNN.Conv1d(inputChannels: inChannels, outputChannels: filterChannels, kernelSize: kernelSize, padding: 0)
        self.conv_2 = MLXNN.Conv1d(inputChannels: filterChannels, outputChannels: outChannels, kernelSize: kernelSize, padding: 0)
        self.drop = MLXNN.Dropout(p: pDropout)
        self.kernel_size = kernelSize
        super.init()
    }
    
    func callAsFunction(_ x: MLXArray, xMask: MLXArray) -> MLXArray {
        let padTotal = kernel_size - 1
        let padL = padTotal / 2
        let padR = padTotal - padL
        
        var h = x * xMask
        h = MLX.padded(h, widths: [IntOrPair((0, 0)), IntOrPair((padL, padR)), IntOrPair((0, 0))])
        h = conv_1(h)
        h = MLX.maximum(h, MLXArray(0))  // ReLU
        h = drop(h)
        
        h = h * xMask
        h = MLX.padded(h, widths: [IntOrPair((0, 0)), IntOrPair((padL, padR)), IntOrPair((0, 0))])
        h = conv_2(h)
        return h * xMask
    }
}

// MARK: - Encoder (Transformer-style)

class RVCEncoder: Module {
    let nLayers: Int
    let drop: MLXNN.Dropout
    var attn_layers: [MultiHeadAttention] = []
    var norm_layers_1: [MLXNN.LayerNorm] = []
    var ffn_layers: [FFN] = []
    var norm_layers_2: [MLXNN.LayerNorm] = []
    
    init(hiddenChannels: Int, filterChannels: Int, nHeads: Int, nLayers: Int, kernelSize: Int = 1, pDropout: Float = 0.0, windowSize: Int = 10) {
        self.nLayers = nLayers
        self.drop = MLXNN.Dropout(p: pDropout)
        
        self.attn_layers = (0..<nLayers).map { _ in
            MultiHeadAttention(channels: hiddenChannels, outChannels: hiddenChannels, nHeads: nHeads, pDropout: pDropout, windowSize: windowSize)
        }
        self.norm_layers_1 = (0..<nLayers).map { _ in
            MLXNN.LayerNorm(dimensions: hiddenChannels)
        }
        self.ffn_layers = (0..<nLayers).map { _ in
            FFN(inChannels: hiddenChannels, outChannels: hiddenChannels, filterChannels: filterChannels, kernelSize: kernelSize, pDropout: pDropout)
        }
        self.norm_layers_2 = (0..<nLayers).map { _ in
            MLXNN.LayerNorm(dimensions: hiddenChannels)
        }
        
        super.init()
    }
    
    func callAsFunction(_ x: MLXArray, xMask: MLXArray) -> MLXArray {
        // x: (B, L, C), xMask: (B, L, 1)
        let xMaskB = xMask.asType(Float32.self)
        
        // Create attention mask: (B, L, 1) * (B, 1, L) -> (B, L, L) -> (B, 1, L, L)
        let attnMask = (xMaskB * xMaskB.transposed(0, 2, 1)).expandedDimensions(axis: 1)
        
        var h = x * xMask
        
        for i in 0..<nLayers {
            let y = attn_layers[i](h, c: h, attnMask: attnMask)
            h = norm_layers_1[i](h + drop(y))
            
            let y2 = ffn_layers[i](h, xMask: xMask)
            h = norm_layers_2[i](h + drop(y2))
        }
        
        return h * xMask
    }
}

// MARK: - TextEncoder (enc_p)

class TextEncoder: Module {
    let hiddenChannels: Int
    let outChannels: Int
    let emb_phone: MLXNN.Linear
    let emb_pitch: MLXNN.Embedding?
    let encoder: RVCEncoder
    let proj: MLXNN.Conv1d
    
    init(outChannels: Int, hiddenChannels: Int, filterChannels: Int, nHeads: Int, nLayers: Int, kernelSize: Int, pDropout: Float, embeddingDim: Int, f0: Bool = true) {
        self.hiddenChannels = hiddenChannels
        self.outChannels = outChannels
        
        self.emb_phone = MLXNN.Linear(embeddingDim, hiddenChannels)
        self.emb_pitch = f0 ? MLXNN.Embedding(embeddingCount: 256, dimensions: hiddenChannels) : nil
        self.encoder = RVCEncoder(hiddenChannels: hiddenChannels, filterChannels: filterChannels, nHeads: nHeads, nLayers: nLayers, kernelSize: kernelSize, pDropout: pDropout)
        self.proj = MLXNN.Conv1d(inputChannels: hiddenChannels, outputChannels: outChannels * 2, kernelSize: 1)
        
        super.init()
    }
    
    func callAsFunction(_ phone: MLXArray, pitch: MLXArray?, lengths: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        // phone: (B, L, EmbDim), pitch: (B, L), lengths: (B,)
        var x = emb_phone(phone)
        print("DEBUG: TextEncoder emb_phone: min \(x.min().item(Float.self)), max \(x.max().item(Float.self))")
        
        if let pitch = pitch, let embPitch = emb_pitch {
            let pEmb = embPitch(pitch)
            print("DEBUG: TextEncoder emb_pitch: min \(pEmb.min().item(Float.self)), max \(pEmb.max().item(Float.self))")
            x = x + pEmb
        }
        
        x = x * sqrt(Float(hiddenChannels))
        x = leakyRelu(x, negativeSlope: 0.1)
        
        // Create mask
        let xMask = sequenceMask(lengths: lengths, maxLength: x.shape[1])
        let xMaskExpanded = xMask.expandedDimensions(axis: -1)  // (B, L, 1)
        
        x = encoder(x, xMask: xMaskExpanded)

        let stats = proj(x) * xMaskExpanded

        // CRITICAL: Transpose to match Python PyTorch format (B, C, T) before splitting
        // Python line 141: stats = stats.transpose(0, 2, 1)  # (B, T, C*2) -> (B, C*2, T)
        // This was the "dimension format mismatch" bug!
        let statsTransposed = stats.transposed(0, 2, 1)  // (B, L, C*2) -> (B, C*2, L)

        // Split on channel dimension (axis=1) to match Python
        // Python line 144: m, logs = mx.split(stats, 2, axis=1)
        let splitIdx = outChannels
        let m = statsTransposed[0..., 0..<splitIdx, 0...]  // (B, C, L)
        let logs = statsTransposed[0..., splitIdx..., 0...]  // (B, C, L)

        // Return xMask in (B, 1, L) format to match Python
        // Python line 148: x_mask_out = x_mask[:, None, :]
        let xMaskOut = xMask.expandedDimensions(axis: 1)  // (B, L) -> (B, 1, L)

        return (m, logs, xMaskOut)
    }
}

// MARK: - WaveNet for Flow (Matches Python MLX exactly)

class WaveNet: Module {
    let hiddenChannels: Int
    let nLayers: Int
    let cond_layer: MLXNN.Conv1d?

    // Individual layers registered as in_layer_0, in_layer_1, etc. to match Python weight keys
    let in_layer_0: MLXNN.Conv1d
    let in_layer_1: MLXNN.Conv1d
    let in_layer_2: MLXNN.Conv1d

    // res_skip_layer_2 (last layer) outputs hidden_channels, others output 2*hidden_channels
    let res_skip_layer_0: MLXNN.Conv1d
    let res_skip_layer_1: MLXNN.Conv1d
    let res_skip_layer_2: MLXNN.Conv1d

    init(hiddenChannels: Int, kernelSize: Int, dilationRate: Int, nLayers: Int, ginChannels: Int) {
        assert(nLayers == 3, "WaveNet hardcoded for 3 layers to match Python architecture")
        self.hiddenChannels = hiddenChannels
        self.nLayers = nLayers

        // Single cond_layer at WaveNet level: outputs 2 * hidden * n_layers channels
        self.cond_layer = ginChannels != 0 ? MLXNN.Conv1d(inputChannels: ginChannels, outputChannels: 2 * hiddenChannels * nLayers, kernelSize: 1) : nil

        // Create in_layers with proper dilation and padding
        let dilations = (0..<nLayers).map { Int(pow(Double(dilationRate), Double($0))) }
        let paddings = dilations.map { (kernelSize * $0 - $0) / 2 }

        self.in_layer_0 = MLXNN.Conv1d(inputChannels: hiddenChannels, outputChannels: 2 * hiddenChannels, kernelSize: kernelSize, padding: paddings[0], dilation: dilations[0])
        self.in_layer_1 = MLXNN.Conv1d(inputChannels: hiddenChannels, outputChannels: 2 * hiddenChannels, kernelSize: kernelSize, padding: paddings[1], dilation: dilations[1])
        self.in_layer_2 = MLXNN.Conv1d(inputChannels: hiddenChannels, outputChannels: 2 * hiddenChannels, kernelSize: kernelSize, padding: paddings[2], dilation: dilations[2])

        // res_skip_layers: last layer outputs hidden_channels, others output 2*hidden_channels
        self.res_skip_layer_0 = MLXNN.Conv1d(inputChannels: hiddenChannels, outputChannels: 2 * hiddenChannels, kernelSize: 1)
        self.res_skip_layer_1 = MLXNN.Conv1d(inputChannels: hiddenChannels, outputChannels: 2 * hiddenChannels, kernelSize: 1)
        self.res_skip_layer_2 = MLXNN.Conv1d(inputChannels: hiddenChannels, outputChannels: hiddenChannels, kernelSize: 1)  // Last layer: hidden only

        super.init()
    }

    func callAsFunction(_ x: MLXArray, xMask: MLXArray, g: MLXArray?) -> MLXArray {
        var h = x
        var outputAcc = MLX.zeros(like: x)

        // Apply cond_layer once to g, then slice per layer
        var gCond: MLXArray? = nil
        if let g = g, let condLayer = cond_layer {
            gCond = condLayer(g)  // Shape: (B, T, 2*hidden*n_layers)
        }

        // Get in_layers and res_skip_layers as arrays for iteration
        let inLayers = [in_layer_0, in_layer_1, in_layer_2]
        let resSkipLayers = [res_skip_layer_0, res_skip_layer_1, res_skip_layer_2]

        for i in 0..<nLayers {
            let xIn = inLayers[i](h)

            // Slice conditioning for this layer
            var acts: MLXArray
            if let gCond = gCond {
                let startCh = i * 2 * hiddenChannels
                let endCh = (i + 1) * 2 * hiddenChannels
                let gSlice = gCond[0..., 0..., startCh..<endCh]  // (B, T, 2*hidden)

                // Gated activation: tanh(xIn + gSlice half) * sigmoid(xIn + gSlice half)
                let combined = xIn + gSlice
                let tAct = tanh(combined[0..., 0..., 0..<hiddenChannels])
                let sAct = sigmoid(combined[0..., 0..., hiddenChannels...])
                acts = tAct * sAct
            } else {
                // No conditioning
                let tAct = tanh(xIn[0..., 0..., 0..<hiddenChannels])
                let sAct = sigmoid(xIn[0..., 0..., hiddenChannels...])
                acts = tAct * sAct
            }

            let resSkipActs = resSkipLayers[i](acts)

            if i < nLayers - 1 {
                // Non-last layer: split into res and skip
                let resActs = resSkipActs[0..., 0..., 0..<hiddenChannels]
                h = (h + resActs) * xMask
                outputAcc = outputAcc + resSkipActs[0..., 0..., hiddenChannels...]
            } else {
                // Last layer: entire output is skip (no res split)
                outputAcc = outputAcc + resSkipActs
            }
        }

        return outputAcc * xMask
    }
}

// MARK: - Residual Coupling Layer

class ResidualCouplingLayer: Module {
    let halfChannels: Int
    let meanOnly: Bool
    let pre: MLXNN.Conv1d
    let enc: WaveNet
    let post: MLXNN.Conv1d
    
    init(channels: Int, hiddenChannels: Int, kernelSize: Int, dilationRate: Int, nLayers: Int, ginChannels: Int, meanOnly: Bool = false) {
        self.halfChannels = channels / 2
        self.meanOnly = meanOnly
        
        self.pre = MLXNN.Conv1d(inputChannels: halfChannels, outputChannels: hiddenChannels, kernelSize: 1)
        self.enc = WaveNet(hiddenChannels: hiddenChannels, kernelSize: kernelSize, dilationRate: dilationRate, nLayers: nLayers, ginChannels: ginChannels)
        let postOutChannels = meanOnly ? halfChannels : halfChannels * 2
        self.post = MLXNN.Conv1d(inputChannels: hiddenChannels, outputChannels: postOutChannels, kernelSize: 1)
        
        super.init()
    }
    
    func callAsFunction(_ x: MLXArray, xMask: MLXArray, g: MLXArray?, reverse: Bool = false) -> MLXArray {
        // x: (B, L, C)
        let x0 = x[0..., 0..., 0..<halfChannels]
        var x1 = x[0..., 0..., halfChannels...]
        
        var h = pre(x0) * xMask
        h = enc(h, xMask: xMask, g: g)
        let stats = post(h) * xMask
        
        let m: MLXArray
        let logs: MLXArray
        
        if meanOnly {
            m = stats
            logs = MLX.zeros(like: m)
        } else {
            m = stats[0..., 0..., 0..<halfChannels]
            logs = stats[0..., 0..., halfChannels...]
        }
        
        if !reverse {
            x1 = (m + x1 * exp(logs)) * xMask
        } else {
            x1 = (x1 - m) * exp(-logs) * xMask
        }
        
        return MLX.concatenated([x0, x1], axis: -1)
    }
}

// MARK: - Residual Coupling Block (flow)
// Uses named properties flow_0, flow_1, etc. to match weight file keys

class ResidualCouplingBlock: Module {
    let nFlows: Int

    // Named properties to match weight keys: flow.flow_0.enc..., flow.flow_1.enc..., etc.
    let flow_0: ResidualCouplingLayer
    let flow_1: ResidualCouplingLayer
    let flow_2: ResidualCouplingLayer
    let flow_3: ResidualCouplingLayer

    init(channels: Int, hiddenChannels: Int, kernelSize: Int, dilationRate: Int, nLayers: Int, nFlows: Int = 4, ginChannels: Int) {
        assert(nFlows == 4, "ResidualCouplingBlock hardcoded for 4 flows to match Python architecture")
        self.nFlows = nFlows

        self.flow_0 = ResidualCouplingLayer(channels: channels, hiddenChannels: hiddenChannels, kernelSize: kernelSize, dilationRate: dilationRate, nLayers: nLayers, ginChannels: ginChannels, meanOnly: true)
        self.flow_1 = ResidualCouplingLayer(channels: channels, hiddenChannels: hiddenChannels, kernelSize: kernelSize, dilationRate: dilationRate, nLayers: nLayers, ginChannels: ginChannels, meanOnly: true)
        self.flow_2 = ResidualCouplingLayer(channels: channels, hiddenChannels: hiddenChannels, kernelSize: kernelSize, dilationRate: dilationRate, nLayers: nLayers, ginChannels: ginChannels, meanOnly: true)
        self.flow_3 = ResidualCouplingLayer(channels: channels, hiddenChannels: hiddenChannels, kernelSize: kernelSize, dilationRate: dilationRate, nLayers: nLayers, ginChannels: ginChannels, meanOnly: true)

        super.init()
    }

    func callAsFunction(_ x: MLXArray, xMask: MLXArray, g: MLXArray?, reverse: Bool = false) -> MLXArray {
        var h = x
        let flows = [flow_0, flow_1, flow_2, flow_3]

        if !reverse {
            // Forward: flow then flip
            for i in 0..<nFlows {
                h = flows[i](h, xMask: xMask, g: g, reverse: false)
                h = h[0..., 0..., .stride(by: -1)]  // Flip channels
            }
        } else {
            // Reverse: flip then flow (CRITICAL: different order from forward!)
            for i in (0..<nFlows).reversed() {
                h = h[0..., 0..., .stride(by: -1)]  // Flip first!
                h = flows[i](h, xMask: xMask, g: g, reverse: true)
            }
        }

        return h
    }
}

// MARK: - Full Synthesizer

public class Synthesizer: Module {
    let enc_p: TextEncoder
    let dec: Generator  // Use existing Generator from RVCModel.swift
    let flow: ResidualCouplingBlock
    let emb_g: MLXNN.Embedding
    let useF0: Bool
    
    public init(
        interChannels: Int = 192,
        hiddenChannels: Int = 192,
        filterChannels: Int = 768,
        nHeads: Int = 2,
        nLayers: Int = 6,
        kernelSize: Int = 3,
        pDropout: Float = 0.0,
        embeddingDim: Int = 768,
        speakerEmbedDim: Int = 256,
        ginChannels: Int = 256,
        useF0: Bool = true,
        upsampleRates: [Int] = [10, 10, 2, 2],
        upsampleKernelSizes: [Int] = [16, 16, 4, 4],
        sampleRate: Int = 40000
    ) {
        self.useF0 = useF0

        // TextEncoder: transforms 768-dim HuBERT -> 192-dim latent
        self.enc_p = TextEncoder(
            outChannels: interChannels,
            hiddenChannels: hiddenChannels,
            filterChannels: filterChannels,
            nHeads: nHeads,
            nLayers: nLayers,
            kernelSize: kernelSize,
            pDropout: pDropout,
            embeddingDim: embeddingDim,
            f0: useF0
        )

        // Generator: now supports dynamic configuration
        self.dec = Generator(inputChannels: interChannels, ginChannels: ginChannels, upsampleRates: upsampleRates, upsampleKernelSizes: upsampleKernelSizes, sampleRate: sampleRate)
        
        // Flow: reverse flow for voice conversion
        self.flow = ResidualCouplingBlock(
            channels: interChannels,
            hiddenChannels: hiddenChannels,
            kernelSize: 5,
            dilationRate: 1,
            nLayers: 3,
            nFlows: 4,
            ginChannels: ginChannels
        )
        
        // Speaker embedding
        self.emb_g = MLXNN.Embedding(embeddingCount: speakerEmbedDim, dimensions: ginChannels)
        
        super.init()
    }
    
    public func infer(phone: MLXArray, phoneLengths: MLXArray, pitch: MLXArray?, nsff0: MLXArray?, sid: MLXArray) -> MLXArray {
        // phone: (B, L, 768) - HuBERT features
        // pitch: (B, L) - coarse pitch (for embedding lookup)
        // nsff0: (B, L) - continuous F0 in Hz
        // sid: (B,) - speaker ID
        
        // Get speaker embedding: (B, 1, C)
        let g = emb_g(sid).expandedDimensions(axis: 1)
        
        // Encode features
        let (m_p, logs_p, xMask) = enc_p(phone, pitch: pitch, lengths: phoneLengths)
        print("DEBUG: TextEncoder output - m_p: \(m_p.shape) [\(m_p.min().item(Float.self))...\(m_p.max().item(Float.self))], logs_p: \(logs_p.shape) [\(logs_p.min().item(Float.self))...\(logs_p.max().item(Float.self))], xMask: \(xMask.shape)")

        // Sample from encoded distribution
        // CRITICAL: xMask is already (B, 1, T) from TextEncoder - don't expand it!
        // Broadcasts: (B, C, T) * (B, 1, T) = (B, C, T)
        let z_p = (m_p + exp(logs_p) * MLXRandom.normal(m_p.shape).asType(m_p.dtype) * 0.0) * xMask
        print("DEBUG: z_p shape: \(z_p.shape), stats: min \(z_p.min().item(Float.self)), max \(z_p.max().item(Float.self))")

        // Flow reverse pass
        // Convert to (B, T, C) format for Flow (Python line 117-118)
        let z_p_transposed = z_p.transposed(0, 2, 1)  // (B, C, T) -> (B, T, C)
        let xMask_transposed = xMask.transposed(0, 2, 1)  // (B, 1, T) -> (B, T, 1)

        let z_mlx = flow(z_p_transposed, xMask: xMask_transposed, g: g, reverse: true)  // Output: (B, T, C)
        print("DEBUG: z_mlx (after flow) shape: \(z_mlx.shape), stats: min \(z_mlx.min().item(Float.self)), max \(z_mlx.max().item(Float.self))")

        // Convert back to (B, C, T) for decoder (Python line 123)
        let z = z_mlx.transposed(0, 2, 1)  // (B, T, C) -> (B, C, T)

        // Decode to audio
        // Decoder input: z * x_mask where both are (B, C, T)
        // Python line 128: o = self.dec(z * x_mask, nsff0, g=g)
        let output = dec(z * xMask, f0: nsff0 ?? MLX.zeros([phone.shape[0], phone.shape[1], 1]), g: g)
        print("DEBUG: Generator output stats: min \(output.min().item(Float.self)), max \(output.max().item(Float.self))")
        
        return output
    }
}
