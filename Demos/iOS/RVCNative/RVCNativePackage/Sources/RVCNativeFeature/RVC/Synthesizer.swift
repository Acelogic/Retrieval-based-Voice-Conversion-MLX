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

// MARK: - WaveNet for Flow

class WaveNetLayer: Module {
    let in_layer: MLXNN.Conv1d
    let res_skip_layer: MLXNN.Conv1d
    let cond_layer: MLXNN.Conv1d?
    let halfChannels: Int
    
    init(hiddenChannels: Int, kernelSize: Int, dilation: Int, ginChannels: Int) {
        let padding = (kernelSize * dilation - dilation) / 2
        self.in_layer = MLXNN.Conv1d(inputChannels: hiddenChannels, outputChannels: hiddenChannels * 2, kernelSize: kernelSize, padding: padding, dilation: dilation)
        self.res_skip_layer = MLXNN.Conv1d(inputChannels: hiddenChannels, outputChannels: hiddenChannels * 2, kernelSize: 1)
        self.cond_layer = ginChannels != 0 ? MLXNN.Conv1d(inputChannels: ginChannels, outputChannels: hiddenChannels * 2, kernelSize: 1) : nil
        self.halfChannels = hiddenChannels
        super.init()
    }
    
    func callAsFunction(_ x: MLXArray, xMask: MLXArray, g: MLXArray?) -> (MLXArray, MLXArray) {
        var h = in_layer(x)
        
        if let g = g, let condLayer = cond_layer {
            h = h + condLayer(g)
        }
        
        // Gated activation
        let tAct = tanh(h[0..., 0..., 0..<halfChannels])
        let sAct = sigmoid(h[0..., 0..., halfChannels...])
        let acts = tAct * sAct
        
        let resSkip = res_skip_layer(acts)
        let res = resSkip[0..., 0..., 0..<halfChannels]
        let skip = resSkip[0..., 0..., halfChannels...]
        
        return ((x + res) * xMask, skip * xMask)
    }
}

class WaveNet: Module {
    var layers: [WaveNetLayer] = []
    let nLayers: Int
    
    init(hiddenChannels: Int, kernelSize: Int, dilationRate: Int, nLayers: Int, ginChannels: Int) {
        self.nLayers = nLayers
        
        self.layers = (0..<nLayers).map { i in
            let dilation = Int(pow(Double(dilationRate), Double(i)))
            return WaveNetLayer(hiddenChannels: hiddenChannels, kernelSize: kernelSize, dilation: dilation, ginChannels: ginChannels)
        }
        
        super.init()
    }
    
    func callAsFunction(_ x: MLXArray, xMask: MLXArray, g: MLXArray?) -> MLXArray {
        var h = x
        var skipSum: MLXArray? = nil
        
        for i in 0..<nLayers {
            let (res, skip) = layers[i](h, xMask: xMask, g: g)
            h = res
            if skipSum == nil {
                skipSum = skip
            } else {
                skipSum = skipSum! + skip
            }
        }
        
        return skipSum! * xMask
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

class ResidualCouplingBlock: Module {
    var flows: [ResidualCouplingLayer] = []
    let nFlows: Int
    
    init(channels: Int, hiddenChannels: Int, kernelSize: Int, dilationRate: Int, nLayers: Int, nFlows: Int = 4, ginChannels: Int) {
        self.nFlows = nFlows
        
        self.flows = (0..<nFlows).map { _ in
            ResidualCouplingLayer(channels: channels, hiddenChannels: hiddenChannels, kernelSize: kernelSize, dilationRate: dilationRate, nLayers: nLayers, ginChannels: ginChannels, meanOnly: true)
        }
        
        super.init()
    }
    
    func callAsFunction(_ x: MLXArray, xMask: MLXArray, g: MLXArray?, reverse: Bool = false) -> MLXArray {
        var h = x
        
        let iterator: [Int] = reverse ? Array((0..<nFlows).reversed()) : Array(0..<nFlows)
        
        for i in iterator {
            h = flows[i](h, xMask: xMask, g: g, reverse: reverse)
            // Flip channels (axis 2)
            h = h[0..., 0..., .stride(by: -1)]
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
