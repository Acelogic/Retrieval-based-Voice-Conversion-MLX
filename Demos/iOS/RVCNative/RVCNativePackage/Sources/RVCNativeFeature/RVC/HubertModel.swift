import Foundation
import MLX
import MLXNN
import MLXRandom

/// HuBERT Model Implementation for RVC
///
/// This is a Swift/MLX port of the Python MLX implementation in rvc_mlx/lib/mlx/hubert.py
///
/// CRITICAL FIXES APPLIED:
/// 1. layerNormEps: 1e-5 (matches Python, was incorrectly 1e-12)
/// 2. HubertPositionalConvEmbedding: Added GELU activation after conv (line 338 in Python)
/// 3. GELU activation: Using precise GELU (approx='none' in Python)
///
/// Architecture:
/// - Feature Extractor: 7 CNN layers (1 with GroupNorm, 6 without)
/// - Feature Projection: LayerNorm + Linear (512 -> 768)
/// - Positional Conv Embedding: Grouped Conv1d (groups=16, kernel=128)
/// - Transformer Encoder: 12 layers of MultiHeadAttention + FFN
/// - Final Projection: Linear (768 -> 256) for RVC
///
/// Reference: rvc_mlx/lib/mlx/hubert.py

public struct HubertConfig: Codable {
    public var vocabSize: Int = 32
    public var hiddenSize: Int = 768
    public var numHiddenLayers: Int = 12
    public var numAttentionHeads: Int = 12
    public var intermediateSize: Int = 3072
    public var hiddenAct: String = "gelu"
    public var hiddenDropoutProb: Float = 0.1
    public var attentionProbsDropoutProb: Float = 0.1
    public var maxPositionEmbeddings: Int = 512
    public var typeVocabSize: Int = 2
    public var initializerRange: Float = 0.02
    public var layerNormEps: Float = 1e-5  // CRITICAL: Must match Python (was 1e-12)
    public var padTokenId: Int = 1
    public var bosTokenId: Int = 0
    public var eosTokenId: Int = 2
    public var classifierProjSize: Int = 256

    public init() {}
}

class HubertAttention: Module {
    let num_attention_heads: Int
    let attention_head_size: Int
    let all_head_size: Int
    let scale: Float
    
    let q_proj: Linear
    let k_proj: Linear
    let v_proj: Linear
    let out_proj: Linear
    let dropout: Dropout
    
    init(config: HubertConfig) {
        self.num_attention_heads = config.numAttentionHeads
        self.attention_head_size = config.hiddenSize / config.numAttentionHeads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.scale = pow(Float(self.attention_head_size), -0.5)
        
        self.q_proj = Linear(config.hiddenSize, self.all_head_size)
        self.k_proj = Linear(config.hiddenSize, self.all_head_size)
        self.v_proj = Linear(config.hiddenSize, self.all_head_size)
        self.out_proj = Linear(config.hiddenSize, config.hiddenSize)
        self.dropout = Dropout(p: config.attentionProbsDropoutProb)
        
        super.init()
    }
    
    func callAsFunction(_ hiddenStates: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        let (B, L, _) = (hiddenStates.shape[0], hiddenStates.shape[1], hiddenStates.shape[2])
        
        var q = q_proj(hiddenStates)
        var k = k_proj(hiddenStates)
        var v = v_proj(hiddenStates)
        
        // Reshape: [B, L, H, D] -> [B, H, L, D]
        q = q.reshaped(B, L, num_attention_heads, attention_head_size).transposed(0, 2, 1, 3)
        k = k.reshaped(B, L, num_attention_heads, attention_head_size).transposed(0, 2, 1, 3)
        v = v.reshaped(B, L, num_attention_heads, attention_head_size).transposed(0, 2, 1, 3)
        
        // Scores: [B, H, L, D] @ [B, H, D, L] -> [B, H, L, L]
        var scores = matmul(q, k.transposed(0, 1, 3, 2)) * scale
        
        if let mask = mask {
            scores = scores + mask
        }
        
        var probs = softmax(scores, axis: -1)
        probs = dropout(probs)
        
        // Output: [B, H, L, L] @ [B, H, L, D] -> [B, H, L, D]
        var out = matmul(probs, v)
        
        // Reshape back: [B, H, L, D] -> [B, L, H, D] -> [B, L, C]
        out = out.transposed(0, 2, 1, 3).reshaped(B, L, all_head_size)
        
        return out_proj(out)
    }
}

class HubertFeedForward: Module {
    let intermediate_dense: Linear
    let output_dense: Linear
    let act: GELU

    init(config: HubertConfig) {
        self.intermediate_dense = Linear(config.hiddenSize, config.intermediateSize)
        self.output_dense = Linear(config.intermediateSize, config.hiddenSize)
        // CRITICAL: Must use precise GELU (not approximation)
        // Python: nn.GELU() defaults to approx='none'
        self.act = GELU()
        super.init()
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        var x = intermediate_dense(hiddenStates)
        x = act(x)
        x = output_dense(x)
        return x
    }
}

class HubertLayer: Module {
    let attention: HubertAttention
    let feed_forward: HubertFeedForward
    let layer_norm: LayerNorm
    let final_layer_norm: LayerNorm
    let dropout: Dropout
    
    init(config: HubertConfig) {
        self.attention = HubertAttention(config: config)
        self.feed_forward = HubertFeedForward(config: config)
        self.layer_norm = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        self.final_layer_norm = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        self.dropout = Dropout(p: config.hiddenDropoutProb)
        super.init()
    }
    
    func callAsFunction(_ hiddenStates: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        let attnOutput = attention(hiddenStates, mask: mask)
        let h1 = layer_norm(hiddenStates + dropout(attnOutput))
        
        let ffnOutput = feed_forward(h1)
        let h2 = final_layer_norm(h1 + dropout(ffnOutput))
        
        return h2
    }
}

class HubertPositionalConvEmbedding: Module {
    let conv: MLXNN.Conv1d
    let groups: Int = 16
    let padding: Int = 64

    init(config: HubertConfig) {
        let hiddenSize = config.hiddenSize
        let kernelSize = 128
        
        self.conv = MLXNN.Conv1d(
            inputChannels: hiddenSize,
            outputChannels: hiddenSize,
            kernelSize: kernelSize,
            stride: 1,
            padding: padding,
            groups: groups,
            bias: true
        )
        super.init()
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        // hiddenStates: [B, T, C]
        var out = conv(hiddenStates)

        // Crop: Remove last time step (kernel=128 is even, remove 1)
        let L = out.shape[1]
        out = out[0..., 0..<(L-1), 0...]

        // CRITICAL: Apply GELU activation
        out = gelu(out)

        // Residual connection
        return out + hiddenStates
    }
}

class HubertGroupNormConvLayer: Module {
    let conv: MLXNN.Conv1d
    let layer_norm: GroupNorm
    let activation: GELU

    init(config: HubertConfig, inChannels: Int, outChannels: Int, kernelSize: Int, stride: Int) {
        self.conv = MLXNN.Conv1d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: kernelSize,
            stride: stride,
            padding: 0,
            bias: false
        )
        self.layer_norm = GroupNorm(groupCount: outChannels, dimensions: outChannels, affine: true)
        // CRITICAL: Precise GELU activation (not approximation)
        self.activation = GELU()
        super.init()
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        var x = conv(hiddenStates)
        x = layer_norm(x)
        x = activation(x)  // Apply GELU
        return x
    }
}

class HubertNoLayerNormConvLayer: Module {
    let conv: MLXNN.Conv1d
    let activation: GELU

    init(config: HubertConfig, inChannels: Int, outChannels: Int, kernelSize: Int, stride: Int) {
        self.conv = MLXNN.Conv1d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: kernelSize,
            stride: stride,
            padding: 0,
            bias: false
        )
        // CRITICAL: Precise GELU activation (not approximation)
        self.activation = GELU()
        super.init()
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        var x = conv(hiddenStates)
        x = activation(x)  // Apply GELU
        return x
    }
}

class HubertFeatureExtractor: Module {
    let l0: HubertGroupNormConvLayer
    let l1: HubertNoLayerNormConvLayer
    let l2: HubertNoLayerNormConvLayer
    let l3: HubertNoLayerNormConvLayer
    let l4: HubertNoLayerNormConvLayer
    let l5: HubertNoLayerNormConvLayer
    let l6: HubertNoLayerNormConvLayer
    
    init(config: HubertConfig) {
        self.l0 = HubertGroupNormConvLayer(config: config, inChannels: 1, outChannels: 512, kernelSize: 10, stride: 5)
        self.l1 = HubertNoLayerNormConvLayer(config: config, inChannels: 512, outChannels: 512, kernelSize: 3, stride: 2)
        self.l2 = HubertNoLayerNormConvLayer(config: config, inChannels: 512, outChannels: 512, kernelSize: 3, stride: 2)
        self.l3 = HubertNoLayerNormConvLayer(config: config, inChannels: 512, outChannels: 512, kernelSize: 3, stride: 2)
        self.l4 = HubertNoLayerNormConvLayer(config: config, inChannels: 512, outChannels: 512, kernelSize: 3, stride: 2)
        self.l5 = HubertNoLayerNormConvLayer(config: config, inChannels: 512, outChannels: 512, kernelSize: 2, stride: 2)
        self.l6 = HubertNoLayerNormConvLayer(config: config, inChannels: 512, outChannels: 512, kernelSize: 2, stride: 2)
        super.init()
    }
    
    func callAsFunction(_ inputValues: MLXArray) -> MLXArray {
        var h = inputValues.expandedDimensions(axis: -1)
        h = l0(h)
        h = l1(h)
        h = l2(h)
        h = l3(h)
        h = l4(h)
        h = l5(h)
        h = l6(h)
        return h
    }
}

class HubertFeatureProjection: Module {
    let layer_norm: LayerNorm
    let projection: Linear
    let dropout: Dropout
    
    init(config: HubertConfig) {
        self.layer_norm = LayerNorm(dimensions: 512, eps: config.layerNormEps)
        self.projection = Linear(512, config.hiddenSize)
        self.dropout = Dropout(p: config.hiddenDropoutProb)
        super.init()
    }
    
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = layer_norm(x)
        out = projection(out)
        out = dropout(out)
        return out
    }
}

class HubertEncoder: Module {
    let pos_conv_embed: HubertPositionalConvEmbedding
    let layer_norm: LayerNorm
    let dropout: Dropout
    
    // Explicit properties for 12 layers to ensure weight loading
    let l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11: HubertLayer
    
    init(config: HubertConfig) {
        self.pos_conv_embed = HubertPositionalConvEmbedding(config: config)
        self.layer_norm = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        self.dropout = Dropout(p: config.hiddenDropoutProb)
        
        self.l0 = HubertLayer(config: config)
        self.l1 = HubertLayer(config: config)
        self.l2 = HubertLayer(config: config)
        self.l3 = HubertLayer(config: config)
        self.l4 = HubertLayer(config: config)
        self.l5 = HubertLayer(config: config)
        self.l6 = HubertLayer(config: config)
        self.l7 = HubertLayer(config: config)
        self.l8 = HubertLayer(config: config)
        self.l9 = HubertLayer(config: config)
        self.l10 = HubertLayer(config: config)
        self.l11 = HubertLayer(config: config)
        
        super.init()
    }
    
    func callAsFunction(_ hiddenStates: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        var x = hiddenStates
        x = pos_conv_embed(x)
        x = layer_norm(x)
        x = dropout(x)
        
        x = l0(x, mask: mask)
        x = l1(x, mask: mask)
        x = l2(x, mask: mask)
        x = l3(x, mask: mask)
        x = l4(x, mask: mask)
        x = l5(x, mask: mask)
        x = l6(x, mask: mask)
        x = l7(x, mask: mask)
        x = l8(x, mask: mask)
        x = l9(x, mask: mask)
        x = l10(x, mask: mask)
        x = l11(x, mask: mask)
        
        return x
    }
}

public class HubertModel: Module {
    let config: HubertConfig
    let feature_extractor: HubertFeatureExtractor
    let feature_projection: HubertFeatureProjection
    let encoder: HubertEncoder
    let final_proj: Linear
    
    public init(config: HubertConfig = HubertConfig()) {
        self.config = config
        self.feature_extractor = HubertFeatureExtractor(config: config)
        self.feature_projection = HubertFeatureProjection(config: config)
        self.encoder = HubertEncoder(config: config)
        self.final_proj = Linear(config.hiddenSize, config.classifierProjSize)
        super.init()
    }
    
    public func callAsFunction(_ inputValues: MLXArray) -> MLXArray {
        var x = feature_extractor(inputValues)
        x = feature_projection(x)
        x = encoder(x)
        // x = final_proj(x) // Skip projection for RVC v2 (model expects 768-dim)
        print("DEBUG: HubertModel output shape: \(x.shape)")
        return x
    }
}
