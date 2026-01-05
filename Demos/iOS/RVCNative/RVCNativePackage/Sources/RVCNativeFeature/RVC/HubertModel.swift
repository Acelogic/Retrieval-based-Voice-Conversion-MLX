import Foundation
import MLX
import MLXNN
import MLXRandom

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
    public var layerNormEps: Float = 1e-12
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
    let weight: MLXArray
    let bias: MLXArray
    let groups: Int = 16
    let padding: Int = 64
    
    init(config: HubertConfig) {
        let hiddenSize = config.hiddenSize
        let kernelSize = 128
        let inChannelsPerGroup = hiddenSize / self.groups
        
        self.weight = MLXArray.zeros([hiddenSize, kernelSize, inChannelsPerGroup])
        self.bias = MLXArray.zeros([hiddenSize])
        super.init()
    }
    
    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        var out = MLX.conv1d(hiddenStates, weight, stride: 1, padding: padding, groups: groups)
        out = out + bias
        let L = out.shape[1]
        out = out[0..., 0..<(L-1), 0...]
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
        self.activation = GELU()
        super.init()
    }
    
    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        var x = conv(hiddenStates)
        x = layer_norm(x)
        x = activation(x)
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
        self.activation = GELU()
        super.init()
    }
    
    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        var x = conv(hiddenStates)
        x = activation(x)
        return x
    }
}

class HubertFeatureExtractor: Module {
    let conv_layers: [Module]
    
    init(config: HubertConfig) {
        var layers: [Module] = []
        layers.append(HubertGroupNormConvLayer(config: config, inChannels: 1, outChannels: 512, kernelSize: 10, stride: 5))
        
        let kernels = [3, 3, 3, 3, 2, 2]
        let strides = [2, 2, 2, 2, 2, 2]
        
        for (k, s) in zip(kernels, strides) {
             layers.append(HubertNoLayerNormConvLayer(config: config, inChannels: 512, outChannels: 512, kernelSize: k, stride: s))
        }
        
        self.conv_layers = layers
        super.init()
    }
    
    func callAsFunction(_ inputValues: MLXArray) -> MLXArray {
        var h = inputValues.expandedDimensions(axis: -1)
        for layer in conv_layers {
            if let l = layer as? HubertGroupNormConvLayer {
                h = l(h)
            } else if let l = layer as? HubertNoLayerNormConvLayer {
                h = l(h)
            }
        }
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
    let layers: [HubertLayer]
    
    init(config: HubertConfig) {
        self.pos_conv_embed = HubertPositionalConvEmbedding(config: config)
        self.layer_norm = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        self.dropout = Dropout(p: config.hiddenDropoutProb)
        
        var l: [HubertLayer] = []
        for _ in 0..<config.numHiddenLayers {
            l.append(HubertLayer(config: config))
        }
        self.layers = l
        super.init()
    }
    
    func callAsFunction(_ hiddenStates: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        var x = hiddenStates
        x = pos_conv_embed(x)
        x = layer_norm(x)
        x = dropout(x)
        
        for layer in layers {
            x = layer(x, mask: mask)
        }
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
        x = final_proj(x)
        return x
    }
}
