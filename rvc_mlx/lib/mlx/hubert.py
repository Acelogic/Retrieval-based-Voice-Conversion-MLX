import math
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

class HubertConfig:
    def __init__(
        self,
        vocab_size: int = 32,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 2,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-5,
        pad_token_id: int = 1,
        bos_token_id: int = 0,
        eos_token_id: int = 2,
        classifier_proj_size: int = 256, # For "WithFinalProj"
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.classifier_proj_size = classifier_proj_size

class HubertAttention(nn.Module):
    def __init__(self, config: HubertConfig):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.q_proj = nn.Linear(config.hidden_size, self.all_head_size)
        self.k_proj = nn.Linear(config.hidden_size, self.all_head_size)
        self.v_proj = nn.Linear(config.hidden_size, self.all_head_size)
        
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
        self.scale = self.attention_head_size ** -0.5

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        
        B, L, C = hidden_states.shape
        
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape for MultiHead: (B, L, H, D) -> (B, H, L, D) for easier attention?
        # MLX Attention is flexible.
        # Let's use (B, L, H, D)
        
        q = q.reshape(B, L, self.num_attention_heads, self.attention_head_size).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.num_attention_heads, self.attention_head_size).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.num_attention_heads, self.attention_head_size).transpose(0, 2, 1, 3)
        
        # Attention scores
        # (B, H, L, D) @ (B, H, D, L) -> (B, H, L, L)
        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        
        if attention_mask is not None:
             # Mask shape broadcast
             scores = scores + attention_mask
             
        probs = mx.softmax(scores, axis=-1)
        probs = self.dropout(probs)
        
        # Weighted sum: (B, H, L, L) @ (B, H, L, D) -> (B, H, L, D)
        # Wait, v is (B, H, L, D).
        out = probs @ v
        
        # Reshape back to (B, L, H, D) -> (B, L, C)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, self.all_head_size)
        
        out = self.out_proj(out)
        return out

class HubertFeedForward(nn.Module):
    def __init__(self, config: HubertConfig):
        super().__init__()
        self.intermediate_dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output_dense = nn.Linear(config.intermediate_size, config.hidden_size)
        if config.hidden_act == "gelu":
            self.act = nn.GELU()
        else:
            self.act = nn.ReLU() # Default fallback
            
    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.output_dense(hidden_states)
        return hidden_states

class HubertLayer(nn.Module):
    def __init__(self, config: HubertConfig):
        super().__init__()
        self.attention = HubertAttention(config)
        self.feed_forward = HubertFeedForward(config)
        
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def __call__(self, hidden_states: mx.array, attention_mask: Optional[mx.array] = None) -> mx.array:
        # Attention block
        attn_output = self.attention(hidden_states, attention_mask)
        hidden_states = self.layer_norm(hidden_states + self.dropout(attn_output))
        
        # Feed-forward block
        ffn_output = self.feed_forward(hidden_states)
        hidden_states = self.final_layer_norm(hidden_states + self.dropout(ffn_output))
        
        return hidden_states

class HubertEncoder(nn.Module):
    def __init__(self, config: HubertConfig):
        super().__init__()
        self.pos_conv_embed = HubertPositionalConvEmbedding(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.layers = [HubertLayer(config) for _ in range(config.num_hidden_layers)]
        
    def __call__(self, hidden_states: mx.array, attention_mask: Optional[mx.array] = None) -> mx.array:
        # Positional Embeddings
        hidden_states = self.pos_conv_embed(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        for l in self.layers:
            hidden_states = l(hidden_states, attention_mask)
        return hidden_states

class HubertModel(nn.Module):
    def __init__(self, config: HubertConfig):
        super().__init__()
        self.config = config
        
        self.feature_extractor = HubertFeatureExtractor(config)
        self.feature_projection = HubertFeatureProjection(config)
        
        self.encoder = HubertEncoder(config)
        
        # For "WithFinalProj"
        self.final_proj = nn.Linear(config.hidden_size, config.classifier_proj_size)

    def __call__(self, input_values: mx.array, **kwargs) -> mx.array:
        # input_values: (B, T_samples)
        
        extract_features = self.feature_extractor(input_values)
        # extract_features = extract_features.transpose(0, 2, 1) # MLX Conv1d returns (N, L, C) already

        
        hidden_states = self.feature_projection(extract_features)
        
        # Encoder handles pos_embed, norm, dropout, layers
        hidden_states = self.encoder(hidden_states)
        
        # Proj
        if self.config.classifier_proj_size != self.config.hidden_size:
            proj = self.final_proj(hidden_states)
        else:
            proj = hidden_states
            
        return proj


class HubertFeatureProjection(nn.Module):
    def __init__(self, config):
        super().__init__()
        # In standard Hubert, LayerNorm is applied to the 512-dim features 
        # BEFORE the projection to 768 hidden_size.
        self.layer_norm = nn.LayerNorm(512, eps=config.layer_norm_eps)
        self.projection = nn.Linear(512, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def __call__(self, x):
        return self.dropout(self.projection(self.layer_norm(x)))

class HubertGroupNormConvLayer(nn.Module):
    def __init__(self, config, in_channels, out_channels, kernel_size, stride, bias=False):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            bias=bias,
        )
        self.activation = nn.GELU()
        
        # GroupNorm with num_groups=in_channels (which is "LayerNorm" over channels if N=C? No, it's independent)
        # HF uses GroupNorm/LayerNorm.
        # "conv_feature_layers": [(512, 10, 5), (512, 3, 2), ...]
        self.layer_norm = nn.GroupNorm(out_channels, out_channels, affine=True)

    def __call__(self, hidden_states):
        # x: (N, L, C)
        hidden_states = self.conv(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states # (N, L', C')

class HubertNoLayerNormConvLayer(nn.Module):
    def __init__(self, config, in_channels, out_channels, kernel_size, stride, bias=False):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            bias=bias,
        )
        self.activation = nn.GELU()

    def __call__(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states

class HubertFeatureExtractor(nn.Module):
    def __init__(self, config: HubertConfig):
        super().__init__()
        self.conv_layers = []
        
        # Layer 0: GroupNorm
        in_d = 1
        out_d = 512
        self.conv_layers.append(HubertGroupNormConvLayer(config, in_d, out_d, 10, 5))
        
        # Remaining 6 layers
        kernels = [3, 3, 3, 3, 2, 2]
        strides = [2, 2, 2, 2, 2, 2]
        for k, s in zip(kernels, strides):
            self.conv_layers.append(HubertNoLayerNormConvLayer(config, out_d, out_d, k, s))
            
    def __call__(self, input_values):
        # input_values: (B, T)
        # Conv1d expects (B, T, C). We need to create channel dim.
        hidden_states = input_values[:, :, None] 
        
        for l in self.conv_layers:
            hidden_states = l(hidden_states)
            
        return hidden_states

class HubertPositionalConvEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.groups = 16
        self.kernel_size = 128
        self.padding = 64
        
        # Manually manage weights since nn.Conv1d might lack groups arg
        # Weight shape: (Out, K, In/groups)
        in_per_group = self.hidden_size // self.groups
        
        scale = math.sqrt(1 / (in_per_group * self.kernel_size))
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(self.hidden_size, self.kernel_size, in_per_group)
        )
        self.bias = None # Hubert pos conv usually has bias=False? My converter handled bias check. 
                         # Default to None, converter will load if present. 
                         # (Actually converter logic showed bias might exist? 
                         # "encoder.pos_conv_embed.conv.bias: torch.Size([768])".
                         # I need to support bias.
        
        self.bias = mx.zeros((self.hidden_size,)) 
        # But wait, converter check:
        # if "pos_conv_embed.conv.weight" in key: ... manual extraction.
        # if "weight" in key: continue.
        # It fell through to "Bias does not need transpose"?
        # Actually my converter logic:
        # `if "pos_conv_embed.conv.weight" in key:` -> manual extract/transpose/save.
        # Then loop:
        # `if "pos_conv_embed" in key ...`:
        #    `if "weight" in key: continue` (skips duplicate weight).
        #    So it SAVED the bias key if it existed! 
        #    Key name: `encoder.pos_conv_embed.conv.bias`.
        # So I should init bias.

    def __call__(self, hidden_states):
        # hidden_states: (N, L, C)
        
        # Use functional conv1d with groups
        try:
             out = mx.conv1d(
                 hidden_states, 
                 self.weight, 
                 stride=1, 
                 padding=self.padding, 
                 groups=self.groups
             )
        except TypeError:
             # If groups not supported in functional either (unlikely), split.
             # fallback split implementation
             # but check_mlx_ops didn't check functional.
             # assume functional works as it's closer to C++ lib.
             raise
        
        if self.bias is not None:
             out = out + self.bias
             
        # Crop (same-pad logic: kernel=128 is even, remove 1)
        out = out[:, :-1, :]
        
        # GELU activation (was missing!)
        out = nn.gelu(out)
        
        return out + hidden_states
