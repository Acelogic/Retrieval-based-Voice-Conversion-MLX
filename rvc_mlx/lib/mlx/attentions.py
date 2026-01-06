import math
import mlx.core as mx
import mlx.nn as nn
from rvc_mlx.lib.mlx.commons import convert_pad_shape

class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: int,
        n_heads: int,
        p_dropout: float = 0.0,
        window_size: int = None,
        heads_share: bool = True,
        block_length: int = None,
        proximal_bias: bool = False,
        proximal_init: bool = False,
    ):
        super().__init__()
        assert (
            channels % n_heads == 0
        ), "Channels must be divisible by the number of heads."

        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.k_channels = channels // n_heads
        self.window_size = window_size
        self.block_length = block_length
        self.proximal_bias = proximal_bias

        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.conv_o = nn.Conv1d(channels, out_channels, 1)
        self.drop = nn.Dropout(p_dropout)

        if window_size:
            n_heads_rel = 1 if heads_share else n_heads
            rel_stddev = self.k_channels**-0.5
            self.emb_rel_k = mx.random.normal(
                shape=(n_heads_rel, 2 * window_size + 1, self.k_channels)
            ) * rel_stddev
            self.emb_rel_v = mx.random.normal(
                shape=(n_heads_rel, 2 * window_size + 1, self.k_channels)
            ) * rel_stddev
            # Note: In MLX we need to register these as parameters manually if we were training
            # For inference, if we load weights, they will be assigned.

    def __call__(self, x, c, attn_mask=None):
        q, k, v = self.conv_q(x), self.conv_k(c), self.conv_v(c)
        x, _ = self.attention(q, k, v, mask=attn_mask)
        return self.conv_o(x)

    def attention(self, query, key, value, mask=None):
        b, d, t_s = key.shape
        t_t = query.shape[2] # Last dim is time in Conv1d (B, C, T) -> MLX is typically (B, T, C) ... wait

        # CRITICAL: PyTorch Conv1d is (Batch, Channel, Length).
        # MLX Conv1d is (Batch, Length, Channel).
        # Assuming we run input transpositions before calling the model or handle it in the model.
        # RVC PyTorch code typically expects (B, C, T). 
        # For MLX port, we should stick to MLX convention (B, T, C) to use nn.Conv1d efficiently, 
        # OR transpose inputs/outputs at every layer. 
        # Transposing at every layer is inefficient but safest for porting "module by module".
        # Let's inspect modules.py again - I used nn.Conv1d. 
        # MLX nn.Conv1d expects input (N, L, C). 
        # PyTorch code feeds (N, C, L).
        # I MUST handle this. I will assume the Inputs to the whole model are converted to (N, L, C).
        # But wait, looking at my modules.py I implemented standard MLX Conv1d.
        # So `x` coming in here is (N, L, C).
        
        b, t_s, d = key.shape
        t_t = query.shape[1]
        
        # Reshape for multi-head: (B, T, Heads, HeadDim) -> transpose to (B, Heads, T, HeadDim) for matmul?
        # MLX: reshape to (B, T, Heads, HeadDim)
        query = query.reshape(b, t_t, self.n_heads, self.k_channels).transpose(0, 2, 1, 3) 
        key = key.reshape(b, t_s, self.n_heads, self.k_channels).transpose(0, 2, 1, 3)
        value = value.reshape(b, t_s, self.n_heads, self.k_channels).transpose(0, 2, 1, 3)
        
        # query: (B, Heads, T_t, k_dim)
        # key: (B, Heads, T_s, k_dim) 
        # matmul -> (B, Heads, T_t, T_s)
        
        scores = mx.matmul(query / math.sqrt(self.k_channels), key.transpose(0, 1, 3, 2))

        if self.window_size:
             # Relative attention implementation omitted for brevity/complexity in initial port 
             # unless critical. It seems used.
             # Note: Implementing fully requires custom relative embedding ops.
             # Check if RVC utilizes window_size usually. Yes.
             # I will implement a simplified version or come back to it.
             # Let's try to implement _compute_relative_scores
             scores += self._compute_relative_scores(query, t_s)
        
        if mask is not None:
            # Mask is typically (B, 1, 1, T_s) or similar.
            # Convert -1e4 to -inf logic if needed, but -1e4 is fine.
            # mask expected shape (B, 1, 1, T) or matching compatible dims
            scores = mx.where(mask == 0, -1e4, scores)

        p_attn = self.drop(mx.softmax(scores, axis=-1))
        
        output = mx.matmul(p_attn, value)
        # output: (B, Heads, T_t, k_dim)
        
        if self.window_size:
            output += self._apply_relative_values(p_attn, t_s)
            
        # Reshape back to (B, T_t, C)
        output = output.transpose(0, 2, 1, 3).reshape(b, t_t, d)
        return output, p_attn

    def _compute_relative_scores(self, query, length):
        """Compute relative position scores for keys."""
        rel_emb = self._get_relative_embeddings(self.emb_rel_k, length)
        rel_logits = self._matmul_with_relative_keys(
            query / math.sqrt(self.k_channels), rel_emb
        )
        return self._relative_position_to_absolute_position(rel_logits) 
        
    def _apply_relative_values(self, p_attn, length):
        """Apply relative position embeddings to values."""
        rel_weights = self._absolute_position_to_relative_position(p_attn)
        rel_emb = self._get_relative_embeddings(self.emb_rel_v, length)
        return self._matmul_with_relative_values(rel_weights, rel_emb)

    # Helper methods for relative position embeddings
    def _matmul_with_relative_values(self, x, y):
        """Matmul with relative values."""
        # x: (B, Heads, T, 2*T-1), y: (Heads, 2*T-1, HeadDim) or (1, 2*T-1, HeadDim)
        # Unsqueeze y to (1, Heads, 2*T-1, HeadDim) if needed
        return mx.matmul(x, mx.expand_dims(y, 0))

    def _matmul_with_relative_keys(self, x, y):
        """Matmul with relative keys."""
        # x: (B, Heads, T, HeadDim), y: (Heads, 2*T-1, HeadDim)
        # transpose y: (Heads, HeadDim, 2*T-1)
        return mx.matmul(x, mx.expand_dims(y, 0).transpose(0, 1, 3, 2))

    def _get_relative_embeddings(self, embeddings, length):
        """Extract relative embeddings for given length."""
        # embeddings: (Heads, 2*window_size+1, HeadDim)
        pad_length = max(length - (self.window_size + 1), 0)
        start = max((self.window_size + 1) - length, 0)
        end = start + 2 * length - 1

        if pad_length > 0:
            # Pad on dimension 1 (middle dimension)
            # MLX pad expects list of tuples: [(axis0_left, axis0_right), (axis1_left, axis1_right), ...]
            embeddings = mx.pad(embeddings, pad_width=[(0, 0), (pad_length, pad_length), (0, 0)])
        return embeddings[:, start:end, :]

    def _relative_position_to_absolute_position(self, x):
        """Convert relative position to absolute position indexing."""
        batch, heads, length, _ = x.shape

        # Pad with one column on the right
        x = mx.pad(x, pad_width=[(0, 0), (0, 0), (0, 0), (0, 1)])

        # Reshape to (B, Heads, length * 2 * length)
        x_flat = x.reshape(batch, heads, length * 2 * length)

        # Pad with (length - 1) elements
        x_flat = mx.pad(x_flat, pad_width=[(0, 0), (0, 0), (0, length - 1)])

        # Reshape and slice
        x_final = x_flat.reshape(batch, heads, length + 1, 2 * length - 1)
        return x_final[:, :, :length, length - 1:]

    def _absolute_position_to_relative_position(self, x):
        """Convert absolute position to relative position indexing."""
        batch, heads, length, _ = x.shape

        # Pad with (length - 1) columns on the right
        x = mx.pad(x, pad_width=[(0, 0), (0, 0), (0, 0), (0, length - 1)])

        # Reshape to (B, Heads, length^2 + length*(length-1))
        x_flat = x.reshape(batch, heads, length**2 + length * (length - 1))

        # Pad with length elements on the left
        x_flat = mx.pad(x_flat, pad_width=[(0, 0), (0, 0), (length, 0)])

        # Reshape to (B, Heads, length, 2*length) and slice off first column
        x_final = x_flat.reshape(batch, heads, length, 2 * length)
        return x_final[:, :, :, 1:]

class FFN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        filter_channels: int,
        kernel_size: int,
        p_dropout: float = 0.0,
        activation: str = None,
        causal: bool = False,
    ):
        super().__init__()
        self.causal = causal
        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size)
        self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size)
        self.drop = nn.Dropout(p_dropout)
        self.activation = activation
        self.kernel_size = kernel_size


    def __call__(self, x, x_mask):
        # x: (N, L, C)
        # Padding logic: MLX Conv1d supports 'same' or 'valid' or int. 
        # PyTorch uses explicitly calculated padding.
        # We can simulate padding by manually padding input.
        
        # Calculate padding needed for "same" (kernel_size-1)//2 for odd kernels
        pad_total = self.kernel_size - 1
        pad_l, pad_r = pad_total // 2, pad_total // 2
        if self.causal:
            pad_l, pad_r = pad_total, 0
            
        x_padded = mx.pad(x * x_mask, ((0,0), (pad_l, pad_r), (0,0)))
        x = self.conv_1(x_padded)
        
        if self.activation == "gelu":
            x = x * mx.sigmoid(1.702 * x)
        else:
            x = mx.maximum(x, 0) # relu
            
        x = self.drop(x)
        
        pad_total = self.kernel_size - 1

        pad_l, pad_r = pad_total // 2, pad_total // 2
        # FFN usually non-causal for 2nd layer? matching orig code
        # _same_padding in original uses (kernel-1)//2
        
        x_padded = mx.pad(x * x_mask, ((0,0), (pad_l, pad_r), (0,0)))
        x = self.conv_2(x_padded)
        return x * x_mask
