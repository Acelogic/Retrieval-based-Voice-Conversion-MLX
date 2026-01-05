import mlx.core as mx
import mlx.nn as nn
from typing import Optional

def init_weights(m, mean=0.0, std=0.01):
    # MLX initializes weights during layer construction.
    # We generally load weights for inference, so this might be a no-op or specific to training.
    pass

def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)

def convert_pad_shape(pad_shape):
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape

def sequence_mask(length: mx.array, max_length: Optional[int] = None):
    if max_length is None:
        max_length = length.max().item()
    x = mx.arange(max_length, dtype=length.dtype)
    return x[None, :] < length[:, None]

def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0].item()
    in_act = input_a + input_b
    
    # MLX slicing
    t_act = mx.tanh(in_act[:, :n_channels_int, :])
    s_act = mx.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts
