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
    # Ensure n_channels is a Python int for slicing
    if hasattr(n_channels, '__getitem__'):
        n_channels_int = int(n_channels[0].item())
    elif hasattr(n_channels, 'item'):
        n_channels_int = int(n_channels.item())
    else:
        n_channels_int = int(n_channels)

    in_act = input_a + input_b

    # MLX slicing
    t_act = mx.tanh(in_act[:, :, :n_channels_int])
    s_act = mx.sigmoid(in_act[:, :, n_channels_int:])

    acts = t_act * s_act
    return acts


def slice_segments(x: mx.array, ids_str: mx.array, segment_size: int, time_first: bool = True) -> mx.array:
    """
    Slice segments from input tensor.

    Args:
        x: Input tensor of shape (B, T, C) for MLX (time_first=True) or (B, C, T) for PyTorch format
        ids_str: Start indices of shape (B,)
        segment_size: Size of segments to extract (in time dimension)
        time_first: If True, assumes (B, T, C) MLX format. If False, assumes (B, C, T) PyTorch format.

    Returns:
        Sliced segments of shape (B, segment_size, C) or (B, C, segment_size) depending on format
    """
    batch_size = x.shape[0]
    segments = []

    for i in range(batch_size):
        start_idx = int(ids_str[i].item())
        if x.ndim == 3:
            if time_first:
                # MLX format: (B, T, C) - slice along time dimension (dim 1)
                segment = x[i, start_idx:start_idx + segment_size, :]
            else:
                # PyTorch format: (B, C, T) - slice along time dimension (dim 2)
                segment = x[i, :, start_idx:start_idx + segment_size]
        else:
            # 2D tensor (B, T) - slice along time dimension
            segment = x[i, start_idx:start_idx + segment_size]
        segments.append(segment)

    return mx.stack(segments, axis=0)


def rand_slice_segments(x: mx.array, x_lengths: mx.array, segment_size: int, time_first: bool = True):
    """
    Randomly slice segments from input tensor.

    Args:
        x: Input tensor of shape (B, T, C) for MLX format or (B, C, T) for PyTorch format
        x_lengths: Length of each sequence in batch (B,)
        segment_size: Size of segments to extract (in time dimension)
        time_first: If True, assumes (B, T, C) MLX format. If False, assumes (B, C, T) PyTorch format.

    Returns:
        Tuple of (sliced_segments, start_indices)
    """
    batch_size = x.shape[0]

    # Calculate max start indices for each sample
    # Ensure we don't go past the end of the sequence
    max_starts = x_lengths - segment_size
    max_starts = mx.maximum(max_starts, mx.zeros_like(max_starts))

    # Generate random start indices
    # Use uniform random and scale by max_starts
    rand_vals = mx.random.uniform(shape=(batch_size,))
    ids_str = (rand_vals * max_starts.astype(mx.float32)).astype(mx.int32)

    # Slice segments
    segments = slice_segments(x, ids_str, segment_size, time_first=time_first)

    return segments, ids_str


def rand_slice_segments_with_pitch(
    x: mx.array,
    pitch: mx.array,
    x_lengths: mx.array,
    segment_size: int,
    pitch_size: int,
):
    """
    Randomly slice segments from both audio and pitch features.

    Args:
        x: Audio features (B, C, T)
        pitch: Pitch features (B, T_pitch)
        x_lengths: Length of each sequence
        segment_size: Segment size for audio
        pitch_size: Segment size for pitch

    Returns:
        Tuple of (audio_segments, pitch_segments, start_indices)
    """
    batch_size = x.shape[0]

    # Calculate ratio between audio and pitch
    audio_len = x.shape[-1]
    pitch_len = pitch.shape[-1]

    # Calculate max start indices
    max_starts = x_lengths - segment_size
    max_starts = mx.maximum(max_starts, mx.zeros_like(max_starts))

    # Generate random start indices for audio
    rand_vals = mx.random.uniform(shape=(batch_size,))
    ids_str = (rand_vals * max_starts.astype(mx.float32)).astype(mx.int32)

    # Calculate corresponding pitch start indices
    # Pitch features are typically at a different rate (hop_size)
    if audio_len > 0 and pitch_len > 0:
        ratio = pitch_len / audio_len
        pitch_ids_str = (ids_str.astype(mx.float32) * ratio).astype(mx.int32)
    else:
        pitch_ids_str = ids_str

    # Slice segments
    audio_segments = slice_segments(x, ids_str, segment_size)
    pitch_segments = slice_segments(pitch, pitch_ids_str, pitch_size)

    return audio_segments, pitch_segments, ids_str
