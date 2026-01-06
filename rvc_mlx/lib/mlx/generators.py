import math
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from rvc_mlx.lib.mlx.modules import WaveNet
from rvc_mlx.lib.mlx.residuals import ResBlock, LRELU_SLOPE


class SineGenerator(nn.Module):
    def __init__(
        self,
        sample_rate: int,
        harmonic_num: int = 0,
        sine_amp: float = 0.1,
        add_noise_std: float = 0.003,
        voiced_threshod: float = 0,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.harmonic_num = harmonic_num
        self.sine_amp = sine_amp
        self.noise_std = add_noise_std
        self.voiced_threshold = voiced_threshod
        self.waveform_dim = harmonic_num + 1

    def _compute_voiced_unvoiced(self, f0):
        return (f0 > self.voiced_threshold).astype(mx.float32)

    def _generate_sine_wave(self, f0, upsampling_factor):
        # f0: (B, L, 1)
        B, L, _ = f0.shape
        
        # Upsampling grid
        upsampling_grid = mx.arange(1, upsampling_factor + 1).astype(f0.dtype)
        # grid shape (upsample,) -> broadcast with (B, L, 1) to (B, L, upsample)?
        # We need to flatten time.
        
        # RVC Torch logic:
        # phase_increments = (f0 / sr) * grid
        # phase_remainder ...
        
        # We need (B, L, upsampling)
        # f0 is (B, L, 1)
        phase_increments = (f0 / self.sample_rate) * upsampling_grid[None, None, :]
        
        # (B, L, U)
        # Check torch logic:
        # phase_remainder = torch.fmod(phase_increments[:, :-1, -1:] + 0.5, 1.0) - 0.5
        # This takes the phase of the LAST sample of previous frame?
        # phase_increments[..., -1:] is the last upsampled point of each frame.
        
        # Slice: all batch, all but last length, last upsample point
        prev_last_phase = phase_increments[:, :-1, -1:] 
        phase_remainder = (prev_last_phase + 0.5) % 1.0 - 0.5
        
        # Cumsum the remainder?
        # cumulative_phase = phase_remainder.cumsum(dim=1).fmod(1.0)
        cumulative_phase = mx.cumsum(phase_remainder, axis=1) % 1.0
        
        # Pad strict: (0,0, 1,0) -> pad 1 at start of dim 1
        # cumulative_phase shape (B, L-1, 1). Pad to (B, L, 1)
        # Pad with 0 at start.
        cumulative_phase = mx.pad(cumulative_phase, ((0,0), (1,0), (0,0)))
        
        # Add to phase_increments
        phase_increments = phase_increments + cumulative_phase
        
        # Reshape to (B, L*U, 1)
        phase_increments = phase_increments.reshape(B, -1, 1)
        
        # Scale for harmonics: (1, 1, H)
        harmonic_scale = mx.arange(1, self.waveform_dim + 1).astype(f0.dtype).reshape(1, 1, -1)
        phase_increments = phase_increments * harmonic_scale
        
        # Random phase
        random_phase = mx.random.uniform(shape=(1, 1, self.waveform_dim)).astype(f0.dtype) 
        # Range? Torch rand is [0,1).
        # Torch code: "random_phase[..., 0] = 0"
        # Since MLX doesn't support inplace item assignment easily on array, we construct carefully.
        # Actually random frequency offset? No, simple phase offset.
        
        # Masking fundamental (idx 0) to 0
        mask = mx.ones((1, 1, self.waveform_dim), dtype=f0.dtype)
        # Can't index assign easily.
        # Use concat: 0 for first, rand for rest.
        idx0 = mx.zeros((1, 1, 1), dtype=f0.dtype)
        idxRest = mx.random.uniform(shape=(1, 1, self.waveform_dim - 1)).astype(f0.dtype)
        random_phase = mx.concatenate([idx0, idxRest], axis=-1)
        
        phase_increments = phase_increments + random_phase
        
        return mx.sin(2 * np.pi * phase_increments)

    def __call__(self, f0, upsampling_factor):
        # f0 input might be (B, L). Expand to (B, L, 1)
        if f0.ndim == 2:
            f0 = f0[:, :, None]
            
        sine_waves = self._generate_sine_wave(f0, upsampling_factor) * self.sine_amp
        
        voiced_mask = self._compute_voiced_unvoiced(f0)
        
        # Upsample mask: (B, L, 1) -> (B, L*U, 1)
        # Use simple repeat
        B, L, C = voiced_mask.shape
        voiced_mask = mx.repeat(voiced_mask, upsampling_factor, axis=1)
        
        noise_amp = voiced_mask * self.noise_std + (1 - voiced_mask) * (self.sine_amp / 3)
        noise = noise_amp * mx.random.normal(sine_waves.shape).astype(sine_waves.dtype)
        
        sine_waveforms = sine_waves * voiced_mask + noise
        return sine_waveforms, voiced_mask, noise


class SourceModuleHnNSF(nn.Module):
    def __init__(
        self,
        sample_rate: int,
        harmonic_num: int = 0,
        sine_amp: float = 0.1,
        add_noise_std: float = 0.003,
        voiced_threshod: float = 0,
    ):
        super().__init__()
        self.l_sin_gen = SineGenerator(sample_rate, harmonic_num, sine_amp, add_noise_std, voiced_threshod)
        self.l_linear = nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = nn.Tanh()

    def __call__(self, x, upsample_factor: int = 1):
        # x is F0
        sine_wavs, uv, _ = self.l_sin_gen(x, upsample_factor)
        # sine_wavs: (B, T_high, Harmonics)
        # Linear expects (..., H). Correct.
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))
        return sine_merge

class HiFiGANNSFGenerator(nn.Module):
    def __init__(
        self,
        initial_channel: int,
        resblock_kernel_sizes: list,
        resblock_dilation_sizes: list,
        upsample_rates: list,
        upsample_initial_channel: int,
        upsample_kernel_sizes: list,
        gin_channels: int,
        sr: int,
    ):
        super().__init__()
        self.upsample_rates = upsample_rates
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.m_source = SourceModuleHnNSF(sample_rate=sr, harmonic_num=0)
        
        self.conv_pre = nn.Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)

        self.ups = []
        self.noise_convs = []
        self.output_paddings = []  # Store output_padding for manual application
        
        channels = []
        # Replicate channel calculation logic
        for i in range(len(upsample_rates)):
            channels.append(upsample_initial_channel // (2 ** (i + 1)))

        stride_f0s = []
        for i in range(len(upsample_rates)):
             if i + 1 < len(upsample_rates):
                 stride = math.prod(upsample_rates[i + 1 :])
             else:
                 stride = 1
             stride_f0s.append(stride)

        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            # Padding - match PyTorch behavior for odd upsample rates
            if u % 2 == 0:
                p = (k - u) // 2
            else:
                p = u // 2 + u % 2
                
            in_ch = upsample_initial_channel // (2**i)
            out_ch = channels[i]
            
            # Store output_padding to apply manually (MLX doesn't support it natively)
            self.output_paddings.append(u % 2)
            
            # Use native ConvTranspose1d (output_padding applied in forward)
            self.ups.append(nn.ConvTranspose1d(
                in_ch, out_ch, k, stride=u, padding=p
            ))
            
            stride = stride_f0s[i]
            kernel = 1 if stride == 1 else stride * 2 - stride % 2
            padding = 0 if stride == 1 else (kernel - stride) // 2
            
            self.noise_convs.append(nn.Conv1d(1, channels[i], kernel, stride=stride, padding=padding))

        self.resblocks = []
        for i in range(len(self.ups)):
            ch = channels[i]
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(ResBlock(ch, k, d))

        self.conv_post = nn.Conv1d(channels[-1], 1, 7, 1, padding=3) # remove bias? orig said bias=False
        
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

        self.upp = math.prod(upsample_rates)
        self.lrelu_slope = LRELU_SLOPE
        
        # Attribute registration
        for i, l in enumerate(self.ups): setattr(self, f"up_{i}", l)
        for i, l in enumerate(self.noise_convs): setattr(self, f"noise_conv_{i}", l)
        for i, l in enumerate(self.resblocks): setattr(self, f"resblock_{i}", l)

    def __call__(self, x, f0, g=None):
        # Input x is in PyTorch format (B, C, T), transpose to MLX format (B, T, C)
        x = x.transpose(0, 2, 1)  # (B, C, T) -> (B, T, C)

        har_source = self.m_source(f0, self.upp)

        x = self.conv_pre(x)
        
        if g is not None:
            x = x + self.cond(g)

        for i in range(self.num_upsamples):
            up = getattr(self, f"up_{i}")
            noise_conv = getattr(self, f"noise_conv_{i}")
            
            x = nn.leaky_relu(x, self.lrelu_slope)
            x = up(x)
            
            # Apply output_padding manually (MLX doesn't support it in ConvTranspose1d)
            # output_padding adds to one side of the output to match PyTorch behavior
            out_pad = self.output_paddings[i]
            if out_pad > 0:
                # Pad on the right side of the time dimension (dim 1 in NLC format)
                x = mx.pad(x, [(0, 0), (0, out_pad), (0, 0)])
            
            n = noise_conv(har_source)
            
            if x.shape[1] != n.shape[1]:
                min_len = min(x.shape[1], n.shape[1])
                x = x[:, :min_len, :]
                n = n[:, :min_len, :]
            
            x = x + n
            
            xs = None
            for j in range(self.num_kernels):
                resblock = getattr(self, f"resblock_{i * self.num_kernels + j}")
                out = resblock(x)
                if xs is None:
                    xs = out
                else:
                    xs = xs + out
            x = xs / self.num_kernels
            
        x = nn.leaky_relu(x)
        x = self.conv_post(x)
        x = mx.tanh(x)

        # conv_post reduces to 1 channel: x is (B, T, 1)
        # Keep in (B, T, 1) format for pipeline compatibility
        # Pipeline expects (B, T, 1) and extracts with [0, :, 0]

        return x
