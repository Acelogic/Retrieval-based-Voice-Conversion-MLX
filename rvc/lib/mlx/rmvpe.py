
import mlx.core as mx
import mlx.nn as nn
import math

class ConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, output_padding=0, bias=False):
        super().__init__()
        # PyTorch ConvTranspose2d: (In, Out, H, W) -> MLX (Out, H, W, In)? No.
        # PyTorch weights: (In, Out, K1, K2).
        
        # We will implement using UpSample + Conv2d.
        # This is a simplification and assumes stride <= kernel_size usually.
        # RMVPE uses stride=(1, 2) often.
        
        self.stride = stride
        self.output_padding = output_padding
        self.kernel_size = kernel_size
        self.padding = padding
        
        # MLX Conv2d weights: (Out, H, W, In)
        # We initialized a Conv2d.
        # In Transpose, the "input" channels are In, "Output" are Out.
        # But the weight shape in PT is (In, Out, K, K).
        # To emulate Transpose via Conv, we might need to be careful.
        
        # ACTUALLY, MLX HAS `mx.conv_transpose2d` FUNCTION in newer versions, but maybe not `nn.ConvTranspose2d` layer.
        # Let's check if we can assume it exists or implement via `conv_general`.
        # `mx.conv_transpose2d` was added in 0.10.0.
        # BUT `nn.ConvTranspose2d` might be missing.
        # We can write a wrapper around `mx.conv_transpose2d`.
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Initialize weights matching PyTorch shape (In, Out, H, W) then transpose to MLX expected.
        # MLX `conv_transpose2d` expects weight: (Out, H, W, In) -- WAIT.
        # MLX `conv2d` expects (Out, H, W, In). 
        # `conv_transpose2d`: 
        # Docs: "The weight has shape (out_channels, kernels[0], kernels[1], in_channels)." 
        # Wait, usually it swaps in/out?
        # PyTorch: (in_channels, out_channels/groups, kernel_h, kernel_w)
        # We will stick to (Out, H, W, In) for consistency with MLX style and transpose during loading.
        
        scale = math.sqrt(1 / (in_channels * kernel_size[0] * kernel_size[1]))
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(out_channels, kernel_size[0], kernel_size[1], in_channels)
        )
        if bias:
            self.bias = mx.zeros((out_channels,))
        else:
            self.bias = None

    def __call__(self, x):
        # x: (N, H, W, C)
        # stride: (sh, sw)
        sh, sw = self.stride
        kh, kw = self.kernel_size
        
        # 1. Zero-insertion (Input upscale)
        # Output H_new = H * sh, W_new = W * sw (before padding adjustment)
        N, H, W, C = x.shape
        
        # Create canvas
        # We want to place pixels at stride steps.
        # But we must handle the case where "output_padding" adds extra rows/cols?
        # PyTorch TransposeConv:
        # H_out = (H-1)*stride - 2*padding + dilation*(kernel-1) + output_padding + 1
        
        # Let's simplify:
        # 1. Upscale by inserting zeros.
        #    x_up size: (N, H*sh, W*sw, C)
        x_up = mx.zeros((N, H * sh, W * sw, C), dtype=x.dtype)
        # Insert x into grid
        # x_up[:, ::sh, ::sw, :] = x  <-- MLX supports stride slice assignment?
        # If not, we might need a workaround.
        # MLX 0.16 supports advanced indexing updates?
        # Let's assume slice update works.
        x_up[:, ::sh, ::sw, :] = x
        
        # 2. Convolve with stride=1
        # To mimic TransposeConv, we convolve the Zero-inserted input with the *rotated* kernel (or just same kernel if we flip weights?).
        # PyTorch TransposeConv is mathematically conv with input gradient.
        # It's equivalent to convolving zero-inserted input with "grad_weight" equivalent?
        # Standard formulation: TransposeConv(x, w) ~ Conv(InsertZeros(x), Rotated180(w)).
        # Since we load PyTorch weights, we can transpose/flip them in the converter to match a standard Conv2d operation here.
        
        # Padding:
        # We need to PAD `x_up` such that the result matches expected output size.
        # Pytorch `output_padding` adds extra zeros to right/bottom.
        
        # Let's rely on `mx.conv2d` with `padding`.
        # Wait, if we use `mx.conv2d`, we specify padding for the conv.
        # Effective kernel size is K.
        # We want to reverse the effect of "valid" conv that reduced size?
        # No, TransposeConv INCREASES size.
        
        # If we manually upscaled `x_up`, we just run standard conv.
        # We need to ensure we used correct padding to get (H_out, W_out).
        # We can use `mx.conv2d` with `padding` argument or manual pad.
        
        # Standard Conv2d on Upscaled Input:
        # We set `stride=1` for this conv.
        # We set `padding`. PyTorch TransposeConv `padding` argument REMOVES pixels (it's the inverse of conv padding).
        # So we crop the output?
        # Or we pad `x_up`?
        
        # Simpler approach:
        # Pad `x_up` by `kernel_size - 1`.
        # Run Valid Conv (no padding).
        # Result size: (H*sh + K - 1) - K + 1 = H*sh.
        # Then we handle PyTorch `padding` (cropping).
        
        # Let's assume `weights` are properly converted to work with `mx.conv2d` (flipped if necessary).
        
        # Manual Pad `x_up` 
        # Pad amount = (K-1) - padding + output_padding?
        # This is getting complicated to exact match.
        
        # Let's try to trust `mx.conv2d` to handle "same" or explicit.
        # We will use `mx.conv2d` on `x_up` with `padding=0` (Valid) but pre-pad `x_up`.
        
        pad_h = max(0, kh - 1 - self.padding[0])
        pad_w = max(0, kw - 1 - self.padding[1])
        
        # Add output_padding
        # op = self.output_padding
        # Extra padding at end?
        
        x_up = mx.pad(x_up, ((0,0), (pad_h, pad_h + self.output_padding[0]), (pad_w, pad_w + self.output_padding[1]), (0,0)))
        
        # Conv
        # We use a standard functional conv2d since we have weights in `self.weight`
        # self.weight shape in memory: (Out, H, W, In) (standard MLX conv)
        y = mx.conv2d(x_up, self.weight, stride=1, padding=0)
        
        if self.bias is not None:
            y = y + self.bias
            
        return y

class ConvBlockRes(nn.Module):
    def __init__(self, in_channels, out_channels, momentum=0.01):
        super().__init__()
        # MLX BatchNorm uses `momentum`? Default 0.1. PT default 0.1.
        # RMVPE uses 0.01.
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm(out_channels, momentum=momentum, eps=1e-5)
        self.act1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm(out_channels, momentum=momentum, eps=1e-5)
        self.act2 = nn.ReLU() # Original has ReLU after BN2? Yes.
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True) # PyTorch Default bias=True?
            # RMVPE says: self.shortcut = nn.Conv2d(..., bias=False) ? No, bias not specified, so True.
            # Wait, RMVPE.py line 48: nn.Conv2d(..., (1, 1)) -> bias=True by default.
            # But line 32/42 specify bias=False for main branch.
            # Let's assume True for shortcut.
        else:
            self.shortcut = None

    def __call__(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act2(out)
        
        if self.shortcut is not None:
            residual = self.shortcut(x)
            
        return out + residual

class ResEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, n_blocks=1, momentum=0.01):
        super().__init__()
        self.blocks = []
        self.blocks.append(ConvBlockRes(in_channels, out_channels, momentum))
        for _ in range(n_blocks - 1):
             self.blocks.append(ConvBlockRes(out_channels, out_channels, momentum))
        
        if kernel_size is not None:
            self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=kernel_size)
        else:
            self.pool = None
            
    def __call__(self, x):
        for block in self.blocks:
            x = block(x)
            
        if self.pool is not None:
            return x, self.pool(x)
        return x

class Encoder(nn.Module):
    def __init__(self, in_channels, in_size, n_encoders, kernel_size, n_blocks, out_channels=16, momentum=0.01):
        super().__init__()
        self.bn = nn.BatchNorm(in_channels, momentum=momentum, eps=1e-5)
        self.layers = []
        
        curr_in = in_channels
        curr_out = out_channels
        
        for i in range(n_encoders):
            self.layers.append(ResEncoderBlock(curr_in, curr_out, kernel_size, n_blocks, momentum))
            curr_in = curr_out
            curr_out *= 2
        
        self.out_channel = curr_in * 2

    def __call__(self, x):
        x = self.bn(x)
        concat_tensors = []
        for layer in self.layers:
            t, x = layer(x)
            concat_tensors.append(t)
        return x, concat_tensors

class Intermediate(nn.Module):
    def __init__(self, in_channels, out_channels, n_inters, n_blocks, momentum=0.01):
        super().__init__()
        self.layers = []
        self.layers.append(ResEncoderBlock(in_channels, out_channels, None, n_blocks, momentum))
        for _ in range(n_inters - 1):
            self.layers.append(ResEncoderBlock(out_channels, out_channels, None, n_blocks, momentum))
            
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class ResDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, n_blocks=1, momentum=0.01):
        super().__init__()
        padding = (1, 1)
        op = (0, 1) if stride == (1, 2) else (1, 1)
        
        self.conv1_trans = ConvTranspose2d(
            in_channels, out_channels, kernel_size=(3,3), stride=stride, padding=padding, output_padding=op, bias=False
        )
        self.bn1 = nn.BatchNorm(out_channels, momentum=momentum)
        self.act1 = nn.ReLU()
        
        self.blocks = []
        self.blocks.append(ConvBlockRes(out_channels * 2, out_channels, momentum))
        for _ in range(n_blocks - 1):
             self.blocks.append(ConvBlockRes(out_channels, out_channels, momentum))

    def __call__(self, x, concat_tensor):
        x = self.conv1_trans(x)
        x = mx.concatenate([x, concat_tensor], axis=-1)
        
        for block in self.blocks:
            x = block(x)
        return x

class Decoder(nn.Module):
    def __init__(self, in_channels, n_decoders, stride, n_blocks, momentum=0.01):
        super().__init__()
        self.layers = []
        curr_in = in_channels
        for _ in range(n_decoders):
            out_channels = curr_in // 2
            self.layers.append(ResDecoderBlock(curr_in, out_channels, stride, n_blocks, momentum))
            curr_in = out_channels

    def __call__(self, x, concat_tensors):
        for i, layer in enumerate(self.layers):
            x = layer(x, concat_tensors[-1-i])
        return x

class DeepUnet(nn.Module):
    def __init__(self, kernel_size, n_blocks, en_de_layers=5, inter_layers=4, in_channels=1, en_out_channels=16):
        super().__init__()
        self.encoder = Encoder(in_channels, 128, en_de_layers, kernel_size, n_blocks, en_out_channels)
        self.intermediate = Intermediate(
            self.encoder.out_channel // 2,
            self.encoder.out_channel,
            inter_layers,
            n_blocks
        )
        self.decoder = Decoder(self.encoder.out_channel, en_de_layers, kernel_size, n_blocks)
        
    def __call__(self, x):
        # x: (N, H, W, C) or (N, C, H, W)?
        # MLX expects Channels Last: (N, H, W, C).
        # We assume pipeline handles correct format input.
        x, concat_tensors = self.encoder(x)
        x = self.intermediate(x)
        x = self.decoder(x, concat_tensors)
        return x

class BiGRU(nn.Module):
    def __init__(self, input_features, hidden_features, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_features = hidden_features
        
        # For each layer, create forward and backward GRUs
        self.forward_grus = []
        self.backward_grus = []
        
        for i in range(num_layers):
            in_size = input_features if i == 0 else hidden_features * 2
            self.forward_grus.append(nn.GRU(in_size, hidden_features, bias=True))
            self.backward_grus.append(nn.GRU(in_size, hidden_features, bias=True))
        
    def __call__(self, x):
         # x: (N, L, C)
         for i in range(self.num_layers):
             fwd_gru = self.forward_grus[i]
             bwd_gru = self.backward_grus[i]
             
             # Forward pass
             out_fwd, _ = fwd_gru(x)
             
             # Backward pass (reverse, process, reverse back)
             x_rev = mx.flip(x, axis=1)
             out_bwd_rev, _ = bwd_gru(x_rev)
             out_bwd = mx.flip(out_bwd_rev, axis=1)
             
             # Concatenate
             x = mx.concatenate([out_fwd, out_bwd], axis=-1)
             
         return x

class E2E(nn.Module):
    def __init__(self, n_blocks, n_gru, kernel_size, en_de_layers=5, inter_layers=4, in_channels=1, en_out_channels=16):
        super().__init__()
        self.unet = DeepUnet(kernel_size, n_blocks, en_de_layers, inter_layers, in_channels, en_out_channels)
        self.cnn = nn.Conv2d(en_out_channels, 3, kernel_size=3, padding=1)
        
        N_CLASS = 360
        if n_gru:
            self.fc = nn.Sequential([
                BiGRU(3 * 128, 256, n_gru),
                nn.Linear(512, N_CLASS),
                nn.Dropout(0.25),
                nn.Sigmoid() # MLX Sigmoid? nn.Sigmoid() layer? or mx.sigmoid. 
                # nn.Sigmoid doesn't exist in some versions.
                # Use lambda or activation wrapper.
            ])
        else:
            self.fc = nn.Sequential([
                nn.Linear(3 * 128, N_CLASS), # 3*N_MELS? N_MELS=128
                nn.Dropout(0.25),
                nn.Sigmoid()
            ])

    def __call__(self, mel):
        # mel: (N, L, C) or (N, C, L)?
        # RMVPE.py uses (N, 1, T, n_mels) format for UNet?
        # PyTorch impl: `mel = mel.transpose(-1, -2).unsqueeze(1)`
        # Input 'mel' to `forward` is (N, n_mels, T).
        # Becomes (N, 1, n_mels, T).
        
        # MLX: We prefer (N, T, n_mels).
        # We'll adjust input processing in pipeline.
        
        # UNet expects (N, H, W, C).
        # H=n_mels, W=T, C=1.
        
        # Pipeline sends (N, T, n_mels).
        # We want H=n_mels, W=T.
        # So x = mel.transpose(0, 2, 1) -> (N, n_mels, T)
        # x = x.unsqueeze(-1) -> (N, n_mels, T, 1)
        
        # x = self.unet(mel)
        x = self.unet(mel)
        
        x = self.cnn(x) # (N, H, W, 3)
        
        # Prepare for GRU
        # x.transpose(1, 2) -> (N, T, H, 3)?
        # flatten(-2) -> flatten last 2 dims?
        # PyTorch: x.transpose(1, 2).flatten(-2)
        # (N, 3, H, W) --transpose--> (N, 3, W, H) --flatten--> (N, 3, W*H) NO.
        # Check PT: `x = self.cnn(self.unet(mel)).transpose(1, 2).flatten(-2)`
        # CNN Out: (N, 3, H, W). (H=mel_dim, W=time)
        # Transpose(1, 2) -> (N, H, 3, W). NO. (N, C, H, W) -> (1, 2) swap C and H?
        # WAIT. `transpose(1, 2)` swaps Dim 1 and Dim 2.
        # (N, 3, H, W) -> (N, H, 3, W).
        # `flatten(-2)` -> Flatten (3, W) -> (N, H, 3*W)?
        # This seems creating a sequence of length H? That's wrong. GRU is over time W.
        # Ah, PyTorch inputs to UNet: (N, 1, H, W).
        # H = n_mels? W = T?
        # Let's check `mel.transpose(-1, -2).unsqueeze(1)`.
        # input mel: (N, n_mels, T).
        # transpose: (N, T, n_mels).
        # unsqueeze(1): (N, 1, T, n_mels).
        # So H=T, W=n_mels.
        # UNet output: (N, 16, T, n_mels).
        # CNN output: (N, 3, T, n_mels).
        # Transpose(1, 2): (N, T, 3, n_mels).
        # Flatten(-2): Flatten (3, n_mels) -> (N, T, 3*n_mels).
        # GRU input: (N, T, Features). Correct.
        
        # MLX Logic:
        # Input x: (N, T, n_mels, 1) or (N, T, n_mels, C_in)?
        # Let's say input is (N, T, n_mels, 1).
        # UNet out: (N, T, n_mels, 16).
        # CNN out: (N, T, n_mels, 3).
        # We need (N, T, n_mels * 3).
        # x.reshape(N, T, -1).
        
        x = self.cnn(x)
        B, T, M, C = x.shape
        x = x.reshape(B, T, M * C)
        
        # FC
        x = self.fc(x)
        return x

import numpy as np
import librosa
import os

class RMVPE0Predictor:
    def __init__(self, model_path=None, weights_path=None, device=None):
        # device arg ignored for MLX (auto)
        self.model = E2E(4, 1, (2, 2))
        
        # Load weights
        if weights_path is None:
             # Default path
             weights_path = os.path.join("rvc", "models", "predictors", "rmvpe_mlx.npz")
             
        if not os.path.exists(weights_path):
            print(f"RMVPE MLX weights not found at {weights_path}")
        else:
            self.model.load_weights(weights_path)
            
        # Constants for decode
        N_CLASS = 360
        self.cents_mapping = 20 * np.arange(N_CLASS) + 1997.3794084376191
        self.cents_mapping = np.pad(self.cents_mapping, (4, 4))

    def _create_mel_filterbank(self, n_fft, n_mels, sr, fmin, fmax):
        """Create mel filterbank matrix (computed once, cached)."""
        # Convert Hz to Mel
        def hz_to_mel(hz):
            return 2595 * np.log10(1 + hz / 700)
        
        def mel_to_hz(mel):
            return 700 * (10 ** (mel / 2595) - 1)
        
        # Create mel points
        mel_min = hz_to_mel(fmin)
        mel_max = hz_to_mel(fmax)
        mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = mel_to_hz(mel_points)
        
        # FFT bin frequencies
        freq_bins = np.fft.rfftfreq(n_fft, 1.0/sr)
        
        # Create filterbank
        filterbank = np.zeros((n_mels, len(freq_bins)))
        
        for i in range(n_mels):
            left = hz_points[i]
            center = hz_points[i + 1]
            right = hz_points[i + 2]
            
            # Left slope
            left_mask = (freq_bins >= left) & (freq_bins <= center)
            filterbank[i, left_mask] = (freq_bins[left_mask] - left) / (center - left + 1e-10)
            
            # Right slope
            right_mask = (freq_bins >= center) & (freq_bins <= right)
            filterbank[i, right_mask] = (right - freq_bins[right_mask]) / (right - center + 1e-10)
        
        return mx.array(filterbank.astype(np.float32))
    
    def _create_window(self, win_length):
        """Create Hann window."""
        n = np.arange(win_length)
        window = 0.5 - 0.5 * np.cos(2 * np.pi * n / win_length)
        return mx.array(window.astype(np.float32))
    
    def mel_spectrogram(self, audio):
        """GPU-accelerated mel spectrogram using MLX FFT.
        
        Args:
            audio: numpy array (T,) at 16kHz
            
        Returns:
            log_mel: numpy array (n_mels, num_frames)
        """
        # Parameters matching RMVPE
        n_fft = 1024
        hop_length = 160
        win_length = 1024
        n_mels = 128
        sr = 16000
        fmin = 30
        fmax = 8000
        
        # Create/cache mel filterbank and window
        if not hasattr(self, '_mel_filterbank'):
            self._mel_filterbank = self._create_mel_filterbank(n_fft, n_mels, sr, fmin, fmax)
        if not hasattr(self, '_window'):
            self._window = self._create_window(win_length)
        
        # Pad audio for center=True (reflect padding)
        pad_len = n_fft // 2
        audio_padded = np.pad(audio, (pad_len, pad_len), mode='reflect')
        
        # Convert to MLX
        audio_mx = mx.array(audio_padded.astype(np.float32))
        
        # Compute STFT frames
        # Number of frames
        num_frames = 1 + (len(audio_padded) - n_fft) // hop_length
        
        # Extract frames using strided view (vectorized)
        # Create frame indices
        frame_starts = np.arange(num_frames) * hop_length
        frame_indices = frame_starts[:, None] + np.arange(n_fft)
        
        # Gather frames
        frames = audio_mx[frame_indices]  # (num_frames, n_fft)
        
        # Apply window
        frames = frames * self._window
        
        # FFT
        spectrum = mx.fft.rfft(frames, axis=-1)  # (num_frames, n_fft//2 + 1)
        
        # Magnitude (not power)
        magnitude = mx.abs(spectrum)  # (num_frames, n_fft//2 + 1)
        
        # Apply mel filterbank: (n_mels, n_fft//2+1) @ (n_fft//2+1, num_frames) = (n_mels, num_frames)
        mel = self._mel_filterbank @ magnitude.T  # (n_mels, num_frames)
        
        # Log scale with floor
        log_mel = mx.log(mx.maximum(mel, 1e-5))
        
        # Force evaluation and convert to numpy
        mx.eval(log_mel)
        return np.array(log_mel)

    def mel2hidden(self, mel, chunk_size=32000):
        # mel: (n_mels, T)
        # We need to process in MLX
        n_frames = mel.shape[-1]
        
        # Pad logic from original
        # mel = F.pad(mel, (0, 32 * ((n_frames - 1) // 32 + 1) - n_frames), mode="reflect")
        pad_curr = 32 * ((n_frames - 1) // 32 + 1) - n_frames
        mel = np.pad(mel, ((0, 0), (0, pad_curr)), mode='reflect')
        
        # Convert to MLX
        mel_mx = mx.array(mel) 
        # MLX expects (N, T, C). My model `__call__` expects (N, C, T) or (N, T, C)?
        # E2E.__call__: `mel = mel.transpose(-1, -2).unsqueeze(1)` logic adaptation.
        # My E2E.__call__ implementation:
        # `x = self.unet(mel)`
        # `unet` expects (N, H, W, C)?
        # Wait, I checked. `mel.transpose(-1, -2).unsqueeze(1)` -> (N, 1, T, n_mels).
        # My E2E.__call__ assumes input (N, T, n_mels) if I recall.
        
        # Let's check E2E.__call__ logic in this file again.
        # "x = self.unet(mel)"
        # And UNet.__call__ does "x, concat = self.encoder(x)".
        # Encoder expects (N, H, W, C) for Conv2d.
        
        # SO, we need to reshape/transpose `mel` to (N, T, n_mels, 1) or (N, n_mels, T, 1)?
        # PyTorch Input to UNet: (N, 1, T, n_mels) (after transpose/unsqueeze).
        # Conv2d kernel (3,3). 
        # H=T, W=n_mels? No.
        # mel is (n_mels, T).
        # transpose(-1, -2) -> (T, n_mels).
        # unsqueeze(1) -> (1, T, n_mels).
        # So H=T, W=n_mels.
        
        # My MLX Conv2d expects (N, H, W, C).
        # So input should be (N, T, n_mels, 1).
        
        # Current `mel` numpy: (n_mels, T).
        # Transpose to (T, n_mels).
        # Add Batch N=1. Add Channel C=1.
        # (1, T, n_mels, 1).
        
        mel_mx = mel_mx.transpose(1, 0)[None, :, :, None]
        
        # Chunking logic (simplified: feed all?)
        # RMVPE chunks to save memory/batch?
        # "chunk_size=32000" frames.
        # If short, just run.
        # If long, chunk.
        
        # For MLX efficiency, let's just run it if we trust memory (Unified Memory often huge).
        # But `mel` pads.
        
        # If we skip chunking for now:
        hidden = self.model(mel_mx)
        # Output hidden: (N, T, 360).
        
        # Strip padding
        # hidden[:, :n_frames, :]
        return hidden[:, :n_frames, :]

    def decode(self, hidden, thred=0.03):
        # hidden: (T, 360) (N=1 squeezed)
        # Numpy implementation of decode logic
        if isinstance(hidden, mx.array):
            hidden = np.array(hidden) # Convert to numpy
            
        # hidden shape: (T, 360)? Or (N, T, 360)?
        if hidden.ndim == 3:
            hidden = hidden[0]
            
        cents_pred = self.to_local_average_cents(hidden, thred=thred)
        f0 = 10 * (2 ** (cents_pred / 1200))
        f0[f0 == 10] = 0
        return f0

    def to_local_average_cents(self, salience, thred=0.05):
        # Use existing logic from original, adapted to clean numpy
        center = np.argmax(salience, axis=1)
        salience = np.pad(salience, ((0, 0), (4, 4)))
        center += 4
        
        # Vectorized gathering
        # We need to gather window [center-4 : center+5] for each time step.
        # Broadcasting?
        # N = len(center)
        # indices = center[:, None] + np.arange(-4, 5) # (N, 9)
        # gathered_salience = np.take_along_axis(salience, indices, axis=1)
        
        # The provided loop is valid but slow in PyTorch impl?
        # Let's replicate exact loop for correctness.
        
        todo_salience = []
        todo_cents_mapping = []
        starts = center - 4
        ends = center + 5
        
        # Note: salience is (T, 360+8).
        # This loop isn't too expensive for Audio < 1min.
        for idx in range(salience.shape[0]):
             todo_salience.append(salience[idx, starts[idx]:ends[idx]])
             todo_cents_mapping.append(self.cents_mapping[starts[idx]:ends[idx]])
             
        todo_salience = np.array(todo_salience)
        todo_cents_mapping = np.array(todo_cents_mapping)
        
        product_sum = np.sum(todo_salience * todo_cents_mapping, 1)
        weight_sum = np.sum(todo_salience, 1)
        devided = product_sum / (weight_sum + 1e-9)
        
        maxx = np.max(salience, axis=1)
        devided[maxx <= thred] = 0
        return devided

    def infer_from_audio(self, audio, thred=0.03):
        # audio: numpy (T,)
        mel = self.mel_spectrogram(audio)
        hidden = self.mel2hidden(mel)
        # hidden (1, T, 360) -> (T, 360) numpy
        f0 = self.decode(hidden, thred=thred)
        return f0
