
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
        # x: (N, H, W, C_in)
        # MLX has mx.conv_transpose2d available in version 0.10.0+
        # Note: MLX doesn't support output_padding, so we'll add it manually

        # MLX conv_transpose2d signature:
        # mx.conv_transpose2d(input, weight, stride=1, padding=0, dilation=1, groups=1)
        # input: (N, H, W, C_in)
        # weight: (C_out, kernel_h, kernel_w, C_in)  <- This is MLX format
        # Returns: (N, H_out, W_out, C_out)

        y = mx.conv_transpose2d(
            x,
            self.weight,
            stride=self.stride,
            padding=self.padding
        )

        # Manually add output_padding if needed
        # output_padding adds extra rows/columns to the output
        if self.output_padding != (0, 0) and self.output_padding != 0:
            op_h, op_w = self.output_padding if isinstance(self.output_padding, tuple) else (self.output_padding, self.output_padding)
            if op_h > 0 or op_w > 0:
                # Add padding to bottom and right
                y = mx.pad(y, ((0, 0), (0, op_h), (0, op_w), (0, 0)))

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

        # Match spatial dimensions to concat_tensor
        # This handles cases where output_padding doesn't perfectly align dimensions
        target_shape = concat_tensor.shape
        current_shape = x.shape

        if current_shape[1] != target_shape[1] or current_shape[2] != target_shape[2]:
            # Pad to match target spatial dimensions
            pad_h = target_shape[1] - current_shape[1]
            pad_w = target_shape[2] - current_shape[2]

            if pad_h > 0 or pad_w > 0:
                # Pad on the right/bottom
                x = mx.pad(x, ((0, 0), (0, pad_h), (0, pad_w), (0, 0)))
            elif pad_h < 0 or pad_w < 0:
                # Crop if decoder output is larger (shouldn't happen but handle it)
                x = x[:, :target_shape[1], :target_shape[2], :]

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

        # Create containers for forward and backward GRUs
        class GRUContainer(nn.Module):
            def __init__(self):
                super().__init__()

        self.forward_grus = GRUContainer()
        self.backward_grus = GRUContainer()

        # Add GRUs as numbered attributes
        for i in range(num_layers):
            in_size = input_features if i == 0 else hidden_features * 2
            setattr(self.forward_grus, str(i), nn.GRU(in_size, hidden_features, bias=True))
            setattr(self.backward_grus, str(i), nn.GRU(in_size, hidden_features, bias=True))

    def __call__(self, x):
         # x: (N, L, C)
         for i in range(self.num_layers):
             fwd_gru = getattr(self.forward_grus, str(i))
             bwd_gru = getattr(self.backward_grus, str(i))

             # Forward pass (MLX GRU returns just output, not tuple)
             out_fwd = fwd_gru(x)

             # Backward pass (reverse along time axis, process, reverse back)
             x_rev = x[:, ::-1, :]  # Reverse time dimension
             out_bwd_rev = bwd_gru(x_rev)
             out_bwd = out_bwd_rev[:, ::-1, :]  # Reverse back

             # Concatenate
             x = mx.concatenate([out_fwd, out_bwd], axis=-1)

         return x

class E2E(nn.Module):
    def __init__(self, n_blocks, n_gru, kernel_size, en_de_layers=5, inter_layers=4, in_channels=1, en_out_channels=16):
        super().__init__()
        self.unet = DeepUnet(kernel_size, n_blocks, en_de_layers, inter_layers, in_channels, en_out_channels)
        self.cnn = nn.Conv2d(en_out_channels, 3, kernel_size=3, padding=1)
        
        N_CLASS = 360

        # FCLayers with manual weight loading workaround
        # Weights are structured as fc.layers.0.0.* and fc.layers.0.1.*
        # But MLX can't traverse nested containers, so we'll load manually
        class FCLayers(nn.Module):
            def __init__(self, bigru, linear, has_gru):
                super().__init__()
                self.has_gru = has_gru
                self.bigru = bigru
                self.linear = linear

            def __call__(self, x):
                if self.has_gru:
                    x = self.bigru(x)
                x = self.linear(x)
                return x

        if n_gru:
            bigru = BiGRU(3 * 128, 256, n_gru)
            linear = nn.Linear(512, N_CLASS)
            self.fc = FCLayers(bigru, linear, has_gru=True)
        else:
            linear = nn.Linear(3 * 128, N_CLASS)
            self.fc = FCLayers(None, linear, has_gru=False)

        # Dropout and Sigmoid are applied without weights
        self.dropout = nn.Dropout(0.25)
        self.sigmoid = nn.Sigmoid()

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

        # CNN already called above - just reshape
        B, T, M, C = x.shape
        x = x.reshape(B, T, M * C)

        # FC layers (BiGRU + Linear)
        x = self.fc(x)

        # Apply dropout and sigmoid
        x = self.dropout(x)
        x = self.sigmoid(x)
        return x

import numpy as np
import librosa
import os

class RMVPE0Predictor:
    def __init__(self, model_path=None, weights_path=None, device=None):
        # device arg ignored for MLX (auto)
        self.model = E2E(4, 1, (2, 2))
        
        # Load weights
        # Load weights
        if weights_path is None:
             # Default path - Prioritize rvc_mlx path
             weights_path = os.path.join("rvc_mlx", "models", "predictors", "rmvpe_mlx.npz")
             
             if not os.path.exists(weights_path):
                 # Try relative path from this file
                 rel_path = os.path.join(os.path.dirname(__file__), "../../../models/predictors/rmvpe_mlx.npz")
                 if os.path.exists(rel_path):
                     weights_path = rel_path
                 else:
                     # Fallback to rvc legacy
                     weights_path = os.path.join("rvc", "models", "predictors", "rmvpe_mlx.npz")

        if not os.path.exists(weights_path):
            print(f"RMVPE MLX weights not found at {weights_path}")
        else:
            # Load weights with strict=False to handle custom structure
            try:
                self.model.load_weights(weights_path, strict=False)
            except Exception as e:
                print(f"Warning: Weight loading with strict=False failed: {e}")
                print("Attempting manual weight loading...")
                self._manual_load_weights(weights_path)

            mx.eval(self.model.parameters())
            
        # Constants for decode
        N_CLASS = 360
        self.cents_mapping = 20 * np.arange(N_CLASS) + 1997.3794084376191
        self.cents_mapping = np.pad(self.cents_mapping, (4, 4))

    def _manual_load_weights(self, weights_path):
        """Manually load FC layer weights that don't match the model structure."""
        import numpy as np

        # Load all weights
        weights_dict = np.load(weights_path)

        # Extract FC layer weights (fc.layers.0.0.* and fc.layers.0.1.*)
        fc_weights = {}
        for key in weights_dict.keys():
            if key.startswith('fc.layers.0.'):
                fc_weights[key] = weights_dict[key]

        # Map to model structure
        # fc.layers.0.0.forward_grus.0.* -> fc.bigru.forward_grus.0.*
        # fc.layers.0.1.* -> fc.linear.*

        if hasattr(self.model.fc, 'bigru') and self.model.fc.bigru is not None:
            # Load BiGRU weights
            bigru = self.model.fc.bigru

            # Forward GRUs
            for key, value in fc_weights.items():
                if 'forward_grus' in key:
                    # Extract layer index and parameter name
                    # fc.layers.0.0.forward_grus.0.Wh -> forward_grus.0.Wh
                    parts = key.split('.')
                    layer_idx = parts[4]  # '0'
                    param_name = parts[5]  # 'Wh', 'Wx', 'b', 'bhn'

                    gru_layer = getattr(bigru.forward_grus, layer_idx)
                    if param_name == 'Wh':
                        gru_layer.Wh = mx.array(value)
                    elif param_name == 'Wx':
                        gru_layer.Wx = mx.array(value)
                    elif param_name == 'b':
                        gru_layer.b = mx.array(value)
                    elif param_name == 'bhn':
                        gru_layer.bhn = mx.array(value)

                elif 'backward_grus' in key:
                    parts = key.split('.')
                    layer_idx = parts[4]
                    param_name = parts[5]

                    gru_layer = getattr(bigru.backward_grus, layer_idx)
                    if param_name == 'Wh':
                        gru_layer.Wh = mx.array(value)
                    elif param_name == 'Wx':
                        gru_layer.Wx = mx.array(value)
                    elif param_name == 'b':
                        gru_layer.b = mx.array(value)
                    elif param_name == 'bhn':
                        gru_layer.bhn = mx.array(value)

        # Load Linear layer weights
        # fc.layers.0.1.weight -> fc.linear.weight
        # fc.layers.0.1.bias -> fc.linear.bias
        if 'fc.layers.0.1.weight' in fc_weights:
            self.model.fc.linear.weight = mx.array(fc_weights['fc.layers.0.1.weight'])
        if 'fc.layers.0.1.bias' in fc_weights:
            self.model.fc.linear.bias = mx.array(fc_weights['fc.layers.0.1.bias'])

        print(f"✅ Manually loaded {len(fc_weights)} FC layer weights")

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

        # Gather frames (convert indices to MLX array for indexing)
        frame_indices_mx = mx.array(frame_indices)
        frames = audio_mx[frame_indices_mx]  # (num_frames, n_fft)
        
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
        pad_frames = mel_mx.shape[1]

        # Process: chunked or single-pass
        if chunk_size is None or chunk_size <= 0 or pad_frames <= chunk_size:
            # Single pass for short audio
            hidden = self.model(mel_mx)
        else:
            # Chunked processing for better GPU cache utilization
            output_chunks = []
            for start in range(0, pad_frames, chunk_size):
                end = min(start + chunk_size, pad_frames)
                mel_chunk = mel_mx[:, start:end, :, :]

                out_chunk = self.model(mel_chunk)
                mx.eval(out_chunk)  # Force evaluation (MLX lazy evaluation)
                output_chunks.append(out_chunk)

            # Concatenate along time dimension
            hidden = mx.concatenate(output_chunks, axis=1)
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
