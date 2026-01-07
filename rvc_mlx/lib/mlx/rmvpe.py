
import mlx.core as mx
import mlx.nn as nn
import math
import librosa
import numpy as np
import os

class ModuleList(nn.Module):
    def __init__(self, modules=None):
        super().__init__()
        self._length = 0
        if modules is not None:
            for m in modules:
                self.append(m)

    def append(self, module):
        setattr(self, str(self._length), module)
        self._length += 1

    def __getitem__(self, idx):
        if idx < 0: idx += self._length
        if idx < 0 or idx >= self._length: raise IndexError()
        return getattr(self, str(idx))

    def __len__(self):
        return self._length

    def __iter__(self):
        for i in range(self._length):
            yield self[i]

class ConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, output_padding=0, bias=False):
        super().__init__()
        self.stride = stride
        self.output_padding = output_padding
        self.kernel_size = kernel_size
        self.padding = padding
        
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
        y = mx.conv_transpose2d(
            x,
            self.weight,
            stride=self.stride,
            padding=self.padding
        )

        if self.output_padding != (0, 0) and self.output_padding != 0:
            op_h, op_w = self.output_padding if isinstance(self.output_padding, tuple) else (self.output_padding, self.output_padding)
            if op_h > 0 or op_w > 0:
                y = mx.pad(y, ((0, 0), (0, op_h), (0, op_w), (0, 0)))

        if self.bias is not None:
            y = y + self.bias
        return y

class ConvBlockRes(nn.Module):
    def __init__(self, in_channels, out_channels, momentum=0.01):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm(out_channels, momentum=momentum, eps=1e-5)
        self.act1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm(out_channels, momentum=momentum, eps=1e-5)
        self.act2 = nn.ReLU()
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
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
        self.blocks = ModuleList()
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
        self.layers = ModuleList()
        curr_in = in_channels
        curr_out = out_channels
        for i in range(n_encoders):
            self.layers.append(ResEncoderBlock(curr_in, curr_out, kernel_size, n_blocks, momentum))
            curr_in = curr_out
            curr_out *= 2
        self.out_channel = curr_in

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
        self.layers = ModuleList()
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
        self.conv1_trans = ConvTranspose2d(in_channels, out_channels, kernel_size=(3,3), stride=stride, padding=padding, output_padding=op, bias=False)
        self.bn1 = nn.BatchNorm(out_channels, momentum=momentum, eps=1e-5)
        self.act1 = nn.ReLU()
        self.blocks = ModuleList()
        self.blocks.append(ConvBlockRes(out_channels * 2, out_channels, momentum))
        for _ in range(n_blocks - 1):
             self.blocks.append(ConvBlockRes(out_channels, out_channels, momentum))

    def __call__(self, x, concat_tensor):
        x = self.conv1_trans(x)
        x = self.bn1(x)
        x = self.act1(x)
        
        target_shape = concat_tensor.shape
        current_shape = x.shape
        if current_shape[1] != target_shape[1] or current_shape[2] != target_shape[2]:
            pad_h = max(0, target_shape[1] - current_shape[1])
            pad_w = max(0, target_shape[2] - current_shape[2])
            if pad_h > 0 or pad_w > 0:
                x = mx.pad(x, ((0, 0), (0, pad_h), (0, pad_w), (0, 0)))
            x = x[:, :target_shape[1], :target_shape[2], :]

        x = mx.concatenate([x, concat_tensor], axis=-1)
        for block in self.blocks:
            x = block(x)
        return x

class Decoder(nn.Module):
    def __init__(self, in_channels, n_decoders, stride, n_blocks, momentum=0.01):
        super().__init__()
        self.layers = ModuleList()
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
        self.intermediate = Intermediate(self.encoder.out_channel, self.encoder.out_channel * 2, inter_layers, n_blocks)
        self.decoder = Decoder(self.encoder.out_channel * 2, en_de_layers, kernel_size, n_blocks)
        
    def __call__(self, x):
        x, concat_tensors = self.encoder(x)
        x = self.intermediate(x)
        x = self.decoder(x, concat_tensors)
        return x

# Import custom PyTorch-compatible GRU
from rvc_mlx.lib.mlx.pytorch_gru import BiGRU

class E2E(nn.Module):
    def __init__(self, n_blocks, n_gru, kernel_size, en_de_layers=5, inter_layers=4, in_channels=1, en_out_channels=16):
        super().__init__()
        self.unet = DeepUnet(kernel_size, n_blocks, en_de_layers, inter_layers, in_channels, en_out_channels)
        self.cnn = nn.Conv2d(en_out_channels, 3, kernel_size=3, padding=1)
        
        N_CLASS = 360
        class FCLayers(nn.Module):
            def __init__(self, bigru, linear, has_gru):
                super().__init__()
                self.has_gru = has_gru
                self.bigru = bigru
                self.linear = linear
            def __call__(self, x):
                if self.has_gru: x = self.bigru(x)
                x = self.linear(x)
                return x

        if n_gru:
            bigru = BiGRU(3 * 128, 256, n_gru, bias=True)
            linear = nn.Linear(512, N_CLASS)
            self.fc = FCLayers(bigru, linear, has_gru=True)
        else:
            linear = nn.Linear(3 * 128, N_CLASS)
            self.fc = FCLayers(None, linear, has_gru=False)

        self.dropout = nn.Dropout(0.25)
        self.sigmoid = nn.Sigmoid()

    def __call__(self, x):
        x = self.unet(x)
        print(f"DEBUG: RMVPE UNet output stats: min {x.min().item():.6f}, max {x.max().item():.6f}, mean {x.mean().item():.6f}")

        x = self.cnn(x)
        print(f"DEBUG: RMVPE CNN output stats: min {x.min().item():.6f}, max {x.max().item():.6f}, mean {x.mean().item():.6f}")

        x = x.transpose(0, 1, 3, 2)
        B, T, C, M = x.shape
        x = x.reshape(B, T, C * M)
        x = self.fc(x)
        x = self.dropout(x)
        x = self.sigmoid(x)
        return x

class RMVPE0Predictor:
    def __init__(self, weights_path=None):
        if weights_path is None:
            weights_path = os.path.join("rvc_mlx", "models", "predictors", "rmvpe_mlx.npz")

        self.model = E2E(4, 1, (2, 2))
        if os.path.exists(weights_path):
            # Use MLX's built-in load_weights for most layers
            loaded_params = self.model.load_weights(weights_path, strict=False)

            # Manually load BiGRU weights (nested modules don't auto-load)
            weights_dict = dict(np.load(weights_path))

            if hasattr(self.model.fc, 'bigru'):
                # Manually load GRU weights by creating new GRU instances
                # This ensures proper initialization in MLX's computation graph
                from rvc_mlx.lib.mlx.pytorch_gru import PyTorchGRU

                # Create new forward GRU with weights
                fwd_gru = PyTorchGRU(384, 256, bias=True)
                fwd_gru.weight_ih = mx.array(weights_dict['fc.bigru.forward_grus.0.weight_ih'])
                fwd_gru.weight_hh = mx.array(weights_dict['fc.bigru.forward_grus.0.weight_hh'])
                fwd_gru.bias_ih = mx.array(weights_dict['fc.bigru.forward_grus.0.bias_ih'])
                fwd_gru.bias_hh = mx.array(weights_dict['fc.bigru.forward_grus.0.bias_hh'])

                # Create new backward GRU with weights
                bwd_gru = PyTorchGRU(384, 256, bias=True)
                bwd_gru.weight_ih = mx.array(weights_dict['fc.bigru.backward_grus.0.weight_ih'])
                bwd_gru.weight_hh = mx.array(weights_dict['fc.bigru.backward_grus.0.weight_hh'])
                bwd_gru.bias_ih = mx.array(weights_dict['fc.bigru.backward_grus.0.bias_ih'])
                bwd_gru.bias_hh = mx.array(weights_dict['fc.bigru.backward_grus.0.bias_hh'])

                # Replace the GRU modules directly
                setattr(self.model.fc.bigru.forward_grus, '0', fwd_gru)
                setattr(self.model.fc.bigru.backward_grus, '0', bwd_gru)

            self.model.eval()
        
        N_CLASS = 360
        self.cents_mapping = 20 * np.arange(N_CLASS) + 1997.3794084376191
        self.cents_mapping = np.pad(self.cents_mapping, (4, 4), mode='constant')
        
        self._mel_filterbank = librosa.filters.mel(sr=16000, n_fft=1024, n_mels=128, fmin=30, fmax=8000, htk=True)

    def mel_spectrogram(self, audio):
        # audio: (T,)
        if isinstance(audio, mx.array):
            audio = np.array(audio)
            
        # Pad audio for center=True
        pad_len = 512
        audio_padded = np.pad(audio, (pad_len, pad_len), mode='reflect')
        
        # Use librosa for STFT parity
        stft = librosa.stft(audio_padded, n_fft=1024, hop_length=160, win_length=1024, window='hann', center=False)
        magnitude = np.abs(stft)
        
        mel = self._mel_filterbank @ magnitude
        log_mel = np.log(np.maximum(mel, 1e-5))
        
        print(f"DEBUG: Mel Spectrogram Stats: min {log_mel.min():.6f}, max {log_mel.max():.6f}, mean {log_mel.mean():.6f}, shape {log_mel.shape}")
        
        # Log first frame, first 10 bins
        mel_slice = log_mel[:10, 0] # (K, T) -> slice first 10 freqs of 0th frame?
        # check shape: (128, T)
        # So first frame is [:, 0] which is (128,)
        # First 10 bins: [:10, 0]
        slice_print = [f"{x:.4f}" for x in log_mel[:10, 0]]
        print(f"DEBUG: Mel[0, :10]: [{', '.join(slice_print)}]")
        
        return mx.array(log_mel)

    def mel2hidden(self, mel):
        # mel: (128, T)
        n_frames = mel.shape[-1]
        pad_curr = 32 * ((n_frames - 1) // 32 + 1) - n_frames

        # Use reflect padding to match PyTorch
        if pad_curr > 0:
            # Reflect padding: mirror the signal WITHOUT including the edge value
            # PyTorch reflect mode: [1,2,3,4] + pad(2) = [1,2,3,4,3,2]
            mel_np = np.array(mel)
            if pad_curr <= n_frames - 1:
                # Reflect the last (pad_curr) elements, excluding the edge
                # E.g., for [..., a, b, c] with pad_curr=2, add [b, a]
                reflected = mel_np[:, -(pad_curr+1):-1][:, ::-1]
            else:
                # Complex case: need to wrap around multiple times
                reflected_parts = []
                remaining = pad_curr
                offset = 1
                while remaining > 0:
                    # Available elements for reflection (excluding edges)
                    available = n_frames - offset
                    if available <= 0:
                        # Wrap around: restart from the other end
                        offset = 1
                        available = n_frames - offset
                    chunk_size = min(remaining, available)
                    reflected_parts.append(mel_np[:, -(offset+chunk_size):-offset][:, ::-1])
                    remaining -= chunk_size
                    offset += chunk_size
                reflected = np.concatenate(reflected_parts, axis=1)[:, :pad_curr]
            mel_padded = mx.array(np.concatenate([mel_np, reflected], axis=1))
        else:
            mel_padded = mel

        # (N, T, M, C) = (1, T, 128, 1)
        mel_mx = mel_padded.transpose(1, 0)[None, :, :, None]
        hidden = self.model(mel_mx)
        return hidden[:, :n_frames, :]

    def decode(self, hidden, thred=0.03):
        """
        Decodes hidden representation to F0.
        Matches PyTorch's to_local_average_cents exactly.
        """
        hidden = np.array(hidden)
        if hidden.ndim == 3: 
            hidden = hidden[0]
        
        # Find center (argmax) for each frame
        center = np.argmax(hidden, axis=1)
        
        # Pad hidden for window gathering (matches PyTorch)
        salience = np.pad(hidden, ((0, 0), (4, 4)), mode='constant')
        
        # Adjust center indices to account for padding (PyTorch does: center += 4)
        center = center + 4
        
        # Extract 9-sample windows around each center
        starts = center - 4
        ends = center + 5
        
        # Vectorized extraction (matching PyTorch's loop)
        todo_salience = []
        todo_cents_mapping = []
        for idx in range(salience.shape[0]):
            todo_salience.append(salience[idx, starts[idx]:ends[idx]])
            todo_cents_mapping.append(self.cents_mapping[starts[idx]:ends[idx]])
        
        todo_salience = np.array(todo_salience)
        todo_cents_mapping = np.array(todo_cents_mapping)
        
        # Weighted average of cents
        product_sum = np.sum(todo_salience * todo_cents_mapping, axis=1)
        weight_sum = np.sum(todo_salience, axis=1)
        
        # Avoid division by zero
        cents_pred = np.divide(product_sum, weight_sum, 
                               out=np.zeros_like(product_sum), 
                               where=weight_sum != 0)
        
        # Apply threshold based on max salience
        maxx = np.max(salience, axis=1)
        cents_pred[maxx <= thred] = 0
        
        # Convert cents to F0 (matching PyTorch: f0 = 10 * 2^(cents/1200))
        f0 = 10 * (2 ** (cents_pred / 1200))
        f0[f0 == 10] = 0  # Zero out where cents_pred was 0
        
        return f0

    def infer_from_audio(self, audio, thred=0.03):
        mel = self.mel_spectrogram(audio)
        hidden = self.mel2hidden(mel)
        f0 = self.decode(hidden, thred=thred)
        return f0
