import os
import sys
import numpy as np
import mlx.core as mx
import librosa
from scipy import signal
try:
    import faiss
except ImportError:
    faiss = None
    print("Warning: faiss not installed. Index usage will fail.")

from rvc_mlx.lib.mlx.rmvpe import RMVPE0Predictor
from rvc_mlx.lib.mlx.pitch_extractors import PitchExtractor

class AudioProcessor:
    @staticmethod
    def change_rms(source_audio, source_rate, target_audio, target_rate, rate):
        # Calculate RMS
        rms1 = librosa.feature.rms(
            y=source_audio,
            frame_length=source_rate // 2 * 2,
            hop_length=source_rate // 2,
        )
        rms2 = librosa.feature.rms(
            y=target_audio,
            frame_length=target_rate // 2 * 2,
            hop_length=target_rate // 2,
        )

        # Interpolate RMS to match target audio length
        # Using scipy.ndimage or numpy interp
        # rms1 shape: (1, T_frames)
        # target_audio shape: (T_samples,)
        
        t_out = target_audio.shape[0]
        
        def interp_rms(rms, target_len):
            x_old = np.linspace(0, 1, rms.shape[1])
            x_new = np.linspace(0, 1, target_len)
            return np.interp(x_new, x_old, rms[0])

        rms1 = interp_rms(rms1, t_out)
        rms2 = interp_rms(rms2, t_out)
        
        rms2 = np.maximum(rms2, 1e-6)

        # Adjust
        # formula: target * (rms1^(1-rate) * rms2^(rate-1))
        # wait, original: target * (rms1^(1-rate) * rms2^(rate-1))
        # = target * (rms1 / rms2)^(1-rate) ? No.
        # factor = rms1^(1-rate) * rms2^(rate-1)
        
        factor = np.power(rms1, 1 - rate) * np.power(rms2, rate - 1)
        adjusted_audio = target_audio * factor
        return adjusted_audio

class Autotune:
    def __init__(self):
        self.note_dict = [
             49.00, 51.91, 55.00, 58.27, 61.74, 65.41, 69.30, 73.42, 77.78, 82.41, 87.31, 92.50,
             98.00, 103.83, 110.00, 116.54, 123.47, 130.81, 138.59, 146.83, 155.56, 164.81, 
             174.61, 185.00, 196.00, 207.65, 220.00, 233.08, 246.94, 261.63, 277.18, 293.66, 
             311.13, 329.63, 349.23, 369.99, 392.00, 415.30, 440.00, 466.16, 493.88, 523.25, 
             554.37, 587.33, 622.25, 659.25, 698.46, 739.99, 783.99, 830.61, 880.00, 932.33, 
             987.77, 1046.50
        ]

    def autotune_f0(self, f0, strength):
        autotuned_f0 = np.zeros_like(f0)
        # Vectorized closest note finding might be faster but loop is okay for 1D F0
        for i, freq in enumerate(f0):
            if freq <= 0:
                autotuned_f0[i] = freq
                continue
            closest_note = min(self.note_dict, key=lambda x: abs(x - freq))
            autotuned_f0[i] = freq + (closest_note - freq) * strength
        return autotuned_f0

class PipelineMLX:
    # Supported pitch extraction methods
    SUPPORTED_F0_METHODS = PitchExtractor.METHODS

    def __init__(self, tgt_sr, config, hubert_model=None, rmvpe_model=None, f0_method="rmvpe"):
        self.x_pad = config.x_pad
        self.x_query = config.x_query
        self.x_center = config.x_center
        self.x_max = config.x_max
        self.sample_rate = 16000
        self.tgt_sr = tgt_sr
        self.window = 160
        self.t_pad = self.sample_rate * self.x_pad
        self.t_pad_tgt = tgt_sr * self.x_pad
        self.t_pad2 = self.t_pad * 2
        self.t_query = self.sample_rate * self.x_query
        self.t_center = self.sample_rate * self.x_center
        self.t_max = self.sample_rate * self.x_max
        self.time_step = self.window / self.sample_rate * 1000
        self.f0_min = 50
        self.f0_max = 1100
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)

        self.autotune = Autotune()

        # Inject MLX models
        self.hubert_model = hubert_model

        # Initialize pitch extractor - supports multiple methods
        self._f0_method = f0_method
        self._pitch_extractor = None

        # Keep legacy RMVPE model for backward compatibility
        if rmvpe_model:
            self.rmvpe_model = rmvpe_model
        else:
            self.rmvpe_model = None  # Will be initialized on demand

    def _get_pitch_extractor(self, method):
        """Get pitch extractor for given method, caching the current one."""
        if self._pitch_extractor is None or self._f0_method != method:
            self._f0_method = method
            try:
                self._pitch_extractor = PitchExtractor(
                    method=method,
                    sample_rate=self.sample_rate,
                    hop_size=self.window,
                )
            except Exception as e:
                print(f"Warning: Failed to initialize {method} extractor: {e}. Falling back to rmvpe.")
                self._pitch_extractor = PitchExtractor(method="rmvpe")
                self._f0_method = "rmvpe"
        return self._pitch_extractor

    def get_f0(self, x, p_len, f0_method, pitch, f0_autotune, f0_autotune_strength, proposed_pitch, proposed_pitch_threshold):
        # x: numpy array

        # Use PitchExtractor for all methods
        extractor = self._get_pitch_extractor(f0_method)
        f0 = extractor.extract(x, f0_min=self.f0_min, f0_max=self.f0_max)

        # F0 adjustments
        if f0_autotune:
            f0 = self.autotune.autotune_f0(f0, f0_autotune_strength)
        
        # Pitch shift
        f0 *= pow(2, pitch / 12)
        
        # Coarse F0
        f0bak = f0.copy()
        
        # Map to Mel-like scale for quantization (0-255)
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * 254 / (self.f0_mel_max - self.f0_mel_min) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse = np.rint(f0_mel).astype(int)
        
        print(f"DEBUG DATA DUMP (Python):")
        f0_print = [f"{x:.4f}" for x in f0bak[:20]]
        print(f"F0 (First 20): [{', '.join(f0_print)}]")
        print(f"Pitch (First 20): {f0_coarse[:20].tolist()}")
        
        return f0_coarse, f0bak

    def voice_conversion(self, model, net_g, sid, audio0, pitch, pitchf, index, big_npy, index_rate, version, protect):
        # model: Hubert (MLX)
        # net_g: Synthesizer (MLX)
        # audio0: numpy (T,)
        
        # 1. Feature Extraction (Hubert)
        audio_mx = mx.array(audio0)[None, :]
        feats = self.hubert_model(audio_mx) 
        # feats is MLX array. Convert to CPU/Numpy for Faiss
        feats_np = np.array(feats) # (1, L, C)
        
        phone_print = [f"{x:.4f}" for x in feats_np[0, :20, 0]]
        print(f"Phone[0] (First 20): [{', '.join(phone_print)}]")
        
        # Store raw features BEFORE index retrieval for protection logic
        feats0_np = feats_np.copy()
        
        if index is not None and index_rate > 0:
            # feats_np[0] shape (L, C)
            score, ix = index.search(feats_np[0], k=8)
            weight = np.square(1 / score)
            weight /= weight.sum(axis=1, keepdims=True)
            # big_npy: (N_vecs, C)
            # ix: (L, 8)
            # Gather neighbors
            
            # numpy advanced indexing
            # big_npy[ix] -> (L, 8, C)
            neighbors = big_npy[ix]
            
            # Weighted sum
            # (L, 8, C) * (L, 8, 1) -> sum(1) -> (L, C)
            new_feats = np.sum(neighbors * np.expand_dims(weight, axis=2), axis=1)
            
            # Blend
            feats_np = index_rate * new_feats[None, :, :] + (1 - index_rate) * feats_np
            
            # Back to MLX
            feats = mx.array(feats_np)
            
        # 3. Upsample Features
        # Original: F.interpolate(permute...)
        # features (B, L, C). We need to double L.
        # Nearest neighbor upsampling
        B, L, C = feats.shape
        # Broadcast repeat
        # (B, L, 1, C) -> (B, L, 2, C) -> (B, L*2, C)
        feats = mx.broadcast_to(feats[:, :, None, :], (B, L, 2, C)).reshape(B, L*2, C)
        
        # 4. Prepare Pitch
        p_len = min(audio0.shape[0] // self.window, feats.shape[1])
        feats = feats[:, :p_len, :]
        
        pitch_guidance = (pitch is not None and pitchf is not None)
        
        pitch_mx = None
        pitchf_mx = None
        
        if pitch_guidance:
            # pitch, pitchf are numpy arrays from get_f0
            pitch = pitch[:p_len]
            pitchf = pitchf[:p_len]
            
            # Protection: blend indexed features with raw features for unvoiced segments
            if protect < 0.5:
                # Create protection mask based on pitch
                # pitchff = 1 for voiced (pitchf > 0), protect for unvoiced
                pitchff = np.where(pitchf > 0, 1.0, protect)
                pitchff = pitchff[:, None]  # (L, 1) for broadcasting with (L, C)
                
                # Get upsampled feats0 (raw features)
                feats0_np_up = feats0_np.copy()
                B0, L0, C0 = feats0_np_up.shape
                feats0_np_up = np.repeat(feats0_np_up, 2, axis=1)[:, :p_len, :]  # Match upsampled length
                
                # Blend: for unvoiced parts, use more of the raw features
                feats_np_current = np.array(feats)
                feats_np_current = feats_np_current * pitchff[None, :, :] + feats0_np_up * (1 - pitchff[None, :, :])
                feats = mx.array(feats_np_current)
            
            pitch_mx = mx.array(pitch)[None, :] # (1, L)
            pitchf_mx = mx.array(pitchf)[None, :] # (1, L)

        sid_mx = mx.array(sid)[None] # (1,)
        
        # 5. Inference
        out_audio, _, _ = net_g.infer(
            feats,
            mx.array([p_len]),
            pitch_mx,
            pitchf_mx,
            sid_mx
        )
        
        # out_audio (1, T, 1) -> (T,)
        return np.array(out_audio[0, :, 0])
    
    def pipeline(self, model, net_g, sid, audio, pitch, f0_method, file_index, index_rate, pitch_guidance, volume_envelope, version, protect, f0_autotune, f0_autotune_strength, proposed_pitch, proposed_pitch_threshold):
        # Load Index
        index = None
        big_npy = None
        if file_index and os.path.exists(file_index) and index_rate > 0:
            try:
                index = faiss.read_index(file_index)
                if index.is_trained:
                     # reconstruct_n might fail if not IVF?
                     # Standard RVC indices are usually IVF.
                     # But sometimes we might need direct access.
                     pass
                # faiss big_npy reconstruction depends on index type.
                # RVC typical usage:
                if hasattr(index, 'reconstruct_n'):
                     big_npy = index.reconstruct_n(0, index.ntotal)
                # Fallback?
            except Exception as e:
                print(f"Index load error: {e}")
        
        # Filter audio (High pass)
        bh, ah = signal.butter(5, 48, btype="high", fs=16000)
        audio = signal.filtfilt(bh, ah, audio)
        
        # Pad audio
        # audio_pad = np.pad(audio, (self.window // 2, self.window // 2), mode="reflect")
        
        # Chunking Logic (opt_ts)
        # Simplified: Just process whole audio if short, or chunk if long.
        # Original logic finds zero-crossings (or min amplitude) to split.
        
        # For simplicity in this port, let's assume one chunk (or simple time split) for now
        # unless strict compliance is needed. The original split logic is complex numpy.
        # I'll perform simple processing for now to ensure MVP works. 
        # RVC often splits to avoid OOM on GPU. MLX UMA might handle larger batches better?
        # But let's verify.
        
        # t_pad = self.t_pad
        t_pad = 1600 # Force 0.1s padding for parity check stability
        t_pad_tgt = self.t_pad_tgt
        
        # Just calling voice_conversion on the padded audio
        # Note: Proper implementation needs the overlap-add logic for high quality on long files.
        # We will wrap the core `voice_conversion` call.
        
        # Let's reuse proper padding from original:
        audio_pad = np.pad(audio, (int(t_pad), int(t_pad)), mode="reflect")
        
        audio_print = [f"{x:.4f}" for x in audio_pad[:20]]
        print(f"DEBUG: Audio input (padded, filtered) first 20 samples: [{', '.join(audio_print)}]")

        p_len = audio_pad.shape[0] // self.window
        
        pitch_data = None
        pitchf_data = None
        
        if pitch_guidance:
            pitch_data, pitchf_data = self.get_f0(
                audio_pad,
                p_len,
                f0_method,
                pitch,
                f0_autotune,
                f0_autotune_strength,
                proposed_pitch,
                proposed_pitch_threshold
            )
            pitch_data = pitch_data[:p_len]
            pitchf_data = pitchf_data[:p_len]
            
        # Run inference
        audio_out = self.voice_conversion(
            model,
            net_g,
            sid,
            audio_pad,
            pitch_data,
            pitchf_data,
            index,
            big_npy,
            index_rate,
            version,
            protect
        )
        
        # Calculate upsample factor from net_g if possible, or use global
        upsample_factor = getattr(net_g, "upp", self.window // 160 * (self.tgt_sr // 100)) # fallback
        if hasattr(net_g, "dec") and hasattr(net_g.dec, "upp"):
             upsample_factor = net_g.dec.upp

        # Trim padding
        # The padding was self.t_pad samples at 16k.
        # At target SR, this is self.t_pad * (self.tgt_sr / 16000)
        # Or more simply: self.t_pad * (upsample_factor / self.window)
        # Actually t_pad is already samples. t_pad_tgt = t_pad * (tgt_sr / 16000)
        
        actual_t_pad_tgt = int(t_pad * (self.tgt_sr / 16000))
        audio_out = audio_out[actual_t_pad_tgt : -actual_t_pad_tgt]
        
        # Volume Envelope
        if volume_envelope != 1:
            audio_out = AudioProcessor.change_rms(
                audio, self.sample_rate, audio_out, self.tgt_sr, volume_envelope
            )
        
        # Audio max normalization (match PyTorch behavior)
        audio_max = np.abs(audio_out).max() / 0.99
        if audio_max > 1:
            audio_out = audio_out / audio_max
            
        return audio_out
