"""
Feature Extractor for RVC MLX Training

Extracts HuBERT embeddings, F0 (pitch), and spectrograms using existing MLX RMVPE.
Pre-computing spectrograms avoids redundant STFT computation during training.
"""

import os
import numpy as np
import mlx.core as mx
from scipy.io import wavfile
from tqdm import tqdm
from typing import Optional, Tuple
import json

from rvc_mlx.train.mel_processing import spectrogram


class FeatureExtractor:
    """
    Extract features for RVC training.

    Features:
    - HuBERT/ContentVec embeddings (768-dim)
    - F0 (fundamental frequency) via RMVPE
    - Pre-computed spectrograms (avoids STFT during training)
    """

    def __init__(
        self,
        exp_dir: str,
        f0_method: str = "rmvpe",
        hop_size: int = 160,
        f0_min: float = 50.0,
        f0_max: float = 1100.0,
        sample_rate: int = 40000,
        n_fft: int = 2048,
        win_length: int = 2048,
        spec_hop_length: int = 320,
    ):
        self.exp_dir = exp_dir
        self.f0_method = f0_method
        self.hop_size = hop_size
        self.f0_min = f0_min
        self.f0_max = f0_max

        # Spectrogram params (matching training)
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.spec_hop_length = spec_hop_length

        # Directories
        self.wavs_dir = os.path.join(exp_dir, "sliced_audios")  # 40kHz for spectrogram
        self.wavs16k_dir = os.path.join(exp_dir, "sliced_audios_16k")  # 16kHz for HuBERT
        self.f0_dir = os.path.join(exp_dir, "f0")
        self.f0_coarse_dir = os.path.join(exp_dir, "f0_coarse")
        self.embeddings_dir = os.path.join(exp_dir, "embeddings")
        self.spectrograms_dir = os.path.join(exp_dir, "spectrograms")

        os.makedirs(self.f0_dir, exist_ok=True)
        os.makedirs(self.f0_coarse_dir, exist_ok=True)
        os.makedirs(self.embeddings_dir, exist_ok=True)
        os.makedirs(self.spectrograms_dir, exist_ok=True)

        # Models (lazy loaded)
        self._rmvpe = None
        self._hubert = None

    @property
    def rmvpe(self):
        """Lazy load RMVPE model."""
        if self._rmvpe is None:
            from rvc_mlx.lib.mlx.rmvpe import RMVPE0Predictor
            self._rmvpe = RMVPE0Predictor()
            print("Loaded RMVPE model")
        return self._rmvpe

    @property
    def hubert(self):
        """Lazy load HuBERT/ContentVec model."""
        if self._hubert is None:
            from rvc_mlx.lib.mlx.utils import load_embedding
            # Try common locations for converted HuBERT weights
            hubert_paths = [
                "rvc_mlx/models/embedders/contentvec/hubert_mlx.npz",
                "rvc/models/embedders/contentvec/hubert_mlx.npz",
                "rvc_mlx/models/embedders/contentvec/hubert_mlx.safetensors",
                "rvc/models/embedders/contentvec/hubert_mlx.safetensors",
            ]
            model_path = None
            for p in hubert_paths:
                if os.path.exists(p):
                    model_path = p
                    break
            self._hubert = load_embedding(embedder_model_custom=model_path)
            print("Loaded HuBERT model")
        return self._hubert

    def _f0_to_coarse(self, f0: np.ndarray) -> np.ndarray:
        """Convert F0 to coarse (quantized) representation."""
        f0_mel = np.zeros_like(f0)
        voiced = f0 > 0
        f0_mel[voiced] = 1127.0 * np.log(1 + f0[voiced] / 700.0)

        # Quantize to 256 bins
        f0_mel_min = 1127.0 * np.log(1 + self.f0_min / 700.0)
        f0_mel_max = 1127.0 * np.log(1 + self.f0_max / 700.0)

        f0_coarse = np.zeros_like(f0, dtype=np.int32)
        f0_coarse[voiced] = np.round(
            (f0_mel[voiced] - f0_mel_min) / (f0_mel_max - f0_mel_min) * 255
        ).astype(np.int32)
        f0_coarse = np.clip(f0_coarse, 0, 255)

        return f0_coarse

    def extract_f0(self, audio: np.ndarray, sr: int = 16000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract F0 using RMVPE.

        Returns:
            f0: Continuous F0 values
            f0_coarse: Quantized F0 (0-255)
        """
        # RMVPE0Predictor expects 16kHz audio
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        # Extract F0 using infer_from_audio
        f0 = self.rmvpe.infer_from_audio(audio, thred=0.03)
        f0 = np.array(f0)

        # Convert to coarse
        f0_coarse = self._f0_to_coarse(f0)

        return f0, f0_coarse

    def extract_embeddings(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        """
        Extract HuBERT/ContentVec embeddings.

        Returns:
            embeddings: (T, 768) feature matrix
        """
        # Convert to MLX array
        audio_mx = mx.array(audio.astype(np.float32))

        # Add batch dimension if needed
        if audio_mx.ndim == 1:
            audio_mx = audio_mx[None, :]

        # Extract 768-dim features (hidden states, not projected)
        # Training requires 768-dim features, not the 256-dim projected output
        features = self.hubert(audio_mx, output_hidden_states=True)

        # Remove batch dimension
        if features.ndim == 3:
            features = features[0]

        return np.array(features)

    def extract_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Pre-compute spectrogram for training.

        Args:
            audio: Audio waveform at self.sample_rate (40kHz)

        Returns:
            spec: Spectrogram (n_fft//2+1, T) in PyTorch format (C, T) for training
        """
        # Convert to MLX array with batch dimension
        audio_mx = mx.array(audio.astype(np.float32))
        if audio_mx.ndim == 1:
            audio_mx = audio_mx[None, :]  # (1, T)

        # Compute spectrogram: returns (B, T_spec, n_fft//2+1) in MLX format
        spec = spectrogram(
            audio_mx,
            n_fft=self.n_fft,
            hop_length=self.spec_hop_length,
            win_length=self.win_length,
            center=True,
        )

        # Transpose to PyTorch format: (B, T, C) -> (B, C, T)
        spec = spec.transpose(0, 2, 1)

        # Remove batch dimension and convert to numpy
        spec = np.array(spec[0])  # (C, T)

        return spec

    def process_file(self, filename: str) -> bool:
        """
        Process a single audio file: extract F0, embeddings, and spectrogram.

        Returns True if successful.
        """
        filepath_16k = os.path.join(self.wavs16k_dir, filename)
        filepath_full = os.path.join(self.wavs_dir, filename)

        if not os.path.exists(filepath_16k):
            return False

        basename = os.path.splitext(filename)[0]

        try:
            # Load 16kHz audio for F0 and embeddings
            sr, audio_16k = wavfile.read(filepath_16k)
            if audio_16k.dtype != np.float32:
                audio_16k = audio_16k.astype(np.float32) / 32768.0

            # Extract F0
            f0, f0_coarse = self.extract_f0(audio_16k, sr)

            # Save F0
            np.save(os.path.join(self.f0_dir, f"{basename}.npy"), f0)
            np.save(os.path.join(self.f0_coarse_dir, f"{basename}.npy"), f0_coarse)

            # Extract embeddings
            embeddings = self.extract_embeddings(audio_16k, sr)

            # Save embeddings
            np.save(os.path.join(self.embeddings_dir, f"{basename}.npy"), embeddings)

            # Load full sample rate audio for spectrogram (40kHz)
            if os.path.exists(filepath_full):
                sr_full, audio_full = wavfile.read(filepath_full)
                if audio_full.dtype != np.float32:
                    audio_full = audio_full.astype(np.float32) / 32768.0

                # Extract spectrogram
                spec = self.extract_spectrogram(audio_full)

                # Save spectrogram
                np.save(os.path.join(self.spectrograms_dir, f"{basename}.npy"), spec)

            return True

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            import traceback
            traceback.print_exc()
            return False

    def extract_all(self) -> int:
        """
        Extract features for all preprocessed audio files.

        Returns number of successfully processed files.
        """
        # Get list of 16kHz audio files
        if not os.path.exists(self.wavs16k_dir):
            print(f"No preprocessed audio found at {self.wavs16k_dir}")
            return 0

        files = [f for f in os.listdir(self.wavs16k_dir) if f.endswith(".wav")]
        print(f"Found {len(files)} audio files to process")

        success_count = 0
        with tqdm(total=len(files), desc="Extracting features") as pbar:
            for filename in files:
                if self.process_file(filename):
                    success_count += 1
                pbar.update(1)

        # Save extraction info
        info = {
            "total_files": len(files),
            "successful": success_count,
            "f0_method": self.f0_method,
            "hop_size": self.hop_size,
            "spectrogram": {
                "n_fft": self.n_fft,
                "hop_length": self.spec_hop_length,
                "win_length": self.win_length,
                "sample_rate": self.sample_rate,
            },
        }
        with open(os.path.join(self.exp_dir, "extract_info.json"), "w") as f:
            json.dump(info, f, indent=2)

        print(f"Extracted features for {success_count}/{len(files)} files")
        print(f"Pre-computed spectrograms saved to: {self.spectrograms_dir}")
        return success_count


def extract_features(
    exp_dir: str,
    f0_method: str = "rmvpe",
) -> int:
    """
    Extract features for all preprocessed audio in experiment directory.

    Args:
        exp_dir: Experiment directory with preprocessed audio
        f0_method: F0 extraction method ("rmvpe")

    Returns:
        Number of successfully processed files
    """
    extractor = FeatureExtractor(exp_dir, f0_method=f0_method)
    return extractor.extract_all()
