"""
Data Loader for RVC MLX Training

Dataset and batching utilities for training.
"""

import os
import numpy as np
import mlx.core as mx
from scipy.io import wavfile
from typing import List, Dict, Tuple, Optional, Iterator
import random

from rvc_mlx.train.mel_processing import spectrogram


class RVCDataset:
    """
    Dataset for RVC training.

    Loads preprocessed features:
    - Audio waveform
    - HuBERT embeddings
    - F0 (pitch)
    - F0 coarse (quantized)
    - Speaker ID
    - Pre-computed spectrogram (optional, for faster training)
    """

    def __init__(
        self,
        filelist_path: str,
        sample_rate: int = 40000,
        hop_length: int = 320,
        max_frames: int = 900,
        use_precomputed_spec: bool = True,
    ):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.max_frames = max_frames
        self.use_precomputed_spec = use_precomputed_spec

        # Load filelist
        self.samples = self._load_filelist(filelist_path)
        print(f"Loaded {len(self.samples)} samples from {filelist_path}")
        if use_precomputed_spec:
            print("Using pre-computed spectrograms for faster training")

    def _load_filelist(self, path: str) -> List[Dict]:
        """Load filelist with format: audio|embedding|f0|f0_coarse|speaker_id[|spectrogram]"""
        samples = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("|")
                if len(parts) >= 5:
                    sample = {
                        "audio": parts[0],
                        "embedding": parts[1],
                        "f0": parts[2],
                        "f0_coarse": parts[3],
                        "speaker_id": int(parts[4]),
                    }
                    # Optional spectrogram path (6th field)
                    if len(parts) >= 6 and parts[5]:
                        sample["spectrogram"] = parts[5]
                    samples.append(sample)
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """
        Get a single sample.

        Returns dict with:
        - phone: HuBERT embeddings (T, 768), upsampled 2x
        - pitch: Coarse pitch (T,)
        - pitchf: Fine pitch/F0 (T,)
        - spec: Pre-computed spectrogram (C, T) if available
        - wave: Audio waveform
        - sid: Speaker ID
        """
        sample = self.samples[idx]

        # Load embeddings
        phone = np.load(sample["embedding"])  # (T, 768)
        phone = np.repeat(phone, 2, axis=0)  # Upsample 2x to match pitch resolution

        # Load pitch
        pitch = np.load(sample["f0_coarse"])  # Coarse pitch (T,)
        pitchf = np.load(sample["f0"])  # Fine pitch (T,)

        # Load audio
        _, audio = wavfile.read(sample["audio"])
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0

        # Load pre-computed spectrogram if available
        spec = None
        if self.use_precomputed_spec and "spectrogram" in sample:
            spec = np.load(sample["spectrogram"])  # (C, T) in PyTorch format
            # Apply log transform if not already in log scale
            # Pre-computed specs may be raw magnitude (range 0-100+)
            # Log scale should be around -10 to 5
            if spec.max() > 10:  # Heuristic: raw magnitude has high values
                spec = np.log(np.maximum(spec, 1e-5))

        # Align lengths
        min_len = min(phone.shape[0], pitch.shape[0], pitchf.shape[0])
        min_len = min(min_len, self.max_frames)

        phone = phone[:min_len]
        pitch = pitch[:min_len]
        pitchf = pitchf[:min_len]

        # Trim audio to match
        audio_len = min_len * self.hop_length
        audio = audio[:audio_len]

        # Trim spectrogram to match (spec length should be ~audio_len/hop_length)
        if spec is not None:
            spec_len = min_len  # Spec frames ~= pitch frames
            spec = spec[:, :spec_len]

        result = {
            "phone": phone.astype(np.float32),
            "pitch": pitch.astype(np.int32),
            "pitchf": pitchf.astype(np.float32),
            "wave": audio.astype(np.float32),
            "sid": np.array([sample["speaker_id"]], dtype=np.int32),
        }

        if spec is not None:
            result["spec"] = spec.astype(np.float32)

        return result


class RVCCollator:
    """
    Collate function for batching RVC samples.

    Uses pre-computed spectrograms when available for faster training.
    Falls back to on-the-fly computation if not available.
    """

    def __init__(
        self,
        hop_length: int = 320,
        sample_rate: int = 40000,
        n_fft: int = 2048,
        win_length: int = 2048,
    ):
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length

    def __call__(self, batch: List[Dict[str, np.ndarray]]) -> Dict[str, mx.array]:
        """
        Collate batch of samples.

        Returns dict with padded tensors:
        - phone: (B, T, 768)
        - phone_lengths: (B,)
        - pitch: (B, T)
        - pitchf: (B, T)
        - spec: (B, C, T_spec) - spectrogram for training
        - wave: (B, T_audio)
        - wave_lengths: (B,)
        - sid: (B,)
        """
        # Check if we have pre-computed spectrograms
        has_precomputed_specs = all("spec" in s for s in batch)

        # Find max lengths
        max_phone_len = max(s["phone"].shape[0] for s in batch)
        max_wave_len = max(s["wave"].shape[0] for s in batch)
        if has_precomputed_specs:
            max_spec_len = max(s["spec"].shape[1] for s in batch)  # spec is (C, T)

        # Pad and stack
        phones = []
        phone_lengths = []
        pitches = []
        pitchfs = []
        waves = []
        wave_lengths = []
        sids = []
        specs = [] if has_precomputed_specs else None
        spec_lengths = []

        for sample in batch:
            # Phone
            phone_len = sample["phone"].shape[0]
            phone_padded = np.pad(
                sample["phone"],
                [(0, max_phone_len - phone_len), (0, 0)],
                mode="constant",
            )
            phones.append(phone_padded)
            phone_lengths.append(phone_len)

            # Pitch (pad with zeros)
            pitch_padded = np.pad(
                sample["pitch"],
                [(0, max_phone_len - sample["pitch"].shape[0])],
                mode="constant",
            )
            pitches.append(pitch_padded)

            # PitchF
            pitchf_padded = np.pad(
                sample["pitchf"],
                [(0, max_phone_len - sample["pitchf"].shape[0])],
                mode="constant",
            )
            pitchfs.append(pitchf_padded)

            # Wave
            wave_len = sample["wave"].shape[0]
            wave_padded = np.pad(
                sample["wave"],
                [(0, max_wave_len - wave_len)],
                mode="constant",
            )
            waves.append(wave_padded)
            wave_lengths.append(wave_len)

            # Speaker ID
            sids.append(sample["sid"][0])

            # Pre-computed spectrogram (if available)
            if has_precomputed_specs:
                spec = sample["spec"]  # (C, T)
                spec_len = spec.shape[1]
                spec_padded = np.pad(
                    spec,
                    [(0, 0), (0, max_spec_len - spec_len)],  # Pad T dimension only
                    mode="constant",
                )
                specs.append(spec_padded)
                spec_lengths.append(spec_len)

        # Stack waves: (B, T)
        waves_stacked = np.stack(waves)
        waves_mx = mx.array(waves_stacked.astype(np.float32))

        # Handle spectrograms
        if has_precomputed_specs:
            # Use pre-computed spectrograms (faster!)
            spec = mx.array(np.stack(specs).astype(np.float32))  # (B, C, T)
        else:
            # Compute spectrogram on-the-fly (slower)
            spec = spectrogram(
                waves_mx,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                center=True,
            )
            # Transpose to PyTorch format: (B, T, C) -> (B, C, T)
            spec = spec.transpose(0, 2, 1)
            # Compute spec lengths from wave lengths
            spec_lengths = [(wl // self.hop_length) + 1 for wl in wave_lengths]
            # Evaluate to prevent graph buildup
            mx.eval(spec)

        result = {
            "phone": mx.array(np.stack(phones)),
            "phone_lengths": mx.array(np.array(phone_lengths, dtype=np.int32)),
            "pitch": mx.array(np.stack(pitches)),
            "pitchf": mx.array(np.stack(pitchfs)),
            "spec": spec,  # (B, spec_channels, T_spec) - spectrogram for training
            "spec_lengths": mx.array(np.array(spec_lengths, dtype=np.int32)),
            "wave": waves_mx,  # (B, T) - raw waveform for discriminator
            "wave_lengths": mx.array(np.array(wave_lengths, dtype=np.int32)),
            "sid": mx.array(np.array(sids, dtype=np.int32)),
        }

        return result


class DataLoader:
    """
    Simple DataLoader for RVC training.

    Iterates over dataset with batching and optional shuffling.
    """

    def __init__(
        self,
        dataset: RVCDataset,
        batch_size: int = 4,
        shuffle: bool = True,
        collate_fn: Optional[RVCCollator] = None,
        drop_last: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.collate_fn = collate_fn or RVCCollator()
        self.drop_last = drop_last

        self.indices = list(range(len(dataset)))

    def __len__(self) -> int:
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[Dict[str, mx.array]]:
        """Iterate over batches."""
        indices = self.indices.copy()

        if self.shuffle:
            random.shuffle(indices)

        batch = []
        for idx in indices:
            try:
                sample = self.dataset[idx]
                batch.append(sample)
            except Exception as e:
                print(f"Error loading sample {idx}: {e}")
                continue

            if len(batch) == self.batch_size:
                yield self.collate_fn(batch)
                batch = []

        # Handle remaining samples
        if batch and not self.drop_last:
            yield self.collate_fn(batch)


def create_dataloader(
    filelist_path: str,
    batch_size: int = 4,
    sample_rate: int = 40000,
    hop_length: int = 320,
    shuffle: bool = True,
    max_frames: int = 900,
    n_fft: int = 2048,
    win_length: int = 2048,
    use_precomputed_spec: bool = True,
) -> DataLoader:
    """
    Create a DataLoader for RVC training.

    Args:
        filelist_path: Path to filelist.txt
        batch_size: Batch size
        sample_rate: Audio sample rate
        hop_length: Hop length for frame alignment
        shuffle: Whether to shuffle data
        max_frames: Maximum frames per sample
        n_fft: FFT size for spectrogram (2048 -> 1025 freq bins)
        win_length: Window length for spectrogram
        use_precomputed_spec: Use pre-computed spectrograms if available

    Returns:
        DataLoader instance
    """
    dataset = RVCDataset(
        filelist_path,
        sample_rate=sample_rate,
        hop_length=hop_length,
        max_frames=max_frames,
        use_precomputed_spec=use_precomputed_spec,
    )
    collator = RVCCollator(
        hop_length=hop_length,
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
    )
