"""
Dataset Builder for RVC MLX Training

Creates training filelist and organizes extracted features.
"""

import os
import numpy as np
from typing import List, Tuple, Optional
import json


class DatasetBuilder:
    """
    Build training dataset from extracted features.

    Creates filelist.txt with format:
    audio_path|embedding_path|f0_path|f0_coarse_path|speaker_id|spectrogram_path
    """

    def __init__(self, exp_dir: str, sample_rate: int = 40000):
        self.exp_dir = exp_dir
        self.sample_rate = sample_rate

        # Directories
        self.gt_wavs_dir = os.path.join(exp_dir, "sliced_audios")
        self.wavs16k_dir = os.path.join(exp_dir, "sliced_audios_16k")
        self.f0_dir = os.path.join(exp_dir, "f0")
        self.f0_coarse_dir = os.path.join(exp_dir, "f0_coarse")
        self.embeddings_dir = os.path.join(exp_dir, "embeddings")
        self.spectrograms_dir = os.path.join(exp_dir, "spectrograms")

    def _parse_filename(self, filename: str) -> Tuple[int, int, int]:
        """
        Parse filename to extract speaker_id, file_idx, segment_idx.

        Format: {speaker_id}_{file_idx}_{segment_idx}.wav
        """
        basename = os.path.splitext(filename)[0]
        parts = basename.split("_")

        if len(parts) >= 3:
            try:
                speaker_id = int(parts[0])
                file_idx = int(parts[1])
                segment_idx = int(parts[2])
                return speaker_id, file_idx, segment_idx
            except ValueError:
                pass

        # Fallback
        return 0, 0, 0

    def _validate_sample(self, basename: str) -> Optional[dict]:
        """
        Validate that all required files exist for a sample.

        Returns sample info dict or None if invalid.
        """
        audio_path = os.path.join(self.gt_wavs_dir, f"{basename}.wav")
        embedding_path = os.path.join(self.embeddings_dir, f"{basename}.npy")
        f0_path = os.path.join(self.f0_dir, f"{basename}.npy")
        f0_coarse_path = os.path.join(self.f0_coarse_dir, f"{basename}.npy")
        spectrogram_path = os.path.join(self.spectrograms_dir, f"{basename}.npy")

        # Check required files exist (spectrogram is optional)
        if not all(os.path.exists(p) for p in [audio_path, embedding_path, f0_path, f0_coarse_path]):
            return None

        # Validate feature shapes match
        try:
            embeddings = np.load(embedding_path)
            f0 = np.load(f0_path)
            f0_coarse = np.load(f0_coarse_path)

            # Basic validation
            if embeddings.shape[0] == 0 or f0.shape[0] == 0:
                return None

            # Check for NaN
            if np.isnan(embeddings).any() or np.isnan(f0).any():
                return None

            speaker_id, _, _ = self._parse_filename(basename + ".wav")

            result = {
                "basename": basename,
                "audio_path": audio_path,
                "embedding_path": embedding_path,
                "f0_path": f0_path,
                "f0_coarse_path": f0_coarse_path,
                "speaker_id": speaker_id,
                "embedding_frames": embeddings.shape[0],
                "f0_frames": f0.shape[0],
            }

            # Include spectrogram path if it exists (for faster training)
            if os.path.exists(spectrogram_path):
                result["spectrogram_path"] = spectrogram_path

            return result

        except Exception as e:
            print(f"Error validating {basename}: {e}")
            return None

    def build(self, min_frames: int = 50, max_frames: int = 900) -> str:
        """
        Build the training filelist.

        Args:
            min_frames: Minimum number of frames per sample
            max_frames: Maximum number of frames per sample

        Returns:
            Path to the created filelist
        """
        # Get list of audio files
        if not os.path.exists(self.gt_wavs_dir):
            raise FileNotFoundError(f"No audio found at {self.gt_wavs_dir}")

        audio_files = [f for f in os.listdir(self.gt_wavs_dir) if f.endswith(".wav")]
        print(f"Found {len(audio_files)} audio files")

        # Validate samples
        valid_samples = []
        for filename in audio_files:
            basename = os.path.splitext(filename)[0]
            sample = self._validate_sample(basename)

            if sample is not None:
                # Apply frame filters
                frames = min(sample["embedding_frames"], sample["f0_frames"])
                if min_frames <= frames <= max_frames:
                    valid_samples.append(sample)

        print(f"Valid samples: {len(valid_samples)}/{len(audio_files)}")

        if len(valid_samples) == 0:
            raise ValueError("No valid samples found")

        # Sort by speaker_id and basename for consistent ordering
        valid_samples.sort(key=lambda x: (x["speaker_id"], x["basename"]))

        # Get unique speaker IDs
        speaker_ids = sorted(set(s["speaker_id"] for s in valid_samples))
        print(f"Found {len(speaker_ids)} speakers")

        # Write filelist
        filelist_path = os.path.join(self.exp_dir, "filelist.txt")
        samples_with_spec = 0
        with open(filelist_path, "w") as f:
            for sample in valid_samples:
                fields = [
                    sample["audio_path"],
                    sample["embedding_path"],
                    sample["f0_path"],
                    sample["f0_coarse_path"],
                    str(sample["speaker_id"]),
                ]
                # Include spectrogram path if available (6th field)
                if "spectrogram_path" in sample:
                    fields.append(sample["spectrogram_path"])
                    samples_with_spec += 1
                line = "|".join(fields)
                f.write(line + "\n")

        if samples_with_spec > 0:
            print(f"Pre-computed spectrograms available: {samples_with_spec}/{len(valid_samples)}")

        # Save dataset info
        info = {
            "total_samples": len(valid_samples),
            "num_speakers": len(speaker_ids),
            "speaker_ids": speaker_ids,
            "sample_rate": self.sample_rate,
            "min_frames": min_frames,
            "max_frames": max_frames,
            "filelist_path": filelist_path,
        }

        # Calculate total duration from audio files
        total_duration = 0
        for sample in valid_samples:
            try:
                from scipy.io import wavfile
                sr, audio = wavfile.read(sample["audio_path"])
                total_duration += len(audio) / sr
            except:
                pass

        info["total_duration_seconds"] = total_duration
        info["total_duration_formatted"] = self._format_duration(total_duration)

        with open(os.path.join(self.exp_dir, "dataset_info.json"), "w") as f:
            json.dump(info, f, indent=2)

        print(f"Created filelist at {filelist_path}")
        print(f"Total duration: {info['total_duration_formatted']}")

        return filelist_path

    def _format_duration(self, seconds: float) -> str:
        """Format duration as HH:MM:SS."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def split_train_val(
        self,
        filelist_path: str,
        val_ratio: float = 0.05,
        min_val_samples: int = 4,
    ) -> Tuple[str, str]:
        """
        Split filelist into train and validation sets.

        Args:
            filelist_path: Path to full filelist
            val_ratio: Fraction of samples for validation
            min_val_samples: Minimum number of validation samples

        Returns:
            Tuple of (train_filelist_path, val_filelist_path)
        """
        with open(filelist_path, "r") as f:
            lines = f.readlines()

        # Shuffle deterministically
        np.random.seed(42)
        indices = np.random.permutation(len(lines))

        # Calculate split
        n_val = max(min_val_samples, int(len(lines) * val_ratio))
        n_val = min(n_val, len(lines) // 2)  # Don't use more than half for val

        val_indices = set(indices[:n_val])

        train_lines = [lines[i] for i in range(len(lines)) if i not in val_indices]
        val_lines = [lines[i] for i in val_indices]

        # Write split filelists
        train_path = os.path.join(self.exp_dir, "filelist_train.txt")
        val_path = os.path.join(self.exp_dir, "filelist_val.txt")

        with open(train_path, "w") as f:
            f.writelines(train_lines)

        with open(val_path, "w") as f:
            f.writelines(val_lines)

        print(f"Train samples: {len(train_lines)}, Val samples: {len(val_lines)}")

        return train_path, val_path


def build_dataset(
    exp_dir: str,
    sample_rate: int = 40000,
    min_frames: int = 50,
    max_frames: int = 900,
    create_split: bool = True,
    val_ratio: float = 0.05,
) -> str:
    """
    Build training dataset from extracted features.

    Args:
        exp_dir: Experiment directory with extracted features
        sample_rate: Audio sample rate
        min_frames: Minimum frames per sample
        max_frames: Maximum frames per sample
        create_split: Whether to create train/val split
        val_ratio: Validation set ratio

    Returns:
        Path to the training filelist
    """
    builder = DatasetBuilder(exp_dir, sample_rate)
    filelist_path = builder.build(min_frames, max_frames)

    if create_split:
        train_path, val_path = builder.split_train_val(filelist_path, val_ratio)
        return train_path

    return filelist_path
