"""
Audio Slicer for RVC MLX Training

VAD-based audio slicing with high-pass filtering and normalization.
Pure NumPy implementation (no PyTorch dependencies).
"""

import os
import numpy as np
from scipy import signal
from scipy.io import wavfile
import librosa
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import json
from typing import Optional, List, Tuple

# Constants
OVERLAP = 0.3
PERCENTAGE = 3.0
MAX_AMPLITUDE = 0.9
ALPHA = 0.75
HIGH_PASS_CUTOFF = 48
SAMPLE_RATE_16K = 16000


def get_rms(y: np.ndarray, frame_length: int = 2048, hop_length: int = 512) -> np.ndarray:
    """Calculate RMS energy of audio signal."""
    padding = (frame_length // 2, frame_length // 2)
    y = np.pad(y, padding, mode="constant")

    # Create frames using stride tricks
    out_strides = y.strides + (y.strides[-1],)
    x_shape_trimmed = list(y.shape)
    x_shape_trimmed[-1] -= frame_length - 1
    out_shape = tuple(x_shape_trimmed) + (frame_length,)
    xw = np.lib.stride_tricks.as_strided(y, shape=out_shape, strides=out_strides)

    # Subsample by hop_length
    slices = [slice(None)] * xw.ndim
    slices[-2] = slice(0, None, hop_length)
    x = xw[tuple(slices)]

    # Compute RMS
    power = np.mean(np.abs(x) ** 2, axis=-1, keepdims=True)
    return np.sqrt(power)


class AudioSlicer:
    """
    VAD-based audio slicer for preprocessing training data.

    Slices audio into segments based on silence detection.
    """

    def __init__(
        self,
        sr: int,
        threshold: float = -42.0,
        min_length: int = 1500,
        min_interval: int = 400,
        hop_size: int = 15,
        max_sil_kept: int = 500,
    ):
        if not min_length >= min_interval >= hop_size:
            raise ValueError("min_length >= min_interval >= hop_size is required")
        if not max_sil_kept >= hop_size:
            raise ValueError("max_sil_kept >= hop_size is required")

        min_interval_samples = sr * min_interval / 1000
        self.threshold = 10 ** (threshold / 20.0)
        self.hop_size = round(sr * hop_size / 1000)
        self.win_size = min(round(min_interval_samples), 4 * self.hop_size)
        self.min_length = round(sr * min_length / 1000 / self.hop_size)
        self.min_interval = round(min_interval_samples / self.hop_size)
        self.max_sil_kept = round(sr * max_sil_kept / 1000 / self.hop_size)

    def _apply_slice(self, waveform: np.ndarray, begin: int, end: int) -> np.ndarray:
        """Extract a slice from the waveform."""
        start_idx = begin * self.hop_size
        if len(waveform.shape) > 1:
            end_idx = min(waveform.shape[1], end * self.hop_size)
            return waveform[:, start_idx:end_idx]
        else:
            end_idx = min(waveform.shape[0], end * self.hop_size)
            return waveform[start_idx:end_idx]

    def slice(self, waveform: np.ndarray) -> List[np.ndarray]:
        """Slice waveform into segments based on silence detection."""
        samples = waveform.mean(axis=0) if len(waveform.shape) > 1 else waveform
        if samples.shape[0] <= self.min_length:
            return [waveform]

        rms_list = get_rms(samples, frame_length=self.win_size, hop_length=self.hop_size).squeeze()

        # Detect silence segments
        sil_tags = []
        silence_start, clip_start = None, 0

        for i, rms in enumerate(rms_list):
            if rms < self.threshold:
                if silence_start is None:
                    silence_start = i
                continue

            if silence_start is None:
                continue

            is_leading_silence = silence_start == 0 and i > self.max_sil_kept
            need_slice_middle = (
                i - silence_start >= self.min_interval
                and i - clip_start >= self.min_length
            )

            if not is_leading_silence and not need_slice_middle:
                silence_start = None
                continue

            if i - silence_start <= self.max_sil_kept:
                pos = rms_list[silence_start:i + 1].argmin() + silence_start
                if silence_start == 0:
                    sil_tags.append((0, pos))
                else:
                    sil_tags.append((pos, pos))
                clip_start = pos
            elif i - silence_start <= self.max_sil_kept * 2:
                pos = rms_list[i - self.max_sil_kept:silence_start + self.max_sil_kept + 1].argmin()
                pos += i - self.max_sil_kept
                pos_l = rms_list[silence_start:silence_start + self.max_sil_kept + 1].argmin() + silence_start
                pos_r = rms_list[i - self.max_sil_kept:i + 1].argmin() + i - self.max_sil_kept
                if silence_start == 0:
                    sil_tags.append((0, pos_r))
                    clip_start = pos_r
                else:
                    sil_tags.append((min(pos_l, pos), max(pos_r, pos)))
                    clip_start = max(pos_r, pos)
            else:
                pos_l = rms_list[silence_start:silence_start + self.max_sil_kept + 1].argmin() + silence_start
                pos_r = rms_list[i - self.max_sil_kept:i + 1].argmin() + i - self.max_sil_kept
                if silence_start == 0:
                    sil_tags.append((0, pos_r))
                else:
                    sil_tags.append((pos_l, pos_r))
                clip_start = pos_r
            silence_start = None

        # Handle trailing silence
        total_frames = rms_list.shape[0]
        if silence_start is not None and total_frames - silence_start >= self.min_interval:
            silence_end = min(total_frames, silence_start + self.max_sil_kept)
            pos = rms_list[silence_start:silence_end + 1].argmin() + silence_start
            sil_tags.append((pos, total_frames + 1))

        # Extract segments
        if not sil_tags:
            return [waveform]

        chunks = []
        if sil_tags[0][0] > 0:
            chunks.append(self._apply_slice(waveform, 0, sil_tags[0][0]))

        for i in range(len(sil_tags) - 1):
            chunks.append(self._apply_slice(waveform, sil_tags[i][1], sil_tags[i + 1][0]))

        if sil_tags[-1][1] < total_frames:
            chunks.append(self._apply_slice(waveform, sil_tags[-1][1], total_frames))

        return chunks


class AudioPreprocessor:
    """
    Main audio preprocessor for RVC training.

    Handles high-pass filtering, normalization, slicing, and resampling.
    """

    def __init__(self, sr: int, exp_dir: str):
        self.sr = sr
        self.exp_dir = exp_dir
        self.slicer = AudioSlicer(sr=sr)

        # High-pass filter
        self.b_high, self.a_high = signal.butter(
            N=5, Wn=HIGH_PASS_CUTOFF, btype="high", fs=sr
        )

        # Output directories
        self.gt_wavs_dir = os.path.join(exp_dir, "sliced_audios")
        self.wavs16k_dir = os.path.join(exp_dir, "sliced_audios_16k")
        os.makedirs(self.gt_wavs_dir, exist_ok=True)
        os.makedirs(self.wavs16k_dir, exist_ok=True)

    def _normalize_audio(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """Normalize audio amplitude."""
        tmp_max = np.abs(audio).max()
        if tmp_max > 2.5:
            return None
        return (audio / tmp_max * (MAX_AMPLITUDE * ALPHA)) + (1 - ALPHA) * audio

    def _save_segment(
        self,
        audio: np.ndarray,
        sid: int,
        idx0: int,
        idx1: int,
        normalize_post: bool = False,
    ):
        """Save audio segment at both full SR and 16kHz."""
        if audio is None:
            return

        if normalize_post:
            audio = self._normalize_audio(audio)
            if audio is None:
                return

        # Save at full sample rate
        wavfile.write(
            os.path.join(self.gt_wavs_dir, f"{sid}_{idx0}_{idx1}.wav"),
            self.sr,
            audio.astype(np.float32),
        )

        # Resample to 16kHz and save
        audio_16k = librosa.resample(audio, orig_sr=self.sr, target_sr=SAMPLE_RATE_16K)
        wavfile.write(
            os.path.join(self.wavs16k_dir, f"{sid}_{idx0}_{idx1}.wav"),
            SAMPLE_RATE_16K,
            audio_16k.astype(np.float32),
        )

    def _simple_cut(
        self,
        audio: np.ndarray,
        sid: int,
        idx0: int,
        chunk_len: float = 3.0,
        overlap_len: float = 0.3,
        normalize_post: bool = False,
    ):
        """Cut audio into fixed-length chunks."""
        chunk_length = int(self.sr * chunk_len)
        overlap_length = int(self.sr * overlap_len)

        i = 0
        segment_idx = 0
        while i < len(audio):
            chunk = audio[i:i + chunk_length]
            if len(chunk) == chunk_length:
                self._save_segment(chunk, sid, idx0, segment_idx, normalize_post)
                segment_idx += 1
            i += chunk_length - overlap_length

    def process_audio(
        self,
        path: str,
        idx0: int,
        sid: int = 0,
        cut_mode: str = "Automatic",
        apply_highpass: bool = True,
        normalize_pre: bool = True,
        chunk_len: float = 3.0,
        overlap_len: float = 0.3,
    ) -> float:
        """
        Process a single audio file.

        Returns duration in seconds.
        """
        try:
            # Load audio
            audio, _ = librosa.load(path, sr=self.sr)
            duration = librosa.get_duration(y=audio, sr=self.sr)

            # Apply high-pass filter
            if apply_highpass:
                audio = signal.lfilter(self.b_high, self.a_high, audio)

            # Pre-normalize
            if normalize_pre:
                audio = self._normalize_audio(audio)
                if audio is None:
                    print(f"Skipped {path}: amplitude too high")
                    return 0.0

            # Cut/Slice
            if cut_mode == "Skip":
                self._save_segment(audio, sid, idx0, 0)
            elif cut_mode == "Simple":
                self._simple_cut(audio, sid, idx0, chunk_len, overlap_len)
            elif cut_mode == "Automatic":
                idx1 = 0
                for segment in self.slicer.slice(audio):
                    # Further cut into smaller chunks with overlap
                    i = 0
                    while True:
                        start = int(self.sr * (PERCENTAGE - OVERLAP) * i)
                        i += 1
                        if len(segment[start:]) > (PERCENTAGE + OVERLAP) * self.sr:
                            chunk = segment[start:start + int(PERCENTAGE * self.sr)]
                            self._save_segment(chunk, sid, idx0, idx1)
                            idx1 += 1
                        else:
                            chunk = segment[start:]
                            self._save_segment(chunk, sid, idx0, idx1)
                            idx1 += 1
                            break

            return duration

        except Exception as e:
            print(f"Error processing {path}: {e}")
            return 0.0


def preprocess_audio(
    input_dir: str,
    exp_dir: str,
    sr: int = 40000,
    cut_mode: str = "Automatic",
    apply_highpass: bool = True,
    normalize: bool = True,
    num_workers: int = 4,
) -> float:
    """
    Preprocess all audio files in a directory.

    Args:
        input_dir: Directory containing audio files
        exp_dir: Experiment directory for output
        sr: Target sample rate
        cut_mode: "Skip", "Simple", or "Automatic"
        apply_highpass: Apply high-pass filter
        normalize: Normalize audio amplitude
        num_workers: Number of parallel workers

    Returns:
        Total duration of processed audio in seconds
    """
    preprocessor = AudioPreprocessor(sr, exp_dir)

    # Collect audio files
    files = []
    idx = 0
    for root, _, filenames in os.walk(input_dir):
        try:
            sid = 0 if root == input_dir else int(os.path.basename(root))
        except ValueError:
            sid = 0

        for f in filenames:
            if f.lower().endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a")):
                files.append((os.path.join(root, f), idx, sid))
                idx += 1

    print(f"Found {len(files)} audio files")

    # Process files
    total_duration = 0.0
    with tqdm(total=len(files), desc="Preprocessing") as pbar:
        for filepath, idx0, sid in files:
            duration = preprocessor.process_audio(
                filepath, idx0, sid, cut_mode, apply_highpass, normalize
            )
            total_duration += duration
            pbar.update(1)

    # Save metadata
    metadata = {
        "total_duration_seconds": total_duration,
        "total_duration_formatted": format_duration(total_duration),
        "sample_rate": sr,
        "num_files": len(files),
    }
    with open(os.path.join(exp_dir, "preprocess_info.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Preprocessed {format_duration(total_duration)} of audio")
    return total_duration


def format_duration(seconds: float) -> str:
    """Format duration as HH:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
