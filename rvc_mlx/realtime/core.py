
import os
import sys
import time
import numpy as np
import mlx.core as mx
import librosa
import scipy.signal
from pedalboard import (
    Pedalboard,
    Chorus,
    Distortion,
    Reverb,
    PitchShift,
    Limiter,
    Gain,
    Bitcrush,
    Clipping,
    Compressor,
    Delay,
)

now_dir = os.getcwd()
sys.path.append(now_dir)

from rvc_mlx.realtime.utils.buffer import circular_write
from rvc_mlx.realtime.utils.vad import VADProcessor
from rvc_mlx.realtime.pipeline import create_pipeline

SAMPLE_RATE = 16000
AUDIO_SAMPLE_RATE = 48000


class Realtime:
    def __init__(
        self,
        model_path: str = None,
        index_path: str = None,
        f0_method: str = "rmvpe",
        embedder_model: str = None,
        embedder_model_custom: str = None,
        silent_threshold: int = 0,
        vad_enabled: bool = False,
        vad_sensitivity: int = 3,
        vad_frame_ms: int = 30,
        sid: int = 0,
        clean_audio: bool = False,
        clean_strength: float = 0.5,
        post_process: bool = False,
        **kwargs,
    ):
        self.sample_rate = SAMPLE_RATE
        self.convert_buffer = None
        self.pitch_buffer = None
        self.pitchf_buffer = None
        self.return_length = 0
        self.skip_head = 0
        self.silence_front = 0
        self.input_sensitivity = 10 ** (silent_threshold / 20)
        self.window_size = self.sample_rate // 100
        self.dtype = np.float32

        self.vad = (
            VADProcessor(
                sensitivity_mode=vad_sensitivity,
                sample_rate=self.sample_rate,
                frame_duration_ms=vad_frame_ms,
            )
            if vad_enabled
            else None
        )
        self.board = self.setup_pedalboard(**kwargs) if post_process else None
        
        self.pipeline = create_pipeline(
            model_path,
            index_path,
            f0_method,
            embedder_model,
            embedder_model_custom,
            sid,
        )
        
        # Noise reduction placeholder (removed TorchGate)
        self.reduced_noise = None 

    def setup_pedalboard(self, **kwargs):
        board = Pedalboard()
        if kwargs.get("reverb", False):
            reverb = Reverb(
                room_size=kwargs.get("reverb_room_size", 0.5),
                damping=kwargs.get("reverb_damping", 0.5),
                wet_level=kwargs.get("reverb_wet_level", 0.33),
                dry_level=kwargs.get("reverb_dry_level", 0.4),
                width=kwargs.get("reverb_width", 1.0),
                freeze_mode=kwargs.get("reverb_freeze_mode", 0),
            )
            board.append(reverb)
        if kwargs.get("pitch_shift", False):
            pitch_shift = PitchShift(semitones=kwargs.get("pitch_shift_semitones", 0))
            board.append(pitch_shift)
        if kwargs.get("limiter", False):
            limiter = Limiter(
                threshold_db=kwargs.get("limiter_threshold", -6),
                release_ms=kwargs.get("limiter_release", 0.05),
            )
            board.append(limiter)
        if kwargs.get("gain", False):
            gain = Gain(gain_db=kwargs.get("gain_db", 0))
            board.append(gain)
        if kwargs.get("distortion", False):
            distortion = Distortion(drive_db=kwargs.get("distortion_gain", 25))
            board.append(distortion)
        if kwargs.get("chorus", False):
            chorus = Chorus(
                rate_hz=kwargs.get("chorus_rate", 1.0),
                depth=kwargs.get("chorus_depth", 0.25),
                centre_delay_ms=kwargs.get("chorus_delay", 7),
                feedback=kwargs.get("chorus_feedback", 0.0),
                mix=kwargs.get("chorus_mix", 0.5),
            )
            board.append(chorus)
        if kwargs.get("bitcrush", False):
            bitcrush = Bitcrush(bit_depth=kwargs.get("bitcrush_bit_depth", 8))
            board.append(bitcrush)
        if kwargs.get("clipping", False):
            clipping = Clipping(threshold_db=kwargs.get("clipping_threshold", 0))
            board.append(clipping)
        if kwargs.get("compressor", False):
            compressor = Compressor(
                threshold_db=kwargs.get("compressor_threshold", 0),
                ratio=kwargs.get("compressor_ratio", 1),
                attack_ms=kwargs.get("compressor_attack", 1.0),
                release_ms=kwargs.get("compressor_release", 100),
            )
            board.append(compressor)
        if kwargs.get("delay", False):
            delay = Delay(
                delay_seconds=kwargs.get("delay_seconds", 0.5),
                feedback=kwargs.get("delay_feedback", 0.0),
                mix=kwargs.get("delay_mix", 0.5),
            )
            board.append(delay)

        return board

    def realloc(
        self,
        block_frame: int,
        extra_frame: int,
        crossfade_frame: int,
        sola_search_frame: int,
    ):
        block_frame_16k = int(block_frame / AUDIO_SAMPLE_RATE * self.sample_rate)
        crossfade_frame_16k = int(
            crossfade_frame / AUDIO_SAMPLE_RATE * self.sample_rate
        )
        sola_search_frame_16k = int(
            sola_search_frame / AUDIO_SAMPLE_RATE * self.sample_rate
        )
        extra_frame_16k = int(extra_frame / AUDIO_SAMPLE_RATE * self.sample_rate)

        convert_size_16k = (
            block_frame_16k
            + sola_search_frame_16k
            + extra_frame_16k
            + crossfade_frame_16k
        )
        if (
            modulo := convert_size_16k % self.window_size
        ) != 0:
            convert_size_16k = convert_size_16k + (self.window_size - modulo)
        self.convert_feature_size_16k = convert_size_16k // self.window_size

        self.skip_head = extra_frame_16k // self.window_size
        self.return_length = self.convert_feature_size_16k - self.skip_head
        self.silence_front = (
            extra_frame_16k - (self.window_size * 5) if self.silence_front else 0
        )
        
        audio_buffer_size = block_frame_16k + crossfade_frame_16k
        self.audio_buffer = np.zeros(audio_buffer_size, dtype=self.dtype)
        
        self.convert_buffer = np.zeros(convert_size_16k, dtype=self.dtype)
        
        self.pitch_buffer = np.zeros(
            self.convert_feature_size_16k + 1, dtype=np.int64
        )
        self.pitchf_buffer = np.zeros(
            self.convert_feature_size_16k + 1, dtype=self.dtype
        )

    def inference(
        self,
        audio_input: np.ndarray,
        f0_up_key: int = 0,
        index_rate: float = 0.5,
        protect: float = 0.5,
        volume_envelope: float = 1,
        f0_autotune: bool = False,
        f0_autotune_strength: float = 1,
        proposed_pitch: bool = False,
        proposed_pitch_threshold: float = 155.0,
    ):
        if self.pipeline is None:
            raise RuntimeError("Pipeline is not initialized.")

        # Resample to 16k
        audio_input_16k = librosa.resample(audio_input, orig_sr=AUDIO_SAMPLE_RATE, target_sr=self.sample_rate)
        
        circular_write(audio_input_16k, self.audio_buffer)

        vol_t = np.sqrt(np.square(self.audio_buffer).mean())
        vol = max(float(vol_t), 0)

        if self.vad is not None:
            is_speech = self.vad.is_speech(audio_input_16k)
            if not is_speech:
                # Run through pipeline to keep states but return silence? 
                # Or just return silence.
                # To maintain consistent behavior with original which keeps running to avoid lag spikes
                audio_model = self.pipeline.voice_conversion(
                    self.convert_buffer,
                    self.pitch_buffer,
                    self.pitchf_buffer,
                    f0_up_key,
                    index_rate,
                    self.convert_feature_size_16k,
                    self.silence_front,
                    self.skip_head,
                    self.return_length,
                    protect,
                    volume_envelope,
                    f0_autotune,
                    f0_autotune_strength,
                    proposed_pitch,
                    proposed_pitch_threshold,
                    self.reduced_noise,
                    self.board,
                )
                out_len = audio_model.shape[0] if hasattr(audio_model, 'shape') else 0
                return np.zeros(out_len, dtype=self.dtype), vol

        if vol < self.input_sensitivity:
            audio_model = self.pipeline.voice_conversion(
                self.convert_buffer,
                self.pitch_buffer,
                self.pitchf_buffer,
                f0_up_key,
                index_rate,
                self.convert_feature_size_16k,
                self.silence_front,
                self.skip_head,
                self.return_length,
                protect,
                volume_envelope,
                f0_autotune,
                f0_autotune_strength,
                proposed_pitch,
                proposed_pitch_threshold,
                self.reduced_noise,
                self.board,
            )
            out_len = audio_model.shape[0] if hasattr(audio_model, 'shape') else 0
            return np.zeros(out_len, dtype=self.dtype), vol

        circular_write(audio_input_16k, self.convert_buffer)

        audio_model = self.pipeline.voice_conversion(
            self.convert_buffer,
            self.pitch_buffer,
            self.pitchf_buffer,
            f0_up_key,
            index_rate,
            self.convert_feature_size_16k,
            self.silence_front,
            self.skip_head,
            self.return_length,
            protect,
            volume_envelope,
            f0_autotune,
            f0_autotune_strength,
            proposed_pitch,
            proposed_pitch_threshold,
            self.reduced_noise,
            self.board,
        )
        
        audio_model_np = np.array(audio_model)
        audio_model_np = audio_model_np * vol # simple gain adjust if needed, or rely on pipeline volume

        # Resample back to 48k
        audio_out = librosa.resample(audio_model_np, orig_sr=self.pipeline.tgt_sr, target_sr=AUDIO_SAMPLE_RATE)
        
        return audio_out, vol

    def __del__(self):
        del self.pipeline


class VoiceChanger:
    def __init__(
        self,
        read_chunk_size: int,
        cross_fade_overlap_size: float,
        extra_convert_size: float,
        model_path: str = None,
        index_path: str = None,
        f0_method: str = "rmvpe",
        embedder_model: str = None,
        embedder_model_custom: str = None,
        silent_threshold: int = 0,
        vad_enabled: bool = False,
        vad_sensitivity: int = 3,
        vad_frame_ms: int = 30,
        sid: int = 0,
        clean_audio: bool = False,
        clean_strength: float = 0.5,
        post_process: bool = False,
        **kwargs,
    ):
        self.block_frame = read_chunk_size * 128
        self.crossfade_frame = int(cross_fade_overlap_size * AUDIO_SAMPLE_RATE)
        self.extra_frame = int(extra_convert_size * AUDIO_SAMPLE_RATE)
        self.sola_search_frame = AUDIO_SAMPLE_RATE // 100
        self.sola_buffer = None
        self.vc_model = Realtime(
            model_path,
            index_path,
            f0_method,
            embedder_model,
            embedder_model_custom,
            silent_threshold,
            vad_enabled,
            vad_sensitivity,
            vad_frame_ms,
            sid,
            clean_audio,
            clean_strength,
            post_process,
            **kwargs,
        )
        self.vc_model.realloc(
            self.block_frame,
            self.extra_frame,
            self.crossfade_frame,
            self.sola_search_frame,
        )
        self.generate_strength()

    def generate_strength(self):
        self.fade_in_window = (
            np.sin(
                0.5
                * np.pi
                * np.linspace(
                    0.0,
                    1.0,
                    num=self.crossfade_frame,
                    dtype=np.float32,
                )
            )
            ** 2
        )

        self.fade_out_window = 1 - self.fade_in_window
        self.sola_buffer = np.zeros(
            self.crossfade_frame, dtype=np.float32
        )

    def process_audio(
        self,
        audio_input: np.ndarray,
        f0_up_key: int = 0,
        index_rate: float = 0.5,
        protect: float = 0.5,
        volume_envelope: float = 1,
        f0_autotune: bool = False,
        f0_autotune_strength: float = 1,
        proposed_pitch: bool = False,
        proposed_pitch_threshold: float = 155.0,
    ):
        block_size = audio_input.shape[0]

        audio, vol = self.vc_model.inference(
            audio_input,
            f0_up_key,
            index_rate,
            protect,
            volume_envelope,
            f0_autotune,
            f0_autotune_strength,
            proposed_pitch,
            proposed_pitch_threshold,
        )
        
        # Ensure audio is numpy
        audio = np.array(audio)

        # SOLA (Overlap-Add)
        # Check audio length buffer validity
        min_len = self.crossfade_frame + self.sola_search_frame
        if audio.shape[0] < min_len:
             # Just pad with silence if model output is short (e.g. silence)
             audio = np.pad(audio, (0, min_len - audio.shape[0]))

        conv_input = audio[: self.crossfade_frame + self.sola_search_frame]
        
        # Correlation
        # signal.correlate(in1, in2, mode='valid') -> size N - M + 1 (N=conv_input, M=sola_buffer)
        cor_nom = scipy.signal.correlate(conv_input, self.sola_buffer, mode='valid')
        
        # Denom (Energy normalization)
        # cor_den = sqrt(conv(input^2, ones))
        input_sq = conv_input ** 2
        ones = np.ones(self.crossfade_frame)
        cor_den = scipy.signal.correlate(input_sq, ones, mode='valid')
        cor_den = np.sqrt(cor_den + 1e-8)
        
        # Normalized correlation
        # Check if cor_den == 0? (silence)
        # safe divide
        norm_corr = cor_nom / cor_den
        sola_offset = np.argmax(norm_corr)
        
        # Apply offset
        audio = audio[sola_offset:]
        
        # Crossfade
        if audio.shape[0] < self.crossfade_frame:
              audio = np.pad(audio, (0, self.crossfade_frame - audio.shape[0]))
              
        audio[: self.crossfade_frame] *= self.fade_in_window
        audio[: self.crossfade_frame] += self.sola_buffer * self.fade_out_window
        
        # Update buffer for next block
        # Buffer needs to be saved from the END of the current PROCESSED block
        # We want to return `block_size` samples.
        # But we generated more? 
        # Logic: 
        #   We took audio of shape ~ block + extra...
        #   We trimmed start by `sola_offset`.
        #   We crossfaded start.
        #   Now we output `block_size` samples.
        #   The remaining samples at the end are saved for next crossfade.
        
        # Wait, original logic:
        # self.sola_buffer[:] = audio[block_size : block_size + self.crossfade_frame]
        
        # Ensure we have enough data
        needed_end = block_size + self.crossfade_frame
        if audio.shape[0] < needed_end:
             audio = np.pad(audio, (0, needed_end - audio.shape[0]))
             
        self.sola_buffer[:] = audio[block_size : needed_end]
        
        return audio[:block_size], vol

    def on_request(
        self,
        audio_input: np.ndarray,
        f0_up_key: int = 0,
        index_rate: float = 0.5,
        protect: float = 0.5,
        volume_envelope: float = 1,
        f0_autotune: bool = False,
        f0_autotune_strength: float = 1,
        proposed_pitch: bool = False,
        proposed_pitch_threshold: float = 155.0,
    ):
        if self.vc_model is None:
            raise RuntimeError("Voice Changer is not selected.")

        start = time.perf_counter()
        result, vol = self.process_audio(
            audio_input,
            f0_up_key,
            index_rate,
            protect,
            volume_envelope,
            f0_autotune,
            f0_autotune_strength,
            proposed_pitch,
            proposed_pitch_threshold,
        )
        end = time.perf_counter()

        return result, vol, [0, (end - start) * 1000, 0]
