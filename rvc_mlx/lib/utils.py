import os
import sys
import librosa
import soundfile as sf
import numpy as np
import re
import unicodedata
import logging

# Minimal utils for MLX RVC

def load_audio_16k(file):
    try:
        audio, sr = librosa.load(file, sr=16000)
    except Exception as error:
        raise RuntimeError(f"An error occurred loading the audio: {error}")

    return audio.flatten()


def load_audio(file, sample_rate):
    try:
        file = file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        audio, sr = sf.read(file)
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1) # Mono
        if sr != sample_rate:
             # Use librosa or scipy
             import scipy.signal
             # Resample
             # calculated samples
             num = int(len(audio) * sample_rate / sr)
             audio = scipy.signal.resample(audio, num)
    except Exception as error:
        raise RuntimeError(f"An error occurred loading the audio: {error}")

    return audio.flatten()

def load_audio_infer(file, sample_rate, **kwargs):
    # Simplified version without formant shifting logic for now (requires stftpitchshift which is numpy based but complex to verify)
    return load_audio(file, sample_rate)

def format_title(title):
    formatted_title = unicodedata.normalize("NFC", title)
    formatted_title = re.sub(r"[\u2500-\u257F]+", "", formatted_title)
    formatted_title = re.sub(r"[^\w\s.-]", "", formatted_title, flags=re.UNICODE)
    formatted_title = re.sub(r"\s+", "_", formatted_title)
    return formatted_title
