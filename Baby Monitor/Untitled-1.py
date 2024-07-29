
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
# from microphone import record_audio # add if utilizing microphone and in Microphone directory
from IPython.display import Audio
from typing import Tuple
import librosa

from numba import njit
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure
from scipy.spatial.distance import cdist
from scipy.ndimage.morphology import iterate_structure

from typing import Tuple, Callable, List, Union

import uuid
import os
from pathlib import Path
from collections import Counter
import pickle

import wave, struct, librosa #important

SAMPLING_RATE = 44100

def load_audio_file(file_path: str):
    """Loads a target audio file path.

    Parameters
    ----------
    file_path : str
        File path of song
        
    Returns
    -------
    recorded_audio: np.ndarray
        Audio samples
    """
    audio, samp_rate = librosa.load(file_path, sr=SAMPLING_RATE, mono=True)
    return audio

def convert_mic_frames_to_audio(frames: np.ndarray) -> np.ndarray:
    """Converts frames taken from microphone to 16-bit integers
    
    Parameters
    ----------
    frames : np.ndarray
        List of bytes recorded from a microphone
        
    Returns
    -------
    numpy.ndarray
        Bytes converted to 16-bit integers
    """
    return np.hstack([np.frombuffer(i, np.int16) for i in frames])

def audio_info(file_path):
    """
    input: file path
    output: np array audio
    """
    with open(file_path,'rb') as audio_file:
        header = audio_file.read(44) # In WAV files, first 44 bytes are reserved for the header
        data_chunk_size = struct.unpack('<I', header[40:44])[0]
        audio_file.seek(44)
        data = audio_file.read(data_chunk_size)

        #Read the data from the file
        audio_file.seek(44)
        data = audio_file.read(data_chunk_size)

    # Converting the raw binary data to a list of integers : 
    data_array = np.frombuffer(data, dtype=np.int32)
    # Convert to float32
    data_array = data_array.astype(np.float32)

    return data_array

def spectogram_plot(file_name):
    """
    input: file path
    output: spectogram
    """
    data, SAMPLING_RATE = librosa.load(file_name, sr = SAMPLING_RATE)

    mel_spectrogram = librosa.feature.melspectrogram(y=data, sr=SAMPLING_RATE)

    # Log Mel Spectogram
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Visualize the spectrogram
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(log_mel_spectrogram, sr=SAMPLING_RATE, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Log Mel Spectrogram')
    plt.show()

def amplitude_plot(file_name):
    """
    input: file path
    output: amplitude to time plot
    """
    data, sr = librosa.load(file_name, sr = 44100)
    plt.subplot(1,1,1)
    librosa.display.waveshow(data,sr=sr, x_axis = 'time')
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Amplitude Waveform")
    plt.show()
