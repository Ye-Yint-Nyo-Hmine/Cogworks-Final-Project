#!/usr/bin/env python
# coding: utf-8

# In[80]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
# from microphone import record_audio # add if utilizing microphone and in Microphone directory
from IPython.display import Audio
from typing import Tuple
import librosa
import statistics as stats

from numba import njit
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure
from scipy.spatial.distance import cdist
from scipy.ndimage.morphology import iterate_structure
from microphone import record_audio

from typing import Tuple, Callable, List, Union

import uuid
import os
from pathlib import Path
from collections import Counter
import pickle

import wave, struct, librosa #importan
from scipy.io import wavfile
import time

# hyperparams
CUTOFF_AMP = 0.77 #percentile
NUM_FANOUT = 15
LISTEN_TIME = 7.5 #sec
CUTOFF_SIM = 0.8 # how similar to be considered baby crying
DATA_DIRECTORY = "data/crying/"


# In[81]:


def process_all_audio(directory_path, num_fanout: int = 15) -> np.ndarray:
    db = []
    
    for filename in os.listdir(directory_path):
        if filename.endswith(".wav"):
            file_path = os.path.join(directory_path, filename)

            samplerate, audio = load_audio_file(file_path)
            
            samples = convert_mic_frames_to_audio(audio)
            S = dig_samp_to_spec(samples)

            # Define neighborhood structure for peak detection
            neighborhood = generate_binary_structure(2, 1)
            neighborhood = iterate_structure(neighborhood, 20)

            # Detect peaks
            amp_min = find_cutoff_amp(S, CUTOFF_AMP)
            peaks = local_peak_locations(S, neighborhood, amp_min)

            # Generate fingerprints
            fingerprints = local_peaks_to_fingerprints(peaks, num_fanout) #list
            db.append(fingerprints) # list in list
    
    return np.array(fingerprints)

def audio_to_fingerprints(audio): #from load_audio_file
    samples = convert_mic_frames_to_audio(audio)
    S = dig_samp_to_spec(samples)

    # Define neighborhood structure for peak detection
    neighborhood = generate_binary_structure(2, 1)
    neighborhood = iterate_structure(neighborhood, 20)

    # Detect peaks
    amp_min = find_cutoff_amp(S, 0.77)
    peaks = local_peak_locations(S, neighborhood, amp_min)

    # Generate fingerprints
    fingerprints = local_peaks_to_fingerprints(peaks, NUM_FANOUT) #list
    return fingerprints

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
    samplerate, audio = wavfile.read(file_path)
    # audio = audio.astype(np.int16)     audio data converts to zeros
    return samplerate, audio

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

def record(listen_time):
    frames, samplerate = record_audio(listen_time)
    audio = convert_mic_frames_to_audio(frames)
    return audio, samplerate

def dig_samp_to_spec(samples: np.ndarray):
    """Takes a 1-D sampled audio array and returns a 2-D spectrogram."""
    
    fig, ax = plt.subplots()

    S, freqs, times, im = ax.specgram(
    samples,
    NFFT=4096,
    Fs=SAMPLING_RATE,
    window=mlab.window_hanning,
    noverlap=4096 // 2,
    mode='magnitude'
    )
    
    return S

def find_cutoff_amp(S: np.ndarray, percentile: float):
    """Returns the log_amplitude of a target spectrogram that will be the cutoff for background noise
       in real world samples. Calculated using decimal part percentile.

    Parameters
    ----------
    S : numpy.ndarray
        The target spectrogram

    percentile : float
         A demical < 1.0 for which the cutoff is greater than or equal to the {percentile}
         percentile of log_amplitudes

    Returns
    -------
    Cutoff amplitude"""

    S = S.ravel()  # ravel flattens 2D spectrogram into a 1D array
    ind = round(len(S) * percentile)  # find the index associated with the percentile amplitude
    cutoff_amplitude = np.partition(S, ind)[ind]  # find the actual percentile amplitude
    
    return cutoff_amplitude

@njit
def _peaks(data_2d, rows, cols, amp_min):
    peaks = []

    for c, r in np.ndindex(*data_2d.shape[::-1]):
        if data_2d[r, c] <= amp_min:
            continue
        for dr, dc in zip(rows, cols):
            if dr == 0 and dc == 0:
                continue

            if not (0 <= r + dr < data_2d.shape[0]):
                continue
            if not (0 <= c + dc < data_2d.shape[1]):
                continue

            if data_2d[r, c] < data_2d[r + dr, c + dc]:
                break
        else:
            peaks.append((r, c))
    return peaks

def local_peak_locations(data_2d, neighborhood, amp_min):
    """
    From 
    Defines a local neighborhood and finds the local peaks
    in the spectrogram, which must be larger than the specified `amp_min`.
    
    Parameters
    ----------
    data_2d : numpy.ndarray, shape-(H, W)
        The 2D array of data in which local peaks will be detected
    
    neighborhood : numpy.ndarray, shape-(h, w)
        A boolean mask indicating the "neighborhood" in which each
        datum will be assessed to determine whether or not it is
        a local peak. h and w must be odd-valued numbers
        
    amp_min : float
        All amplitudes at and below this value are excluded from being local 
        peaks.
    
    Returns
    -------
    List[Tuple[int, int]]
        (row, col) index pair for each local peak location.
    
    Notes
    -----
    The local peaks are returned in column-major order.
    """
    rows, cols = np.where(neighborhood)
    assert neighborhood.shape[0] % 2 == 1
    assert neighborhood.shape[1] % 2 == 1

    rows -= neighborhood.shape[0] // 2
    cols -= neighborhood.shape[1] // 2
    
    return _peaks(data_2d, rows, cols, amp_min=amp_min)

def local_peaks_to_fingerprints(local_peaks: List[Tuple[int, int]], num_fanout: int):
    """Returns the fingerprint a set of peaks packaged as a tuple.

    Parameters
    ----------
    local_peaks : List[Tuple[int, int]]
        List of row, column (frequency, time) indexes of the peaks

    num_fanout : int
         Number of fanout points for each reference point

    Returns
    -------
    List[Tuple[int, int, int]]
        List of fingerprints"""
    
    result = [] #should be a list of lists

    if num_fanout <= len(local_peaks):
        for i in range(len(local_peaks) - num_fanout): # subtract because it had to be only peaks after, and dont want index out of bounds error
            i_fingerprints = []
            i_freq, i_time = local_peaks[i]
            for j in range(1, num_fanout+1):
                f_freq, f_time = local_peaks[i+j]
                i_fingerprints.append((i_freq, f_freq, f_time - i_time))
            
            result += i_fingerprints # contatenate lists
        
        return result # should be a 2d list, that can then be zipped w the peaks if we need to know which peak its associated with
    else:
        return "IndexError"


# In[ ]:


def make_db(): #already done
    bob = process_all_audio("data/crying/")
    outfile = "db.npy"
    np.save(outfile, bob)


# In[87]:


def check_similarity(frames):
    fingerprints = audio_to_fingerprints(frames)

    similarities=[]
    for audio in db:
        match=0
        for fp in audio:
            if fp in fingerprints:
                match+=1
        similarities.append(match/len(fingerprints)) # % of fingerprints in recorded audio are in crying audio

    avg_sim = stats.mean(similarities) # should be 0-1
    
    if avg_sim >= CUTOFF_SIM or 1 in similarities: # adjust if needed
        print("baby crying")
    else:
        print("NOT A BABY CRYING D:")


# In[88]:


import random
def test_on_actual_baby_crying():
    samplerate, frames = load_audio_file(DATA_DIRECTORY+random.choice(os.listdir(DATA_DIRECTORY)))
    check_similarity(frames)

