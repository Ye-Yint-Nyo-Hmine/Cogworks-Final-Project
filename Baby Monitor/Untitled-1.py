
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

def audio_file_to_array(file_path):
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
    data_array = data_array.astype(np.int16)

    return data_array #type int16

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

# so audios and recorded noise should both be int16

def dig_samp_to_spec(samples: np.ndarray):
    # using matplotlib's built-in spectrogram function

    S, freqs, times = mlab.specgram(
        samples,
        NFFT=4096,
        Fs=SAMPLING_RATE,
        window=mlab.window_hanning,
        noverlap=int(4096 / 2),
        mode='magnitude'
    )
    
    return S

@njit
def _peaks(data_2d, rows, cols, amp_min):
    """
    A Numba-optimized 2-D peak-finding algorithm.
    
    Parameters
    ----------
    data_2d : numpy.ndarray, shape-(H, W)
        The 2D array of data in which local peaks will be detected.

    rows : numpy.ndarray, shape-(N,)
        The 0-centered row indices of the local neighborhood mask
    
    cols : numpy.ndarray, shape-(N,)
        The 0-centered column indices of the local neighborhood mask
        
    amp_min : float
        All amplitudes at and below this value are excluded from being local 
        peaks.
    
    Returns
    -------
    List[Tuple[int, int]]
        (row, col) index pair for each local peak location. 
    """
    peaks = []
    
    # iterate over the 2-D data in col-major order
    for c, r in np.ndindex(*data_2d.shape[::-1]):
        if data_2d[r, c] <= amp_min:
            continue

        for dr, dc in zip(rows, cols):
            if dr == 0 and dc == 0:
                continue

            if not (0 <= r + dr < data_2d.shape[0]):
                dr *= -1

            if not (0 <= c + dc < data_2d.shape[1]):
                dc *= -1

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

### Turn peaks to fingerprints ###
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

def local_peaks_to_fingerprints_with_absolute_times(local_peaks: List[Tuple[int, int]], num_fanout: int):
    """Returns the fingerprint and absolute time of the fingerprint of a set of peaks.

    Parameters
    ----------
    local_peaks : List[Tuple[int, int]]
        List of row, column (frequency, time) indexes of the peaks

    num_fanout : int
         Number of fanout points for each reference point

    Returns
    -------
    List[Tuple[int, int, int]] contained the reference point frequency, 
    fanout term frequency, and change in time interval, and List[int] of the abs_times of fingerprints."""

    fingerprints = []
    abs_times = []
    
    if num_fanout <= len(local_peaks):
        for i in range(len(local_peaks) - num_fanout): # subtract because it had to be only peaks after, and dont want index out of bounds error
            i_freq, i_time = local_peaks[i]
            for j in range(1, num_fanout+1):
                f_freq, f_time = local_peaks[i+j]
                fingerprints.append((i_freq, f_freq, f_time - i_time))
                abs_times.append(i_time)
            
        
        return fingerprints, abs_times
    else:
        return "IndexError"

def file_path_to_fingerprints(file_path, amplitude_percentile: float=0.75, fanout_number: int=15):
    """Take the music file path of a song and returns it's fingerprints.

    Parameters
    ----------
    file_path : Union[str, Path]
        File path for music file

    amplitude_percentile : float, optional
         A demical < 1.0 for which all amplitudes less than the {percentile}
         percentile of amplitudes will be disregarded

    fanout_number: int, optional
        Number of fanouts for each reference point/peak in the spectrogram

    Returns
    -------
    List[Tuple[int, int, int]] contained the reference point frequency, 
    fanout term frequency, and change in time interval"""

    samples = load_audio_file(file_path)

    S = dig_samp_to_spec(samples)

    neighborhood = iterate_structure(generate_binary_structure(2, 1), 20)

    peak_locations = local_peak_locations(S, neighborhood, amp_min=find_cutoff_amp(S, amplitude_percentile))

    fingerprints = local_peaks_to_fingerprints(peak_locations, fanout_number)
   
    return fingerprints

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

#def process_all_songs(directory_path: str, num_fanout: int = 15):
def process_all_songs(directory_path: str): #"/data"
    fingerprints = []
    
    for i, filename in enumerate(os.listdir(directory_path)):
        file_path = os.path.join(directory_path, filename)
        fingerprint = file_path_to_fingerprints(file_path)
        fingerprints.append(fingerprint)