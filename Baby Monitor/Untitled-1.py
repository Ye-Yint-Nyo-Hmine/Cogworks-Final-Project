
import numpy as np
import librosa
import matplotlib.pyplot as plt
from IPython.display import Audio
from typing import Union, Callable, Tuple
from pathlib import Path

SAMPLING_RATE=41000
audio, samp_rate = librosa.load(file_path, sr=SAMPLING_RATE, mono=True)

