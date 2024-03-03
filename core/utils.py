import numpy as np
from scipy import signal

import matplotlib.pyplot as plt
plt.style.use('dark_background')

from dataclasses import dataclass

@dataclass
class NodeMessage():
    type: str
    id: str
    
@dataclass
class FM_Settings:
    sample_rate: int = int(2e6)
    freq_deviation: int = 10000
    symbol_length: int = 10000 

def plot(data):
    plt.plot(data)
    plt.show()

# TODO: This is not being used.
def generate_carrier(streamer, frequency, duration):
    t = np.arange(0, duration, 1 / streamer.sample_rate)
    carrier_signal = np.cos(2 * np.pi * frequency * t)
    return carrier_signal

def peak_freq(sample, sample_rate):
        freq_domain = np.fft.fftshift(np.fft.fft(sample))
        frequencies = np.fft.fftshift(np.fft.fftfreq(len(sample), 1/sample_rate))
        max_magnitude_index = np.argmax(np.abs(freq_domain))
        center_freq = frequencies[max_magnitude_index]
        return center_freq