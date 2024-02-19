from sdr_tools.classes import Receiver, Segment, QuadDemodSegment
from sdr_tools import utils
import numpy as np

from scipy.signal import resample_poly, firwin, bilinear, lfilter
import matplotlib.pyplot as plt

from IPython import embed

# apply settings
sample_rate = 2e6
frequency = 433.5e6
antenna = 'LNAW'

# buffer_size = 10000000

with Receiver(sample_rate, frequency, antenna) as receiver:
    
    plt.figure(figsize=(10, 8))
    segment = receiver.getSegment()
    # segment.display()
    plt.subplot(2, 2, 1)
    plt.plot(segment.data)
    segment.shift_center(540000)
    # segment.display()
    plt.subplot(2, 2, 2)
    plt.plot(segment.data)
    segment.low_pass_filter(10000)
    # segment.display()
    plt.subplot(2, 2, 3)
    plt.plot(segment.data)
    demod = QuadDemodSegment(segment)
    # demod.plot()
    plt.subplot(2, 2, 4)
    plt.plot(segment.data)
    demod.resample(1, 128000)
    plt.show()
    # demod.plot()
    # demod.decode()
    
    embed()
# receiver = classes.Receiver(sample_rate, frequency, antenna)

# frequency = -540000  # Adjust the frequency as needed
# wave_gen = utils.cos_wave_generator(sample_rate, frequency, buffer_size)
# wave_gen = utils.cos_wave_generator(sample_rate, frequency, len(receiver.read_buffer))

# received = receiver.read()
# modulated = received * next(wave_gen)

# Testing out demodulation
# demodulated = 0.5 * np.angle(modulated[0:-1] * np.conj(modulated[1:]))

#TEsting different way for low pass filter
# taps = firwin(numtaps=101, cutoff=150e3, fs=sample_rate)
# modulated = np.convolve(modulated, taps, 'valid')

# output_file = 'samples.bin'
# print(modulated.shape)
# modulated.tofile(output_file)