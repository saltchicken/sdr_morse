from sdr_tools import classes
from sdr_tools import utils
import numpy as np

from scipy.signal import resample_poly, firwin, bilinear, lfilter
import matplotlib.pyplot as plt

# apply settings
sample_rate = 2e6
frequency = 433.5e6
antenna = 'LNAW'

buffer_size = 10000000

receiver = classes.Receiver(sample_rate, frequency, antenna, buffer_size)

frequency = -540000  # Adjust the frequency as needed
wave_gen = utils.cos_wave_generator(sample_rate, frequency, buffer_size)

received = receiver.read()

modulated = received * next(wave_gen)

# demodulated = 0.5 * np.angle(modulated[0:-1] * np.conj(modulated[1:]))

# taps = firwin(numtaps=101, cutoff=150e3, fs=sample_rate)
# modulated = np.convolve(modulated, taps, 'valid')

# Decimate by 10
# modulated = modulated[::20]
# plt.plot(filtered)
# plt.show()

output_file = 'samples.bin'
print(modulated.shape)
modulated.tofile(output_file)

receiver.close()