import numpy as np
import matplotlib.pyplot as plt
from commpy.filters import rrcosfilter

file_path = "test.bin"
dtype = np.complex64

sample = np.fromfile(file_path, dtype=dtype)

print(sample.shape)

sample_rate = 1e6
beta = 0.35
sps = 8
Ts = sps * (1 / sample_rate)
rrc_time, rrc = rrcosfilter(50, beta, Ts, sample_rate)

sample = sample[::2]

sample = np.convolve(sample, rrc)

plt.plot(sample.real)
plt.plot(sample.imag)
plt.show()
