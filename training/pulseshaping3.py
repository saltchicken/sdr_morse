import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from commpy.filters import rrcosfilter
from scipy.signal import resample_poly

plt.figure(figsize=(10, 8))

sample_rate = 2e6
sps = 8
# sps is samples per symbol
def generate_pulse_from_bits(bits, sps):
    x = np.array([])
    for bit in bits:
        pulse = np.zeros(sps)
        pulse[0] = bit*2-1 # set the first value to either a 1 or -1
        x = np.concatenate((x, pulse)) # add the 8 samples to the signal
    return x

plt.subplot(4,1,1)
pulse = generate_pulse_from_bits([1,0,0,1,0,0,0,1], sps)
pulse = pulse.astype(np.complex64)
plt.plot(pulse.real, '.-')
plt.plot(pulse.imag, '.-')

plt.subplot(4,1,2)
beta = 0.35
Ts = sps * (1 / sample_rate)
rrc_time, rrc = rrcosfilter(50, beta, Ts, sample_rate)
plt.plot(rrc)

plt.subplot(4,1,3)
tx_signal = np.convolve(pulse, rrc)
plt.plot(tx_signal.real)
plt.plot(tx_signal.imag)

plt.subplot(4,1,4)
interpolated_tx_signal = resample_poly(tx_signal, 10000, 1)
plt.plot(interpolated_tx_signal.real)
plt.plot(interpolated_tx_signal.imag)




plt.show()