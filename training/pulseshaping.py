import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


num_symbols = 10
sps = 8
num_taps = 101
beta = 0.35
Ts = sps # Assume sample rate is 1 Hz, so sample period is 1, so *symbol* period is 8
# t = np.arange(num_taps) - (num_taps-1)//2
t = np.arange(0,100)
print(len(t))
# h = 1/Ts*np.sinc(t/Ts) * np.cos(np.pi*beta*t/Ts) / (1 - (2*beta*t/Ts)**2)

h = np.sinc(t/Ts)

freq_domain = np.fft.fftshift(np.fft.fft(h, n=len(h)))
plt.subplot(2,1,1)
plt.plot(t, h, '.')

plt.subplot(2,1,2)
plt.plot(freq_domain)
plt.grid(True)
plt.show()