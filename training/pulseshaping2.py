import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from commpy.filters import rrcosfilter


def rrc(symrate, b, f):
    T = 1.0/symrate

    if f < (1.-b)/(2.0*T):
        return 1.
    elif f <= (1+b)/(2.0*T):
        return np.sqrt(0.5 * (1.0 - np.sin(np.pi * T * (f - 0.5/T)/b)))
    else:
        return 0.

plt.figure(figsize=(10, 8))
plots = 7

num_symbols = 8
sps = 16
sample_rate = 1e6

bits = np.random.randint(0, 2, num_symbols) # Our data to be transmitted, 1's and 0's

x = np.array([])
for bit in bits:
    pulse = np.zeros(sps)
    pulse[0] = bit*2-1 # set the first value to either a 1 or -1
    x = np.concatenate((x, pulse)) # add the 8 samples to the signal
plt.subplot(plots,1,1)
plt.plot(x, '.-')
plt.grid(True)
# plt.show()

num_taps = 101
beta = 0.35
Ts = sps * (1 / sample_rate) # Assume sample rate is 1 Hz, so sample period is 1, so *symbol* period is 8
t = np.arange(num_taps) - (num_taps-1)/2
t *= (1 / sample_rate)
print(t[0], ' ', t[-1])
print(len(t))
h = np.sinc(t/Ts) * np.cos(np.pi*beta*t/Ts) / (1 - (2*beta*t/Ts)**2)
plt.subplot(plots,1,2)
# plt.plot(t, h, '.')
plt.plot(h, '.')
plt.grid(True)
# plt.show()


h_root = np.where(h >= 0, np.sqrt(h), -np.sqrt(np.abs(h)))
# h_root = np.sqrt(h)
plt.subplot(plots,1,3)
# what = np.convolve(h_root, h_root)
what = np.where(h_root >= 0, h_root * h_root, -1 * h_root * h_root)
plt.plot(t, what, '.')
plt.grid(True)

# Filter our signal, in order to apply the pulse shaping
x_shaped = np.convolve(x, h)
plt.subplot(plots,1,4)
plt.plot(x_shaped, '.-')
for i in range(num_symbols):
    plt.plot([i*sps+num_taps//2,i*sps+num_taps//2], [0, x_shaped[i*sps+num_taps//2]])
plt.grid(True)

plt.subplot(plots,1,5)
x_shaped_root = np.convolve(x, h_root)
x_shaped = np.convolve(h_root, x_shaped_root)

plt.plot(x_shaped, '.-')
for i in range(num_symbols):
    plt.plot([i*sps+num_taps//2,i*sps+num_taps//2], [0, x_shaped[i*sps+num_taps//2]])
plt.grid(True)


plt.subplot(plots,1,6)

# rrc = rrc(Ts, beta, t)

rrc_time, rrc = rrcosfilter(50, 0.35, Ts, sample_rate)

what = np.convolve(rrc, rrc)
plt.plot(what)



plt.subplot(plots, 1, 7)
final = np.convolve(x, rrc)
final = np.convolve(final, rrc)
final = final / np.max(final) # Why am I normalizing?
plt.plot(final)





plt.show()