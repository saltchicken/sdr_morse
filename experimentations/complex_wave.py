import numpy as np
import matplotlib.pyplot as plt

# plt.style.use('dark_background')

# Parameters
frequency = 1.0  # Frequency of the cosine wave
amplitude = 1.0  # Amplitude of the cosine wave
phase = -np.pi / 2  # Phase shift of the cosine wave

# Sampling settings
sampling_rate = 1000  # Number of samples per second
duration = 2.0  # Duration of the signal in seconds
num_samples = int(sampling_rate * duration)  # Total number of samples

# Time array
t = np.linspace(0, duration, num_samples)

fig, (ax1, ax2) = plt.subplots(2)

# Generate the complex cosine wave
complex_wave = np.zeros(len(t), dtype=np.complex64)
complex_wave.real = amplitude * np.cos((2 * np.pi * frequency * t))
complex_wave.imag = amplitude * np.cos((2 * np.pi * frequency * t + phase))

complex_wave_original = amplitude * np.exp(1j * (2 * np.pi * frequency * t), dtype=np.complex64)

print(complex_wave.dtype)
print(complex_wave_original.dtype)




# Plot the real and imaginary parts
ax1.plot(t, complex_wave.real, label='Real', color='blue')
ax1.plot(t, complex_wave.imag, label='Imaginary', color='red')
# plt.plot(t, complex_wave)
ax2.plot(t, complex_wave_original.real, label='Real', color='blue')
ax2.plot(t, complex_wave_original.imag, label='Imaginary', color='red')
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.title('Complex Cosine Wave')
# plt.legend()
ax1.grid(True)
plt.show()
