import numpy as np
import matplotlib.pyplot as plt

# Parameters
frequency = 1.0  # Frequency of the cosine wave
amplitude = 1.0  # Amplitude of the cosine wave
phase = np.pi / 4  # Phase shift of the cosine wave

# Sampling settings
sampling_rate = 1000  # Number of samples per second
duration = 2.0  # Duration of the signal in seconds
num_samples = int(sampling_rate * duration)  # Total number of samples

# Time array
t = np.linspace(0, duration, num_samples)

# Generate the complex cosine wave
complex_cosine_wave = amplitude * np.exp(1j * (2 * np.pi * frequency * t + phase))

# Plot the real and imaginary parts
plt.plot(t, complex_cosine_wave.real, label='Real')
plt.plot(t, complex_cosine_wave.imag, label='Imaginary')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Complex Cosine Wave')
plt.legend()
plt.grid(True)
plt.show()
