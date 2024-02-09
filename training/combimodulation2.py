import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

plt.style.use('dark_background')

def generate_transmission_signal(frequency, second_frequency, t, phase_shift, num_samples, sample_number):
    transmission_signal = np.cos(2 * np.pi * frequency * t)
    second_cos_wave = np.cos(2 * np.pi * second_frequency * t + phase_shift)
    transmission_signal[sample_number:num_samples] = second_cos_wave[0:num_samples - sample_number]
    return transmission_signal
    
# Define parameters
frequency = 4
second_frequency = 20
num_samples = 2000
sample_number = 200

# Calculate time corresponding to sample 50
time_interval = 1 / num_samples
sample_time = time_interval * sample_number

# Calculate phase shift at sample 50
phase_shift = 2 * np.pi * frequency * sample_time

# Generate time values for the cosine wave
t = np.linspace(0, 1, num_samples)

# Generate cosine wave
transmission_signal = generate_transmission_signal(frequency, second_frequency, t, phase_shift, num_samples, sample_number)

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)
line1, = plt.plot(transmission_signal)

ax_sample_shift = plt.axes([0.2, 0.1, 0.65, 0.03])
slider_sample_shift = Slider(ax_sample_shift, 'Sample Number', 0, num_samples, valinit=sample_number, valstep=1)

def update(val):
    sample_number = val
    sample_time = time_interval * sample_number
    phase_shift = 2 * np.pi * frequency * sample_time
    transmission_signal = generate_transmission_signal(frequency, second_frequency, t, phase_shift, num_samples, sample_number)
    line1.set_ydata(transmission_signal)
    fig.canvas.draw_idle()
    
slider_sample_shift.on_changed(update)
plt.show()