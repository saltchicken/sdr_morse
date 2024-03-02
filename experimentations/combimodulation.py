import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

plt.style.use('dark_background')

# Define parameters
frequency = 2
second_frequency = 3
num_samples = 1000
sample_number = 200

# Calculate time corresponding to sample 50
time_interval = 1 / num_samples
sample_time = time_interval * sample_number

# Calculate phase shift at sample 50
phase_shift = 2 * np.pi * frequency * sample_time

# Generate time values for the cosine wave
t = np.linspace(0, 1, num_samples)

# Generate cosine wave
first_cos_wave = np.cos(2 * np.pi * frequency * t)
second_cos_wave = np.cos(2 * np.pi * second_frequency * t + phase_shift)

place_holder = np.zeros(1000)
place_holder[0:sample_number] = second_cos_wave[0]
place_holder[sample_number:1000] = second_cos_wave[0:-sample_number]

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)
line1, = plt.plot(first_cos_wave)
line2, = plt.plot(place_holder)

ax_sample_shift = plt.axes([0.2, 0.1, 0.65, 0.03])
slider_sample_shift = Slider(ax_sample_shift, 'Sample Number', 0, num_samples, valinit=sample_number, valstep=1)

def update(val):
    sample_number = val
    sample_time = time_interval * sample_number
    phase_shift = 2 * np.pi * frequency * sample_time
    second_cos_wave = np.cos(2 * np.pi * second_frequency * t + phase_shift)
    place_holder = np.zeros(1000)
    place_holder[0:sample_number] = second_cos_wave[0]
    place_holder[sample_number:1000] = second_cos_wave[0:-sample_number]
    line2.set_ydata(place_holder)
    fig.canvas.draw_idle()
    
slider_sample_shift.on_changed(update)
plt.show()