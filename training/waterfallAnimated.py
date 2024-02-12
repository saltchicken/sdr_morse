# import SoapySDR
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation

plt.style.use('dark_background')

num_freq_bins = 1000
num_time_bins = 100
waterfall_data = np.zeros((num_time_bins, num_freq_bins))
print(waterfall_data.shape)

for i in range(num_time_bins):
    color = np.zeros(num_freq_bins)
    color[0:len(color)//2] = 1.0
    waterfall_data[i, :] = color
    
fig, ax = plt.subplots()
im = ax.imshow(waterfall_data, cmap='viridis')
    
ax.imshow(waterfall_data, aspect='auto')  # extent=[0, sample_rate / 1e3, 0, num_samples] ---- Also used LogNorm?
ax.set_xlabel('Frequency (kHz)')
ax.set_ylabel('Time')
ax.set_title('Waterfall Plot')
fig.colorbar(im, label='Amplitude')

def update_image(frame):
    # Update image data with new random values
    waterfall_data = np.zeros((num_time_bins, num_freq_bins))

    for i in range(num_time_bins):
        color = np.zeros(num_freq_bins)
        color[0:len(color)//2] = 1.0 - (0.05 * frame)
        waterfall_data[i, :] = color
    im.set_array(waterfall_data)
    # Return the updated image
    return im,

interval = 1000  # milliseconds
ani = FuncAnimation(fig, update_image, interval=interval, blit=True)

plt.show()