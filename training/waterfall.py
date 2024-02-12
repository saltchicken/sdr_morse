# import SoapySDR
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('dark_background')

num_freq_bins = 1000
num_time_bins = 100
waterfall_data = np.zeros((num_time_bins, num_freq_bins))
print(waterfall_data.shape)

for i in range(num_time_bins):
    color = np.zeros(num_freq_bins)
    color[0:len(color)//2] = 1.0
    waterfall_data[i, :] = color
    
plt.imshow(waterfall_data, aspect='auto')  # extent=[0, sample_rate / 1e3, 0, num_samples] ---- Also used LogNorm?
plt.xlabel('Frequency (kHz)')
plt.ylabel('Time')
plt.title('Waterfall Plot')
plt.colorbar(label='Amplitude')
plt.show()