import numpy as np

file_path = "samplesExact.bin"
output_path = "samplesExact3.bin"
dtype = np.complex64

sample = np.fromfile(file_path, dtype=dtype)

print(sample.shape)

# cropped = sample[43000:len(sample) - 93000]
# # print(cropped.shape)

# cropped.tofile(output_path)