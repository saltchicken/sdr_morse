import numpy as np
from scipy import signal

import matplotlib.pyplot as plt
plt.style.use('dark_background')

def plot(data):
    plt.plot(data)
    plt.show()

def generate_carrier(streamer, frequency, duration):
    t = np.arange(0, duration, 1 / streamer.sample_rate)
    carrier_signal = np.cos(2 * np.pi * frequency * t)
    return carrier_signal

def generate_fm_packet(binary_string, frequency, second_frequency, duration, sample_rate):
    t = np.arange(0, duration, 1 / sample_rate)
    num_symbols = len(binary_string)
    symbol_length = len(t) / num_symbols
    assert int(symbol_length) == symbol_length, "Sample amount of t must be divisible by num_symbols"
    symbol_length = int(symbol_length)
    print("Num symbols: ", num_symbols, "|", "Symbol length: ", symbol_length)
    transmission_signal = np.zeros(len(t), dtype=np.complex64)
    time_interval = 1 / sample_rate
    
    # TODO: Make more efficient. Calc phase shift right after symbol wave. Use 'np.exp()'
    for i, bit in enumerate(binary_string):
        start_index = symbol_length * i
        end_index = start_index + symbol_length
        symbol_time = symbol_length * time_interval
        if i == 0:
            phase_shift = 0.0
        elif binary_string[i-1] == '0':
            phase_shift += 2 * np.pi * frequency * symbol_time
        elif binary_string[i-1] == '1':
            phase_shift += 2 * np.pi * second_frequency * symbol_time
        else:
            print("Something is wrong with calculating phase shift")
        if bit == '0':
            symbol_wave_real = np.cos(2 * np.pi * frequency * t + phase_shift)
            symbol_wave_imag = np.cos(2 * np.pi * frequency * t + (phase_shift - (np.pi / 2)))
        elif bit == '1':
            symbol_wave_real = np.cos(2 * np.pi * second_frequency * t + phase_shift)
            symbol_wave_imag = np.cos(2 * np.pi * second_frequency * t + (phase_shift - (np.pi / 2)))
        else:
            print("Something is wrong with the binary_string")
        transmission_signal.real[start_index:end_index] = symbol_wave_real[0:symbol_length]
        transmission_signal.imag[start_index:end_index] = symbol_wave_imag[0:symbol_length]
    return transmission_signal

def cos_wave_generator(sample_rate, frequency, samples):
    i = 0
    while True:
        t = (np.arange(samples) + i * samples) / sample_rate
        yield np.exp(1j * 2 * np.pi * frequency * t).astype(np.complex64)
        i += 1

def peak_freq(sample, sample_rate):
        freq_domain = np.fft.fftshift(np.fft.fft(sample))
        frequencies = np.fft.fftshift(np.fft.fftfreq(len(sample), 1/sample_rate))
        max_magnitude_index = np.argmax(np.abs(freq_domain))
        center_freq = frequencies[max_magnitude_index]
        return center_freq

def low_pass_filter(sample, sample_rate, cutoff_frequency, filter_order=5):
        nyquist_frequency = sample_rate / 2
        normalized_cutoff_frequency = cutoff_frequency / nyquist_frequency
        b, a = signal.butter(filter_order, normalized_cutoff_frequency, btype='low')
        filtered = signal.lfilter(b, a, sample)
        filtered = filtered.astype(np.complex64)
        assert sample.dtype == filtered.dtype, "Output of filtered signal mismatched with sample signal"
        return filtered
    
def quad_demod(sample):
    return 0.5 * np.angle(sample[:-1] * np.conj(sample[1:]))

def downsample(sample, downsample_rate):
    return sample[::downsample_rate]

def resample(sample, interpolation, decimation):
    return signal.resample_poly(sample, interpolation, decimation)#interpolation == upsample, decimation == downsample

def decode(sample):
    return (np.real(sample) < 0).astype(int) # Why is real needed

# TODO: There may be an issue with calling this multiple times
def modulate_shift(sample, sample_rate, frequency, buffer_size):
    wave_gen = cos_wave_generator(sample_rate, frequency, buffer_size)
    return sample * next(wave_gen)

def display(sample, sample_rate, buffer_size=1024, fft_size=None):
    if fft_size == None:
        fft_size = buffer_size
    iterations = len(sample) // buffer_size # This needs to be even
    waterfall_data = np.zeros((iterations, fft_size))
    for i, buffer in enumerate(sample.reshape(iterations, fft_size)):
        freq_domain = np.fft.fftshift(np.fft.fft(buffer, n=fft_size))
        max_magnitude_index = np.abs(freq_domain)
        waterfall_data[i, :] = max_magnitude_index
    
    freq_range = sample_rate / 2000 # Half sample_rate and convert to kHz
    sample_time = buffer_size * iterations / sample_rate
    plt.figure(figsize=(12, 10))
    plt.imshow(waterfall_data, extent=[-freq_range, freq_range, 0, sample_time], aspect='auto')
    # plt.imshow(waterfall_data, aspect='auto')  # extent=[0, sample_rate / 1e3, 0, num_samples] ---- Also used LogNorm?
    plt.xlabel('Frequency (kHz)')
    plt.ylabel('Time (s)')
    plt.title('Waterfall Plot')
    plt.colorbar(label='Amplitude')
    plt.show()

def display_sample(receiver, iterations=1000, buffer_size=1024, fft_size=None):
    receiver.set_buffer_size(buffer_size)
    if fft_size == None:
        fft_size = buffer_size
    samples = []
    waterfall_data = np.zeros((iterations, fft_size))
    for i in range(iterations):
            sample = np.copy(receiver.read())
            samples.append(sample)
            freq_domain = np.fft.fftshift(np.fft.fft(sample, n=fft_size))
            max_magnitude_index = np.abs(freq_domain)
            waterfall_data[i, :] = max_magnitude_index
    result = np.concatenate(samples)
    
    freq_range = receiver.sample_rate / 2000 # Half sample_rate and convert to kHz
    sample_time = buffer_size * iterations / receiver.sample_rate
    plt.figure(figsize=(12, 10))
    plt.imshow(waterfall_data, extent=[-freq_range, freq_range, 0, sample_time], aspect='auto')
    # plt.imshow(waterfall_data, aspect='auto')  # extent=[0, sample_rate / 1e3, 0, num_samples] ---- Also used LogNorm?
    plt.xlabel('Frequency (kHz)')
    plt.ylabel('Time (s)')
    plt.title('Waterfall Plot')
    plt.colorbar(label='Amplitude')
    plt.show()
    
    return result            
    