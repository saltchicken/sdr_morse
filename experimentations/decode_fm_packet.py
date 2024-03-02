import numpy as np
from scipy import signal

def decode_fm_packet(sample_file, sample_rate, cutoff_frequency, filter_order, down_sample_rate, num_symbols):
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
    
    sample = np.fromfile(sample_file, dtype=np.complex64)

    filtered = low_pass_filter(sample, sample_rate, cutoff_frequency, filter_order)

    down_sampled = filtered[::down_sample_rate]
    new_sample_rate = sample_rate / down_sample_rate
    # down_sampled.tofile('samplesDownSampled.bin')

    num_symbol_samples = int(new_sample_rate / num_symbols)
    num_symbol_samples = int(num_symbol_samples / 2) # Dividing by 2 because signal was 0.5 seconds in duration

    print("Samples per symbol: ", num_symbol_samples, " | ", "Samples in sample: ", len(down_sampled))
    for i in range(num_symbols):
        start_index = i * num_symbol_samples
        end_index = start_index + num_symbol_samples
        symbol_sample = down_sampled[start_index:end_index]
        print(peak_freq(symbol_sample, sample_rate/down_sample_rate))
        

if __name__ == "__main__":
    sample_rate = 2e6
    cutoff_frequency = 15000
    filter_order = 5  # Adjust this as needed
    down_sample_rate = 5
    num_symbols = 8
    decode_fm_packet("samplesCropped.bin", sample_rate, cutoff_frequency, filter_order, down_sample_rate, num_symbols)
