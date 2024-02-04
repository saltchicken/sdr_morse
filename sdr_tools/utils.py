import numpy as np

def generate_carrier(streamer, frequency, duration):
    t = np.arange(0, duration, 1 / streamer.sample_rate)
    carrier_signal = np.cos(2 * np.pi * frequency * t)
    return carrier_signal

def generate_fm_packet(streamer, binary_string, freq, freq_deviation, duration):
    carrier_signal = generate_carrier(streamer, freq - freq_deviation // 2, duration)
    deviation_signal = generate_carrier(streamer, freq + freq_deviation // 2, duration)
    transmission_signal = np.tile(carrier_signal, 8)
    for char in binary_string:
        if char == '0':
            transmission_signal = np.append(transmission_signal, carrier_signal)
        elif char == '1':
            transmission_signal = np.append(transmission_signal, deviation_signal)
        else:
            print('error')
    return transmission_signal

def cos_wave_generator(sample_rate, frequency, samples):
    i = 0
    while True:
        t = (np.arange(samples) + i * samples) / sample_rate
        yield np.exp(1j * 2 * np.pi * frequency * t).astype(np.complex64)
        i += 1