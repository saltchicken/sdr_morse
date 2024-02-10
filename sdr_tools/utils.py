import numpy as np

def generate_carrier(streamer, frequency, duration):
    t = np.arange(0, duration, 1 / streamer.sample_rate)
    carrier_signal = np.cos(2 * np.pi * frequency * t)
    return carrier_signal

def generate_fm_packet_legacy(streamer, binary_string, freq, freq_deviation, duration):
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

def generate_fm_packet_update(binary_string, frequency, second_frequency, duration, sample_rate):
    t = np.arange(0, duration, 1 / sample_rate)
    num_symbols = len(binary_string)
    symbol_length = len(t) / num_symbols
    assert int(symbol_length) == symbol_length, "Sample amount of t must be divisible by num_symbols"
    symbol_length = int(symbol_length)
    print("Num symbols: ", num_symbols, "|", "Symbol length: ", symbol_length)
    transmission_signal = np.zeros(len(t))
    time_interval = 1 / sample_rate
    
    for i, bit in enumerate(binary_string):
        start_index = symbol_length * i
        end_index = symbol_length * i + symbol_length
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
            symbol_wave = np.cos(2 * np.pi * frequency * t + phase_shift)
        elif bit == '1':
            symbol_wave = np.cos(2 * np.pi * second_frequency * t + phase_shift)
        else:
            print("Something is wrong with the binary_string")
        transmission_signal[start_index:end_index] = symbol_wave[0:symbol_length]
    return transmission_signal

def generate_fm_packet_complex(binary_string, frequency, second_frequency, duration, sample_rate):
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