import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

plt.style.use('dark_background')
    
def generate_fm_packet(binary_string, frequency, second_frequency, duration, sample_rate):
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

# Testing generation of FM packet
binary_string = '11010001'
duration = 1
sample_rate = 10000
frequency = 10
second_frequency = 20
transmission_signal = generate_fm_packet(binary_string, frequency, second_frequency, duration, sample_rate) 
# End Testing generation of FM packet

plt.plot(transmission_signal)

# Vertical lines for delineate symbols
t = np.arange(0, duration, 1 / sample_rate)
num_symbols = len(binary_string)
symbol_length = len(t) / num_symbols
for i in range(num_symbols+1):
    plt.axvline(x=symbol_length * i, color='r', linestyle='--')


plt.show()