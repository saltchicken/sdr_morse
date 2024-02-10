from sdr_tools import utils
import matplotlib.pyplot as plt
import numpy as np



sample_rate = 10000
center_freq = 434e6

# freq = 50000
# freq_deviation = 10000
# duration = 1
# symbol_size = 250000

# transmission_signal = utils.generate_fm_packet(streamer, '1010001', freq, freq_deviation, symbol_size / sample_rate)

freq = 20
freq_deviation = 10
duration = 1
binary_string = '10100010'

transmission_signal = utils.generate_fm_packet_complex(binary_string, freq - freq_deviation, freq + freq_deviation, duration, sample_rate)

# plt.plot(transmission_signal)
plt.plot(transmission_signal.real, label='Real')
plt.plot(transmission_signal.imag, label='Imaginary')
# Vertical lines for delineate symbols
t = np.arange(0, duration, 1 / sample_rate)
num_symbols = len(binary_string)
symbol_length = len(t) / num_symbols
for i in range(num_symbols+1):
    plt.axvline(x=symbol_length * i, color='r', linestyle='--')
plt.show()

