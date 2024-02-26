from sdr_tools.classes import Lime_RX_TX

from IPython import embed

# apply settings
sample_rate = 2e6
rx_freq = 434e6 # frequency
antenna = 'LNAW'

tx_freq = 434e6 # center_freq

with Lime_RX_TX(sample_rate, rx_freq, tx_freq, antenna, 'BAND2') as transceiver:
    freq = 40000
    freq_deviation = 10000
    symbol_length = 10000

    fm_packet = transceiver.generate_fm_packet('10100010', freq, freq_deviation, symbol_length)
    embed()