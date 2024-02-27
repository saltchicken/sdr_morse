from sdr_tools.classes import UHD_RX_TX

from IPython import embed

# apply settings
sample_rate = 2e6
rx_freq = 434e6 # frequency
rx_antenna = ''
tx_antenna = ''

tx_freq = 434e6 # center_freq

with UHD_RX_TX(sample_rate, rx_freq, tx_freq, rx_antenna, tx_antenna) as transceiver:
    freq = 40000
    freq_deviation = 10000
    symbol_length = 10000

    fm_packet = transceiver.generate_fm_packet('10100010', freq, freq_deviation, symbol_length)
    embed()