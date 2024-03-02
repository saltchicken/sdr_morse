from sdr_tools.classes import UHD_RX_TX, TCP_Protocol

from IPython import embed

# apply settings
sample_rate = 2e6
rx_freq = 434e6 # frequency
rx_antenna = ''
tx_antenna = ''

rx_channel = 40000
tx_channel = 25000

tx_freq = 434e6 # center_freq

with UHD_RX_TX(sample_rate, rx_freq, tx_freq, rx_antenna, tx_antenna, rx_channel, tx_channel, full_duplex=True) as transceiver:
    protocol = TCP_Protocol(channel_freq=25000)
    fm_packet = protocol.syn
    embed(quiet=True)