from sdr_tools.classes import UHD_RX_TX, FM_Packet

from IPython import embed

# apply settings
sample_rate = 2e6
rx_freq = 434e6 # frequency
rx_antenna = ''
tx_antenna = ''

tx_freq = 434e6 # center_freq

with UHD_RX_TX(sample_rate, rx_freq, tx_freq, rx_antenna, tx_antenna, 40000, full_duplex=True) as transceiver:
    fm_packet = FM_Packet('10101010', channel_freq=25000)
    embed()