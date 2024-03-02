from sdr_tools.classes import UHD_RX_TX, FM_Packet

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
    fm_packet = FM_Packet('101000111001001000010100101101010110', channel_freq=25000)
    embed()