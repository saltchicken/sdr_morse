from sdr_tools.classes import Lime_RX_TX, FM_Packet

from IPython import embed

# apply settings
sample_rate = 2e6
rx_freq = 434e6 # frequency
antenna = 'LNAW'

tx_freq = 434e6 # center_freq

with Lime_RX_TX(sample_rate, rx_freq, tx_freq, antenna, 'BAND2') as transceiver:
    fm_packet = FM_Packet('10100010')
    embed()