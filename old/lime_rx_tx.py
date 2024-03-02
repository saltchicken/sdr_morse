from sdr_tools.classes import Lime_RX_TX, TCP_Protocol, logger

import sys

from IPython import embed

logger.remove()
logger.add(sys.stderr, level="DEBUG")

# apply settings
sample_rate = 2e6
rx_freq = 434e6 # frequency
antenna = 'LNAW'

rx_channel = 25000
tx_channel = 40000

tx_freq = 434e6 # center_freq

with Lime_RX_TX(sample_rate, rx_freq, tx_freq, antenna, 'BAND2', rx_channel, tx_channel, full_duplex=True) as transceiver:
    protocol = TCP_Protocol(channel_freq=40000)
    fm_packet = protocol.syn
    embed(quiet=True)