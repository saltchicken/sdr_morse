from sdr_tools.classes import UHD_RX

from IPython import embed

sample_rate = 2e6
frequency = 434e6
antenna = 'LNAW'

receiver = UHD_RX(sample_rate, frequency, antenna) 
embed()