from sdr_tools.classes import Receiver

from IPython import embed

# apply settings
sample_rate = 2e6
frequency = 433.5e6
antenna = 'LNAW'

with Receiver(sample_rate, frequency, antenna) as receiver:
    embed()