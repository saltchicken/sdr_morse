from sdr_tools.classes import Receiver, Segment, QuadDemodSegment, DecodedSegment
from sdr_tools import utils
import numpy as np

from scipy.signal import resample_poly, firwin, bilinear, lfilter
import matplotlib.pyplot as plt

from IPython import embed

# apply settings
sample_rate = 2e6
frequency = 433.5e6
antenna = 'LNAW'

with Receiver(sample_rate, frequency, antenna) as receiver:
    embed()