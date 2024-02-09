# from sdr_tools import classes
from sdr_tools import utils
import matplotlib.pyplot as plt
import numpy as np

class DummyStreamer:
    def __init__(self, sample_rate, center_freq, mode):
        self.sample_rate = sample_rate
        self.center_freq = center_freq
        self.mode = mode
        # self.usrp = uhd.usrp.MultiUSRP()
        # self.stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
        # if self.mode == 'tx':
        #     self.usrp.set_tx_rate(self.sample_rate)
        #     self.usrp.set_tx_freq(self.center_freq)
        #     self.streamer = self.usrp.get_tx_stream(self.stream_args)
        #     self.metadata = uhd.types.TXMetadata()
        #     # INIT_DELAY = 0.05
        #     # self.metadata.time_spec = uhd.types.TimeSpec(self.usrp.get_time_now().get_real_secs() + INIT_DELAY)
        #     # self.metadata.has_time_spec = bool(self.streamer.get_num_channels())
        # elif mode == 'rx':
        #     self.usrp.set_rx_rate(self.sample_rate)
        #     self.usrp.set_rx_freq(self.center_freq)
        #     self.streamer = self.usrp.get_rx_stream(self.stream_args)
        #     self.metadata = uhd.types.RXMetadata()
    
    def send(self, message):
        self.streamer.send(message, self.metadata)

freq = 2000
freq_deviation = 100
duration = 1
symbol_size = 100


sample_rate = 100
center_freq = 434e6
mode = 'tx'
streamer = DummyStreamer(sample_rate, center_freq, mode)
# transmission_signal = utils.generate_carrier(streamer, freq, duration)
# wave_gen = utils.cos_wave_generator_not_complex(sample_rate, freq_deviation, 300)
transmission_signal = utils.generate_fm_packet(streamer, '1010001', freq, freq_deviation, symbol_size / sample_rate)
# transmission_signal[0:300] = transmission_signal[0:300] * next(wave_gen)

# fft_transmission = np.fft.fftshift(np.fft.fft(transmission_signal[0:300]))
# # fft_freq = np.fft.fftfreq(len(t), t[1] - t[0])
# plt.figure(figsize=(12, 6))
# plt.subplot(2, 1, 1)
# plt.plot(transmission_signal[0:300],)
plt.plot(transmission_signal,)

# plt.subplot(2, 1, 2)
# plt.plot(np.abs(fft_transmission))
plt.show()