import uhd
import numpy as np
import SoapySDR
import time
from SoapySDR import *

class UHD_TX_Streamer:
    def __init__(self, sample_rate, center_freq):
        self.sample_rate = sample_rate
        self.center_freq = center_freq
        self.usrp = uhd.usrp.MultiUSRP()
        self.stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
        self.usrp.set_tx_rate(self.sample_rate)
        self.usrp.set_tx_freq(self.center_freq)
        self.streamer = self.usrp.get_tx_stream(self.stream_args)
        self.metadata = uhd.types.TXMetadata()
        # INIT_DELAY = 0.05
        # self.metadata.time_spec = uhd.types.TimeSpec(self.usrp.get_time_now().get_real_secs() + INIT_DELAY)
        # self.metadata.has_time_spec = bool(self.streamer.get_num_channels())
    
    def send(self, message):
        self.streamer.send(message, self.metadata)
        
class Receiver:
    def __init__(self, sample_rate, frequency, antenna, freq_correction=0, read_buffer_size=1024):
        self.sample_rate = sample_rate
        self.frequency = frequency
        self.antenna = antenna
        self.freq_correction = freq_correction
        
        self.read_buffer = np.array([0] * read_buffer_size, np.complex64)
        
    def __enter__(self):
        print('Entering Receiver')
        args = dict(driver="lime")
        self.sdr = SoapySDR.Device(args)
        self.sdr.setSampleRate(SOAPY_SDR_RX, 0, self.sample_rate)
        self.sdr.setFrequency(SOAPY_SDR_RX, 0, self.frequency)
        self.sdr.setAntenna(SoapySDR.SOAPY_SDR_RX, 0, self.antenna)
        self.sdr.setFrequencyCorrection(SoapySDR.SOAPY_SDR_RX, 0, self.freq_correction)
        self.rxStream = self.sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
        self.sdr.activateStream(self.rxStream)  # start streaming
        return self
        
    def __exit__(self, *args, **kwargs):
        print("Exiting receiver")
        self.sdr.deactivateStream(self.rxStream)  # stop streaming
        self.sdr.closeStream(self.rxStream)
        del self.sdr
    
    def set_buffer_size(self, buffer_size):
        self.read_buffer = np.zeros(buffer_size, np.complex64)
    
    def read(self):
        sr = self.sdr.readStream(self.rxStream, [self.read_buffer], len(self.read_buffer))
        return self.read_buffer
    
    def timed_read(self, duration):
        start_time = time.time()
        received_sample = []
        while time.time() - start_time < duration:
            received_sample.append(np.copy(self.read()))
        return np.concatenate(received_sample)

    def read_chunk(self, num_samps):
        num_reads = num_samps // len(self.read_buffer)
        received_sample = []
        for i in range(num_reads):
            received_sample.append(np.copy(self.read()))
        return np.concatenate(received_sample)
    
class Sample:
    def __init__(self, receiver, sample_rate, data=None, num_samps=1024000, buffer_size=1024):
        receiver.set_buffer_size(buffer_size)
        self.sample_rate = sample_rate
        self.data = data
        self.samples = []
        if self.data == None:
            iterations = num_samps // buffer_size
            for i in range(iterations):
                sample = np.copy(receiver.read())
                self.samples.append(sample)
        self.data = np.concatenate(self.samples)