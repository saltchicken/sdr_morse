import uhd
import numpy as np
import SoapySDR
import time
from SoapySDR import *

from scipy.signal import butter, lfilter, resample_poly

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
plt.style.use('dark_background')


# TODO: More intuitive way for calling buffer_size
class Segment:
    def __init__(self, data, sample_rate):
        self.sample_rate = sample_rate
        self.data = data
            
    def display(self, buffer_size=1024, fft_size=None, subplot=False):
        if fft_size == None:
            fft_size = buffer_size
        iterations = len(self.data) // buffer_size # This needs to be even
        waterfall_data = np.zeros((iterations, fft_size))
        for i, buffer in enumerate(self.data.reshape(iterations, fft_size)):
            freq_domain = np.fft.fftshift(np.fft.fft(buffer, n=fft_size))
            max_magnitude_index = np.abs(freq_domain)
            waterfall_data[i, :] = max_magnitude_index
        
        freq_range = self.sample_rate / 2000 # Half sample_rate and convert to kHz
        sample_time = buffer_size * iterations / self.sample_rate
        # plt.figure(figsize=(12, 10))
        plt.imshow(waterfall_data, extent=[-freq_range, freq_range, 0, sample_time], aspect='auto')
        # manager = plt.get_current_fig_manager()
        # manager.window.geometry("+100+100")
        # plt.imshow(waterfall_data, aspect='auto')  # extent=[0, sample_rate / 1e3, 0, num_samples] ---- Also used LogNorm?
        plt.xlabel('Frequency (kHz)')
        plt.ylabel('Time (s)')
        plt.title('Waterfall Plot')
        plt.colorbar(label='Amplitude')
        if not subplot:
            plt.show()
        
    def cos_wave_generator(self, sample_rate, frequency, samples):
        i = 0
        while True:
            t = (np.arange(samples) + i * samples) / sample_rate
            yield np.exp(1j * 2 * np.pi * frequency * t).astype(np.complex64)
            i += 1
    
    # TODO: There may be an issue with calling this multiple times
    def shift_center(self, frequency):
        wave_gen = self.cos_wave_generator(self.sample_rate, -frequency, len(self.data))
        self.data = self.data * next(wave_gen)
    
    def plot(self):
        plt.plot(self.data)
        plt.show()

class Packet(Segment): 
    def __init__(self, segment: Segment):
        super().__init__(segment.data, segment.sample_rate)
        
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
    
    def send(self, packet: Packet):
        self.streamer.send(packet.data, self.metadata)
    
    def generate_fm_packet(self, binary_string, frequency, second_frequency, duration, sample_rate):
        t = np.arange(0, duration, 1 / sample_rate)
        num_symbols = len(binary_string)
        symbol_length = len(t) / num_symbols
        assert int(symbol_length) == symbol_length, "Sample amount of t must be divisible by num_symbols"
        symbol_length = int(symbol_length)
        print("Num symbols: ", num_symbols, "|", "Symbol length: ", symbol_length)
        transmission_signal = np.zeros(len(t), dtype=np.complex64)
        time_interval = 1 / sample_rate
        
        # TODO: Make more efficient. Calc phase shift right after symbol wave. Use 'np.exp()'
        for i, bit in enumerate(binary_string):
            start_index = symbol_length * i
            end_index = start_index + symbol_length
            symbol_time = symbol_length * time_interval
            if i == 0:
                phase_shift = 0.0
            elif binary_string[i-1] == '0':
                phase_shift += 2 * np.pi * frequency * symbol_time
            elif binary_string[i-1] == '1':
                phase_shift += 2 * np.pi * second_frequency * symbol_time
            else:
                print("Something is wrong with calculating phase shift")
            if bit == '0':
                symbol_wave_real = np.cos(2 * np.pi * frequency * t + phase_shift)
                symbol_wave_imag = np.cos(2 * np.pi * frequency * t + (phase_shift - (np.pi / 2)))
            elif bit == '1':
                symbol_wave_real = np.cos(2 * np.pi * second_frequency * t + phase_shift)
                symbol_wave_imag = np.cos(2 * np.pi * second_frequency * t + (phase_shift - (np.pi / 2)))
            else:
                print("Something is wrong with the binary_string")
            transmission_signal.real[start_index:end_index] = symbol_wave_real[0:symbol_length]
            transmission_signal.imag[start_index:end_index] = symbol_wave_imag[0:symbol_length]
        transmission_segment = Segment(transmission_signal, sample_rate)
        return Packet(transmission_segment)
    
    def burst(self, packet: Packet, times, pause_delay=0):
        for i in range(times):
            self.send(packet)
            if pause_delay:
                time.sleep(pause_delay)
   
        
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
    
    def getSegment(self, num_samps=2048000, buffer_size=1024, center_frequency=None):
        self.set_buffer_size(buffer_size)
        samples = []
        iterations = num_samps // buffer_size
        for i in range(iterations):
            sample = np.copy(self.read())
            samples.append(sample)
        data = np.concatenate(samples)
        if center_frequency:
            segment = Segment(data, self.sample_rate)
            segment.shift_center(center_frequency)
            return segment
        else:
            return Segment(data, self.sample_rate)
    
    def waterfall(self, iterations=1000, buffer_size=1024, fft_size=None):
        self.set_buffer_size(buffer_size)
        if fft_size == None:
            fft_size = buffer_size
        waterfall_data = np.zeros((iterations, fft_size))
        
        fig, ax = plt.subplots()
        fig.set_size_inches(12, 10)
        im = ax.imshow(waterfall_data, cmap='viridis')
        
        freq_range = self.sample_rate / 2000 # Half sample_rate and convert to kHz
        sample_time = buffer_size * iterations / self.sample_rate
        plt.imshow(waterfall_data, extent=[-freq_range, freq_range, 0, sample_time], aspect='auto')
        ax.set_xlabel('Frequency (kHz)')
        ax.set_ylabel('Time (s)')
        ax.set_title('Waterfall Plot')
        fig.colorbar(im, label='Amplitude')
        
        def update_image(frame):
            sample = self.read()
            freq_domain = np.fft.fftshift(np.fft.fft(sample, n=fft_size))
            max_magnitude_index = np.abs(freq_domain)
            waterfall_data[1:, :] = waterfall_data[:-1, :]
            waterfall_data[0, :] = max_magnitude_index
            im.set_array(waterfall_data)
            im.set_extent([-freq_range, freq_range, 0, sample_time])
            return im,
        
        interval = 0  # milliseconds
        ani = FuncAnimation(fig, update_image, interval=interval, blit=True)
        plt.show()
    

        
class QuadDemodSegment(Segment):
    def __init__(self, segment):
        super().__init__(segment.data, segment.sample_rate)
        self.data = self.quad_demod(self.data)
    def quad_demod(self, segment):
        return 0.5 * np.angle(segment[:-1] * np.conj(segment[1:]))
    
class Filter(Segment):
    def __init__(self, segment: Segment):
        super().__init__(segment.data, segment.sample_rate)
        self.data = self.low_pass_filter(self.data, 10000)
        
    def low_pass_filter(self, data, cutoff_frequency, filter_order=5):
        nyquist_frequency = self.sample_rate / 2
        normalized_cutoff_frequency = cutoff_frequency / nyquist_frequency
        b, a = butter(filter_order, normalized_cutoff_frequency, btype='low')
        filtered = lfilter(b, a, data)
        filtered = filtered.astype(np.complex64)
        assert data.dtype == filtered.dtype, "Output of filtered signal mismatched with sample signal"
        return filtered
    
class Resample(Segment):
    def __init__(self, segment: Segment, interpolation=1, decimation=1):
        super().__init__(segment.data, segment.sample_rate)
        self.data = self.resample(self.data, interpolation, decimation)
    def resample(self, data, interpolation, decimation):
        return resample_poly(data, interpolation, decimation)#interpolation == upsample, decimation == downsample
        # return sample[::downsample_rate] Alternative
        
class DecodedSegment(Segment):
    def __init__(self, segment: Segment):
        super().__init__(segment.data, segment.sample_rate)
        
        plt.figure(figsize=(10, 8))
        plt.subplot(2, 2, 1)
        # plt.plot(self.data)
        self.display(subplot=True)
        
        plt.subplot(2, 2, 2)
        self.lowpass = Filter(self)
        # plt.plot(self.data)
        self.lowpass.display(subplot=True)
        
        plt.subplot(2, 2, 3)
        self.demod = QuadDemodSegment(self.lowpass)
        plt.plot(self.demod.data)
        
        plt.subplot(2, 2, 4)
        self.resample = Resample(self.demod, 1, 128000)
        plt.plot(self.resample.data)
        print(self.decode(self.resample))
        plt.show()
        
    def decode(self, segment:Segment):
        return (np.real(segment.data) < 0).astype(int) # Why is real needed
        
    
    
