import uhd
import numpy as np
import SoapySDR
import time
from SoapySDR import *

from scipy.signal import butter, lfilter, resample_poly
from commpy.filters import rrcosfilter

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from abc import ABC, abstractmethod
from dataclasses import dataclass
import threading, queue
from loguru import logger
import sys
plt.style.use('dark_background')

class ShiftFrequency():
    def __init__(self, sample_rate, frequency, num_samps):
        self.i = 0
        self.frequency = -frequency
        self.sample_rate = sample_rate
        self.num_samps = num_samps
    def next(self):
        t = (np.arange(self.num_samps) + self.i * self.num_samps) / self.sample_rate
        self.i += 1
        return np.exp(1j * 2 * np.pi * self.frequency * t).astype(np.complex64)
    def reset(self):
        self.i = 0
    def set_frequency(self, frequency):
        self.frequency = frequency
        self.reset()

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
    
    # TODO: There may be an issue with calling this multiple times
    def shift_center(self, frequency):
        # wave_gen = cos_wave_generator(self.sample_rate, -frequency, len(self.data))
        wave_gen = ShiftFrequency(self.sample_rate, frequency, len(self.data))
        self.data = self.data * wave_gen.next()
    
    def plot(self):
        plt.plot(self.data)
        plt.show()

class Packet(Segment): 
    def __init__(self, segment: Segment):
        super().__init__(segment.data, segment.sample_rate)


@dataclass
class FM_Settings:
    sample_rate: int = int(2e6)
    freq_deviation: int = 10000
    symbol_length: int = 10000   
    
class FM_Packet(Packet):
    def __init__(self, binary_string, channel_freq):
        settings = FM_Settings()
        segment = self.generate_fm_packet(binary_string, channel_freq, settings)
        super().__init__(segment)
    
    def generate_fm_packet(self, binary_string, channel_freq, settings: FM_Settings):
        frequency = channel_freq - settings.freq_deviation
        second_frequency = channel_freq + settings.freq_deviation
        num_symbols = len(binary_string)
        duration = (num_symbols * settings.symbol_length) / settings.sample_rate
        t = np.arange(0, duration, 1 / settings.sample_rate)
        logger.debug(f"Num symbols: {num_symbols}| Symbol length: {settings.symbol_length}")
        transmission_signal = np.zeros(len(t), dtype=np.complex64)
        time_interval = 1 / settings.sample_rate
        
        # TODO: Make more efficient. Calc phase shift right after symbol wave. Use 'np.exp()'
        for i, bit in enumerate(binary_string):
            start_index = settings.symbol_length * i
            end_index = start_index + settings.symbol_length
            symbol_time = settings.symbol_length * time_interval
            if i == 0:
                phase_shift = 0.0
            elif binary_string[i-1] == '0':
                phase_shift += 2 * np.pi * frequency * symbol_time
            elif binary_string[i-1] == '1':
                phase_shift += 2 * np.pi * second_frequency * symbol_time
            else:
                logger.debug("Something is wrong with calculating phase shift")
            if bit == '0':
                symbol_wave_real = np.cos(2 * np.pi * frequency * t + phase_shift)
                symbol_wave_imag = np.cos(2 * np.pi * frequency * t + (phase_shift - (np.pi / 2)))
            elif bit == '1':
                symbol_wave_real = np.cos(2 * np.pi * second_frequency * t + phase_shift)
                symbol_wave_imag = np.cos(2 * np.pi * second_frequency * t + (phase_shift - (np.pi / 2)))
            else:
                logger.debug("Something is wrong with the binary_string")
            transmission_signal.real[start_index:end_index] = symbol_wave_real[0:settings.symbol_length]
            transmission_signal.imag[start_index:end_index] = symbol_wave_imag[0:settings.symbol_length]
        transmission_segment = Segment(transmission_signal, settings.sample_rate)
        return transmission_segment

class TCP_Protocol():
    preamble = np.array([1,0,1,0,0,0,1,1]).astype(int)
    syn_id = np.array([1,1,0,0]).astype(int)
    syn_ack_id = np.array([1,1,0,1]).astype(int)
    ack_id = np.array([1,1,1,0]).astype(int)
    def __init__(self, channel_freq):
        self.channel_freq = channel_freq
        self.syn = TCP_Packet(self.channel_freq, TCP_Protocol.preamble, TCP_Protocol.syn_id)
        self.syn_ack = TCP_Packet(self.channel_freq, TCP_Protocol.preamble, TCP_Protocol.syn_ack_id)
        self.ack = TCP_Packet(self.channel_freq, TCP_Protocol.preamble, TCP_Protocol.ack_id)
        
        self.syn_flag = False
        
class TCP_Packet(FM_Packet):
    def __init__(self, channel_freq, preamble, id):
        self.channel_freq = channel_freq
        self.preamble = preamble
        self.id = id
        self.binary_string = self.generate_binary_string()
        super().__init__(self.binary_string, self.channel_freq)
    
    def generate_binary_string(self):
        result = np.append(self.preamble, self.id)
        count_ones = np.count_nonzero(result == 1)
        # TODO: Note somewhere this is even parity
        if count_ones % 2 == 0:
            even_parity_bit = np.array([0])
        else:
            even_parity_bit = np.array([1])
        result_with_parity = np.append(result, even_parity_bit)
        binary_string = ''.join([str(x) for x in result_with_parity])
        logger.debug(f"Created FM Packet with binary string: {binary_string}")
        return binary_string   

class Filter(Segment):
    def __init__(self, segment: Segment):
        super().__init__(segment.data, segment.sample_rate)
        self.data = self.low_pass_filter(self.data, self.sample_rate, 10000)
    
    @staticmethod    
    def low_pass_filter(data, sample_rate, cutoff_frequency, filter_order=5):
        nyquist_frequency = sample_rate / 2
        normalized_cutoff_frequency = cutoff_frequency / nyquist_frequency
        b, a = butter(filter_order, normalized_cutoff_frequency, btype='low')
        filtered = lfilter(b, a, data)
        filtered = filtered.astype(np.complex64)
        assert data.dtype == filtered.dtype, "Output of filtered signal mismatched with sample signal"
        return filtered
           
class QuadDemod(Segment):
    def __init__(self, segment):
        super().__init__(segment.data, segment.sample_rate)
        self.data = self.quad_demod(self.data)
    def quad_demod(self, segment):
        return 0.5 * np.angle(segment[:-1] * np.conj(segment[1:]))
    
class Resample(Segment):
    def __init__(self, segment: Segment, interpolation=1, decimation=1):
        super().__init__(segment.data, segment.sample_rate)
        self.data = self.resample(self.data, interpolation, decimation)
    def resample(self, data, interpolation, decimation):
        return resample_poly(data, interpolation, decimation)#interpolation == upsample, decimation == downsample
        # return sample[::downsample_rate] Alternative
        
class Decoded(Segment):
    def __init__(self, segment: Segment, symbol_length=10000):
        super().__init__(segment.data, segment.sample_rate)
        self.decode(symbol_length)
        
    def decode_segment(self, segment:Segment):
        return (np.real(segment.data) < 0).astype(int) # Why is real needed    
        
    def decode(self, symbol_length):
        self.lowpass = Filter(self)
        self.demod = QuadDemod(self.lowpass)
        self.demod.data = self.demod.data[symbol_length//2:] # Offset the sample. Poverty synchronization
        self.resample = Resample(self.demod, 1, symbol_length)
        self.decoded = self.decode_segment(self.resample)
        # logger.debug(self.decoded)
        
    def plot_decoded(self):
        plt.figure(figsize=(10, 8))
        
        plt.subplot(2, 2, 1)
        self.display(subplot=True)
        
        plt.subplot(2, 2, 2)
        self.lowpass.display(subplot=True)
        
        plt.subplot(2, 2, 3)
        plt.plot(self.demod.data)
        
        plt.subplot(2, 2, 4)
        plt.plot(self.resample.data)
        plt.show()    
                
class Transmitter(ABC):
    def __init__(self, sample_rate, center_freq, tx_antenna, tx_gain=20):
        self.sample_rate = sample_rate
        self.center_freq = center_freq
        self.tx_antenna = tx_antenna
        self.tx_gain = tx_gain
        
    @abstractmethod
    def send(self, packet: Packet):
        pass
        
    def generateBPSK(self, bits):
        # bits = np.array([1,0,0,0,1,1,0,0,0,1,1,1,0,0,1,1], np.complex64)
        # bits = np.tile(bits, 100)
        num_symbols = len(bits)
        logger.debug("Num symbols: ", num_symbols)
        sps = 8
        sample_rate = self.sample_rate
        x = np.array([])
        for bit in bits:
            pulse = np.zeros(sps, np.complex64)
            pulse[0] = int(bit)*2-1 # set the first value to either a 1 or -1
            x = np.concatenate((x, pulse)) # add the 8 samples to the signal
        beta = 0.35
        Ts = sps * (1 / sample_rate)
        rrc_time, rrc = rrcosfilter(50, beta, Ts, sample_rate)
        transmission = np.convolve(x, rrc)
        transmission_segment = Segment(transmission, sample_rate)
        return transmission_segment
    
    def burst(self, packet: Packet, times, pause_delay=0):
        for i in range(times):
            self.send(packet)
            if pause_delay:
                time.sleep(pause_delay) 
               
class Receiver(ABC):
    def __init__(self, sample_rate, frequency, antenna, freq_correction=0, read_buffer_size=1024):
        self.sample_rate = sample_rate
        self.frequency = frequency
        self.antenna = antenna
        self.freq_correction = freq_correction
        
        self.read_buffer = np.array([0] * read_buffer_size, np.complex64)
    
    @abstractmethod    
    def read(self):
        pass
    
    def set_buffer_size(self, buffer_size):
        logger.debug(f"Setting read_buffer_size to {buffer_size}")
        self.read_buffer = np.zeros(buffer_size, np.complex64)
    
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
    
    # TODO: This does not work for UHD_RX. Look for efficient TODO about iterating the read inside UHD_RX 
    # TODO: Remove hardcoded frequency_shift of 40000
    def live_samples(self, buffer_size=102400, fft_size=None, frequency_shift=40000, decimator=40):
        if fft_size == None:
            fft_size = buffer_size
        assert buffer_size % decimator == 0, "buffer_size must be equally divisable by decimator"
        assert self.sample_rate % decimator == 0, "sample_rate must be equally divisable by decimator"
        line_data = np.zeros(buffer_size//decimator)
        fig, ax = plt.subplots()
        fig.set_size_inches(12, 10)
        
        x_data = np.arange(1, buffer_size+1)[::decimator]
        line, = ax.plot(x_data, line_data)
        
        # Clear the read_buffer of Soapy Device
        self.set_buffer_size(int(4e6))
        self.read()
        # Set to the corrent buffer_size for reading
        self.set_buffer_size(buffer_size)
        
        # shift_frequency = ShiftFrequency(self.sample_rate, frequency_shift, buffer_size//decimator)
        shift_frequency = ShiftFrequency(self.sample_rate, frequency_shift, len(self.read_buffer))
        
        
        def update(frame):
            sample = self.read()
            sample = sample * shift_frequency.next()
            sample = Filter.low_pass_filter(sample, self.sample_rate, 15000)
            sample = sample[::decimator]
            
            # sample = Filter.low_pass_filter(sample, self.sample_rate//decimator, 10000)
            line.set_ydata(sample)
            return line,
        
        interval = 0
        ani = FuncAnimation(fig, update, interval=interval)
        
        plt.show()
        
    def capture_signal(self, kill_rx, channel_freq, threshold=0.005):
        shift_frequency = ShiftFrequency(self.sample_rate, channel_freq, len(self.read_buffer))
        signal = []
        while not kill_rx.is_set():
            sample = self.read()
            sample = sample * shift_frequency.next()
            if np.max(np.abs(sample)) >= threshold:
                if len(signal) == 0:
                    logger.info(f"Signal found. Writing...")
                signal.append(sample)
            else:
                if len(signal) > 0:
                    break
        if kill_rx.is_set() and len(signal) == 0:
            logger.debug('Exiting capture signal')
            return None
        else:
            captured_signal = Segment(np.concatenate(signal), self.sample_rate)
            logger.info(f"Returning captured signal. Signal contains {len(captured_signal.data)} samples")
            return captured_signal
        
    def capture_signal_decode(self, kill_rx, channel_freq, symbol_length=10000):
        # try:
            # received = func_timeout(5, self.capture_signal)
            # received = received[0]
        # except FunctionTimedOut:
        #     logger.debug("Capture Signal timedout")
        #     return None
        received = self.capture_signal(kill_rx=kill_rx, channel_freq=channel_freq)
        if received:
            decoded = Decoded(received, symbol_length)
            return decoded
        else:
            return None

class Lime_RX(Receiver):
    def __init__(self, sample_rate, frequency, antenna, freq_correction=0, read_buffer_size=1024):
        super().__init__(sample_rate, frequency, antenna, freq_correction, read_buffer_size)
    
    def __enter__(self):
        logger.debug('Entering Receiver')
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
        logger.debug("Exiting receiver")
        self.sdr.deactivateStream(self.rxStream)  # stop streaming
        self.sdr.closeStream(self.rxStream)
        if 'retain_sdr' not in kwargs: # Added so that self.sdr is not deleted before objects with multiple inheritence call __exit__. Bit hacky
            del self.sdr
     
    def waterfall(self, iterations=1000, buffer_size=1024, fft_size=256, decimator=4):
        buffer_fixer = 100
        if fft_size == None:
            fft_size = buffer_size
        waterfall_data = np.zeros((iterations, fft_size))
        
        fig, ax = plt.subplots()
        fig.set_size_inches(12, 10)
        im = ax.imshow(waterfall_data, cmap='viridis')
        
        freq_range = self.sample_rate / 2000 # Half sample_rate and convert to kHz
        time_domain = buffer_size * iterations * decimator / self.sample_rate
        plt.imshow(waterfall_data, extent=[-freq_range, freq_range, 0, time_domain], aspect='auto')
        ax.set_xlabel('Frequency (kHz)')
        ax.set_ylabel('Time (s)')
        ax.set_title('Waterfall Plot')
        fig.colorbar(im, label='Amplitude')
        
        # Clear the read_buffer of Soapy Device
        self.set_buffer_size(int(4e6))
        self.read()
        # Set to the corrent buffer_size for reading
        self.set_buffer_size(buffer_size * buffer_fixer)
        
        def update_image(frame):
            sample = self.read()
            sample = sample.reshape(buffer_fixer, buffer_size)
            sample = sample[::decimator]
            for i in range(buffer_fixer//decimator):
                freq_domain = np.fft.fftshift(np.fft.fft(sample[i], n=fft_size))
                max_magnitude_index = np.abs(freq_domain)
                waterfall_data[1:, :] = waterfall_data[:-1, :]
                waterfall_data[0, :] = max_magnitude_index
            im.set_array(waterfall_data)
            im.set_extent([-freq_range, freq_range, 0, time_domain])
            return im,
        
        interval = 0  # milliseconds
        ani = FuncAnimation(fig, update_image, interval=interval, blit=True)
        plt.show()
           
    # TODO: Add buffer_size as parameter to remove need for set_buffer_size
    def read(self):
        sr = self.sdr.readStream(self.rxStream, [self.read_buffer], len(self.read_buffer))
        return self.read_buffer

    def clear_read_buffer(self, num_samps=int(4e6)):
        logger.debug("Clearing read_buffer for Lime")
        previous_buffer = len(self.read_buffer)
        self.set_buffer_size(int(num_samps))
        self.read()
        self.set_buffer_size(previous_buffer)
               
class UHD_RX(Receiver):
    def __init__(self, sample_rate, frequency, antenna, freq_correction=0, read_buffer_size=2000):
        super().__init__(sample_rate, frequency, antenna, freq_correction, read_buffer_size)
        
    def __enter__(self):
        self.usrp = uhd.usrp.MultiUSRP()
        gain = 50 # dB

        self.usrp.set_rx_rate(self.sample_rate, 0)
        self.usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(self.frequency), 0)
        self.usrp.set_rx_gain(gain, 0)

        # Set up the stream and receive buffer
        st_args = uhd.usrp.StreamArgs("fc32", "sc16")
        st_args.channels = [0]
        self.rx_metadata = uhd.types.RXMetadata()
        self.rx_streamer = self.usrp.get_rx_stream(st_args)
        # recv_buffer = np.zeros((1, self.read_buffer_size), dtype=np.complex64)

        # Start Stream
        stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
        stream_cmd.stream_now = True
        self.rx_streamer.issue_stream_cmd(stream_cmd)
        
        return self
    
    def __exit__(self, *args, **kwargs):
        logger.debug("Exiting Receiver")
        stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
        self.rx_streamer.issue_stream_cmd(stream_cmd)
    
    #TODO: This function needs a LOT of refactoring. Too much experimentation to figure out solution.
    def waterfall(self, iterations=1000, buffer_size=2000, fft_size=256, decimator=4):
        self.uhd_recv_buffer = np.zeros((1, 2000), dtype=np.complex64) # Set for UHD specific. Refactor this out.
        buffer_fixer = 100
        waterfall_data = np.zeros((iterations, fft_size))
        
        fig, ax = plt.subplots()
        fig.set_size_inches(12, 10)
        im = ax.imshow(waterfall_data, cmap='viridis')
        
        freq_range = self.sample_rate / 2000 # Half sample_rate and convert to kHz
        time_domain = buffer_size * iterations * decimator / self.sample_rate
        plt.imshow(waterfall_data, extent=[-freq_range, freq_range, 0, time_domain], aspect='auto')
        ax.set_xlabel('Frequency (kHz)')
        ax.set_ylabel('Time (s)')
        ax.set_title('Waterfall Plot')
        fig.colorbar(im, label='Amplitude')
        
        # Clear the read_buffer of Soapy Device
        # self.set_buffer_size(int(4e6))
        # self.read()
        # Set to the corrent buffer_size for reading
        # self.set_buffer_size(buffer_size * buffer_fixer)
        
        num_samps = 200000
        # TODO: Pick a better name or possibly not need this
        def update_image(frame):
            # stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
            # stream_cmd.stream_now = True
            # self.rx_streamer.issue_stream_cmd(stream_cmd)
            sample_buffer = np.zeros(num_samps, dtype=np.complex64)
            for i in range(num_samps//2000):
                self.rx_streamer.recv(self.uhd_recv_buffer, self.rx_metadata)
                sample_buffer[i*2000:(i+1)*2000] = self.uhd_recv_buffer[0]
            # stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
            # self.rx_streamer.issue_stream_cmd(stream_cmd)
            # sample = self.read()
            sample = np.copy(sample_buffer)
            sample = sample.reshape(num_samps//2000, buffer_size)
            sample = sample[::decimator]
            for i in range(num_samps//2000//decimator):
                freq_domain = np.fft.fftshift(np.fft.fft(sample[i], n=fft_size))
                max_magnitude_index = np.abs(freq_domain)
                waterfall_data[1:, :] = waterfall_data[:-1, :]
                waterfall_data[0, :] = max_magnitude_index
            im.set_array(waterfall_data)
            im.set_extent([-freq_range, freq_range, 0, time_domain])
            return im,
        
        interval = 0  # milliseconds
        ani = FuncAnimation(fig, update_image, interval=interval, blit=True)
        plt.show()
            
    def read(self):
        # stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
        # stream_cmd.stream_now = True
        # self.rx_streamer.issue_stream_cmd(stream_cmd)
        
        # TODO: Possibly implement this for efficiency if larger buffer needed.
        # for i in range(num_samps//1000):
        #     self.rx_streamer.recv(recv_buffer, metadata)
        #     samples[i*1000:(i+1)*1000] = recv_buffer[0]
        
        self.rx_streamer.recv(self.read_buffer, self.rx_metadata)
        # self.read_buffer = np.copy(self.usrp.recv_num_samps(2000, self.frequency, self.sample_rate, [0], 0))
        # stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
        # self.rx_streamer.issue_stream_cmd(stream_cmd)
        return self.read_buffer

class Lime_TX(Transmitter):
    # TODO: Modify default gain behavior to call set_gain and print debug
    def __init__(self, sample_rate, center_freq, antenna="BAND2", gain=15):
        super().__init__(sample_rate, center_freq, antenna, gain)
        
    def __enter__(self):
        args = dict(driver="lime")
        self.sdr = SoapySDR.Device(args)
        self.sdr.setSampleRate(SOAPY_SDR_TX, 0, self.sample_rate)
        self.sdr.setFrequency(SOAPY_SDR_TX, 0, self.center_freq)
        self.sdr.setAntenna(SOAPY_SDR_TX, 0, self.tx_antenna)
        self.sdr.setGain(SOAPY_SDR_TX, 0, self.tx_gain)
        self.txStream = self.sdr.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32)
        self.sdr.activateStream(self.txStream)
        return self
        
    def __exit__(self, *args, **kwargs):
        logger.debug('Exiting Transmitter')
        self.sdr.deactivateStream(self.txStream)
        self.sdr.closeStream(self.txStream)
        del self.sdr
        
    def send(self, packet: Packet):
        # self.sdr.writeStream(SOAPY_SDR_TX, [packet.data], len(packet.data), timeoutUs=int(1e6))
        self.sdr.writeStream(self.txStream, [packet.data], packet.data.size, timeoutUs=1000000)
        
    def set_gain(self, gain):
        self.gain = gain
        self.sdr.setGain(SOAPY_SDR_TX, 0, self.gain)     
                 
class UHD_TX(Transmitter):
    def __init__(self, sample_rate, center_freq, antenna, gain=0):
        super().__init__(sample_rate, center_freq, antenna, gain)
        
    def __enter__(self):
        self.usrp = uhd.usrp.MultiUSRP()
        self.stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
        self.usrp.set_tx_rate(self.sample_rate)
        self.usrp.set_tx_freq(self.center_freq)
        self.usrp.set_tx_gain(self.tx_gain)
        # TODO: Add antenna selection with self.tx_antenna
        self.tx_streamer = self.usrp.get_tx_stream(self.stream_args)
        self.tx_metadata = uhd.types.TXMetadata()
        # INIT_DELAY = 0.05
        # self.tx_metadata.time_spec = uhd.types.TimeSpec(self.usrp.get_time_now().get_real_secs() + INIT_DELAY)
        # self.tx_metadata.has_time_spec = bool(self.tx_streamer.get_num_channels())
        return self
    
    def __exit__(self, *args, **kwargs):
        logger.debug("Exiting Transmitter")
    
    def send(self, packet: Packet):
        self.tx_streamer.send(packet.data, self.tx_metadata)
        
    def set_gain(self, gain):
        self.gain = gain
        self.usrp.set_tx_gain(self.gain)

class TX_Node(threading.Thread):
    def __init__(self, transmitter, TX_to_RX, RX_to_TX):
        super().__init__()
        self.transmitter = transmitter
        self.TX_to_RX = TX_to_RX
        self.RX_to_TX = RX_to_TX
        
        self.dispatcher = TransmitterDispatcher(self.transmitter, self.TX_to_RX, self.RX_to_TX)
        
    def run(self):
        self.kill_tx = threading.Event()
        logger.debug('Starting tx_node')
        while not self.kill_tx.is_set():
            RX_to_TX_data = self.RX_to_TX.get()
            ret = self.dispatcher.action(RX_to_TX_data)
        logger.debug('Killing tx_node')
        
    def stop(self):
        # TODO: Make sure that tx_node is running
        self.kill_tx.set()
        # TODO: Probably a better way to ensure that self.RX_to_TX.get() doesn't block stop
        self.RX_to_TX.put(None)
        self.join()
        logger.debug('TX thread successfully exited')
        
class RX_Node(threading.Thread):
    def __init__(self, receiver, channel_freq, TX_to_RX, RX_to_TX):
        super().__init__()
        self.receiver = receiver
        self.TX_to_RX = TX_to_RX
        self.RX_to_TX = RX_to_TX
        self.channel_freq = channel_freq
        
        self.dispatcher = ReceiverDispatcher(self.TX_to_RX, self.RX_to_TX)
    
    def run(self):
        self.kill_rx = threading.Event()
        logger.debug('Starting rx node')
        while not self.kill_rx.is_set():
            logger.debug('RX_Node listening')
            decoded = self.receiver.capture_signal_decode(self.kill_rx, self.channel_freq)
            if decoded:
                self.dispatcher.action(decoded.decoded)

        logger.debug('Killing rx_node')
        
    def stop(self):
        # TODO: Make sure that rx_node is running
        self.kill_rx.set()
        self.join()
        logger.debug('RX thread successfully exited')

class Dispatcher():
    def __init__(self, TX_to_RX, RX_to_TX):
        self.TX_to_RX = TX_to_RX
        self.RX_to_TX = RX_to_TX

class ReceiverDispatcher(Dispatcher):
    def __init__(self, TX_to_RX, RX_to_TX):
        super().__init__(TX_to_RX, RX_to_TX)
        
    def action(self, message):
        logger.debug(f"Decoded signal: {message}")
        received_preamble = message[:8]
        received_id = message[8:12]
        if np.array_equal(received_preamble, TCP_Protocol.preamble):
            # logger.debug(f'Data received:  {message[8:]}')
            logger.debug(f"ID received: {received_id}")
            if np.array_equal(received_id, TCP_Protocol.syn_id):
                logger.info("Received SYN Packet")
                self.RX_to_TX.put(NodeMessage('command', 'send syn_ack'))
            elif np.array_equal(received_id, TCP_Protocol.syn_ack_id):
                logger.info("Received SYN_ACK Packet")
                self.RX_to_TX.put(NodeMessage('command', 'send ack'))
            elif np.array_equal(received_id, TCP_Protocol.ack_id):
                logger.info("Received ACK Packet")
            else:
                logger.debug('Unrecognized ID found')
        else:
            logger.debug('Preamble missing')
            
class TransmitterDispatcher(Dispatcher):
    def __init__(self, transmitter, TX_to_RX, RX_to_TX):
        super().__init__(TX_to_RX, RX_to_TX)
        self.transmitter = transmitter
        self.protocol = TCP_Protocol(channel_freq=self.transmitter.tx_channel_freq)
    
    def action(self, message):
        logger.debug(f"TX_Node received {message}")
        match message:
            case None:
                return None
            case NodeMessage('command', 'send syn_ack'): # TODO: Freeze this class in initialiation to prevent constantly making the object to check
                logger.info('TX_Node sending SYN ACK Packet')
                self.transmitter.send(self.protocol.syn_ack)
                return True
            case NodeMessage('command', 'send ack'):
                logger.info('TX Node sending ACK Packet')
                self.transmitter.send(self.protocol.ack)
        
@dataclass
class NodeMessage():
    type: str
    id: str
                 
class Lime_RX_TX(Lime_RX, Lime_TX):
    def __init__(self, sample_rate, rx_freq, tx_freq, rx_antenna, tx_antenna, rx_channel_freq, tx_channel_freq, full_duplex=False):
        self.full_duplex = full_duplex
        self.rx_channel_freq = rx_channel_freq
        self.tx_channel_freq = tx_channel_freq
        super().__init__(sample_rate, rx_freq, rx_antenna)
        super(Lime_TX, self).__init__(sample_rate, tx_freq, tx_antenna)
        
    
    def __enter__(self):
        Lime_RX.__enter__(self)
        Lime_TX.__enter__(self)
        TX_to_RX = queue.Queue()
        RX_to_TX = queue.Queue()
        self.rx_node = RX_Node(self, self.rx_channel_freq, TX_to_RX, RX_to_TX)
        self.tx_node = TX_Node(self, TX_to_RX, RX_to_TX)
        if self.full_duplex:
            self.clear_read_buffer() # Specific to Lime devices it seems. More testing needed to be done with this.
            self.rx_node.start()
            self.tx_node.start()
        return self
        
    def __exit__(self, *args, **kwargs):
        if self.full_duplex:
            self.rx_node.stop()
            self.tx_node.stop()
        Lime_RX.__exit__(self, retain_sdr=True)
        Lime_TX.__exit__(self)
            
class UHD_RX_TX(UHD_RX, UHD_TX):
    def __init__(self, sample_rate, rx_freq, tx_freq, rx_antenna, tx_antenna, rx_channel_freq, tx_channel_freq, full_duplex=False):
        self.full_duplex = full_duplex
        self.rx_channel_freq = rx_channel_freq
        self.tx_channel_freq = tx_channel_freq
        super().__init__(sample_rate, rx_freq, rx_antenna)
        super(UHD_TX, self).__init__(sample_rate, tx_freq, tx_antenna)
        
    def __enter__(self):
        UHD_RX.__enter__(self)
        UHD_TX.__enter__(self)
        
        TX_to_RX = queue.Queue()
        RX_to_TX = queue.Queue()
        self.rx_node = RX_Node(self, self.rx_channel_freq, TX_to_RX, RX_to_TX)
        self.tx_node = TX_Node(self, TX_to_RX, RX_to_TX)
        if self.full_duplex:
            self.rx_node.start()
            self.tx_node.start()
        return self
        
    def __exit__(self, *args, **kwargs):
        if self.full_duplex:
            self.rx_node.stop()
            self.tx_node.stop()
        UHD_RX.__exit__(self)
        UHD_TX.__exit__(self)
        
    