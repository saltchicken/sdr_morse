import numpy as np
from scipy.signal import butter, lfilter, resample_poly
from dataclasses import dataclass

from core.utils import FM_Settings
from core.logging import logger


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
    
    # TODO: There may be an issue with calling this multiple times
    def shift_center(self, frequency):
        # wave_gen = cos_wave_generator(self.sample_rate, -frequency, len(self.data))
        wave_gen = ShiftFrequency(self.sample_rate, frequency, len(self.data))
        self.data = self.data * wave_gen.next()

class Packet(Segment): 
    def __init__(self, segment: Segment):
        super().__init__(segment.data, segment.sample_rate)  
    
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
    def __init__(self, segment: Segment):
        super().__init__(segment.data, segment.sample_rate)
        self.settings = FM_Settings()
        self.decode(self.settings.symbol_length)
        
    def decode_segment(self, segment:Segment):
        return (np.real(segment.data) < 0).astype(int) # Why is real needed    
        
    def decode(self, symbol_length):
        self.lowpass = Filter(self)
        self.demod = QuadDemod(self.lowpass)
        self.demod.data = self.demod.data[symbol_length//2:] # Offset the sample. Poverty synchronization
        self.resample = Resample(self.demod, 1, symbol_length)
        self.decoded = self.decode_segment(self.resample)
        # logger.debug(self.decoded)  
                