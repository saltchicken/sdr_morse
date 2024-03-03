from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np

from rxtx_logging import logger
import time


import uhd
# TODO: condense these imports
import SoapySDR
from SoapySDR import *

import threading, queue

from commpy.filters import rrcosfilter

from core.classes import Packet, Segment, ShiftFrequency, Decoded, TCP_Protocol

@dataclass
class NodeMessage():
    type: str
    id: str
    
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
        
    def capture_signal(self, kill_rx, channel_freq, threshold=0.005):
        shift_frequency = ShiftFrequency(self.sample_rate, channel_freq, len(self.read_buffer))
        signal = []
        while not kill_rx.is_set():
            sample = self.read()
            sample = sample * shift_frequency.next()
            if np.max(np.abs(sample)) >= threshold:
                if len(signal) == 0:
                    logger.debug(f"Signal found. Writing...")
                signal.append(sample)
            else:
                if len(signal) > 0:
                    break
        if kill_rx.is_set() and len(signal) == 0:
            logger.debug('Exiting capture signal')
            return None
        else:
            captured_signal = Segment(np.concatenate(signal), self.sample_rate)
            logger.debug(f"Returning captured signal. Signal contains {len(captured_signal.data)} samples")
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
            logger.warning('Preamble missing')
            
class TransmitterDispatcher(Dispatcher):
    def __init__(self, transmitter, TX_to_RX, RX_to_TX):
        super().__init__(TX_to_RX, RX_to_TX)
        self.transmitter = transmitter
        self.protocol = TCP_Protocol(channel_freq=self.transmitter.tx_channel_freq)
    
    def action(self, message: NodeMessage):
        logger.debug(f"TX_Node received {message}")
        match message:
            case None:
                return None
            case NodeMessage('command', 'send syn_ack'): # TODO: Freeze this class in initialiation to prevent constantly making the object to check
                logger.debug('TX_Node sending SYN ACK Packet')
                self.transmitter.send(self.protocol.syn_ack)
            case NodeMessage('command', 'send ack'):
                logger.debug('TX_Node sending ACK Packet')
                self.transmitter.send(self.protocol.ack)
            case NodeMessage('command', 'send syn'):
                logger.debug('TX_Node sending ACK Packet')
                self.transmitter.send(self.protocol.syn)
                 
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
        self.TX_to_RX = queue.Queue()
        self.RX_to_TX = queue.Queue()
        self.rx_node = RX_Node(self, self.rx_channel_freq, self.TX_to_RX, self.RX_to_TX)
        self.tx_node = TX_Node(self, self.TX_to_RX, self.RX_to_TX)
        if self.full_duplex:
            self.clear_read_buffer() # Specific to Lime devices it seems. More testing needed to be done with this.
            self.rx_node.start()
            self.tx_node.start()
        return self
        
    def __exit__(self, *args, **kwargs):
        # TODO: Should queue.Queues be closed?
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
        self.TX_to_RX = queue.Queue()
        self.RX_to_TX = queue.Queue()
        self.rx_node = RX_Node(self, self.rx_channel_freq, self.TX_to_RX, self.RX_to_TX)
        self.tx_node = TX_Node(self, self.TX_to_RX, self.RX_to_TX)
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
