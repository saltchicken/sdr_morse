import numpy as np

from core.segments import TCP_Protocol
from core.utils import NodeMessage
from core.logging import logger

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
 