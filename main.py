from sdr_tools.classes import Lime_RX_TX, UHD_RX_TX, TCP_Protocol, logger

import sys
import argparse
from IPython import embed

def main():
    parser = argparse.ArgumentParser(description="A simple script to demonstrate argument parsing.")
    parser.add_argument('--device', '-d', type=str, required=True, help="Device string (lime | uhd)")
    parser.add_argument('--sr', type=float, required=True, help="Sample rate (Hz). Example: 2e6")
    parser.add_argument('--rx_center', type=float, required=True, help="Center frequency for receiver (Hz). Example: 434e6")
    parser.add_argument('--rx_channel', type=float, required=True, help="Channel frequency for recevier. Offset from center (Hz). Example: 40000")
    parser.add_argument('--tx_center', type=float, required=True, help="Center frequency for transmitter (Hz). Example: 434e6")
    parser.add_argument('--tx_channel', type=float, required=True, help="Channel frequency for transmitter. Offset from center (Hz). Example: 25000")
    parser.add_argument('--verbose', '-v', action='store_true', help="Enable verbose mode")

    args = parser.parse_args()

    logger.remove()
    if args.verbose:
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.add(sys.stderr, level="INFO")
    # TODO: Refactor these match cases to not repeat all of the setup infomation. Possibly create config file
    match args.device:
        case 'uhd':
            sample_rate = args.sr
            rx_freq = args.rx_center
            tx_freq = args.tx_center
            rx_channel = args.rx_channel
            tx_channel = args.tx_channel
            
            rx_antenna = ''
            tx_antenna = ''

            with UHD_RX_TX(sample_rate, rx_freq, tx_freq, rx_antenna, tx_antenna, rx_channel, tx_channel, full_duplex=True) as transceiver:
                protocol = TCP_Protocol(channel_freq=25000)
                fm_packet = protocol.syn
                embed(quiet=True)
        case 'lime':
            sample_rate = args.sr
            rx_freq = args.rx_center
            tx_freq = tx_freq = args.tx_center
            rx_channel = args.rx_channel
            tx_channel = args.tx_channel
            
            antenna = 'LNAW'

            with Lime_RX_TX(sample_rate, rx_freq, tx_freq, antenna, 'BAND2', rx_channel, tx_channel, full_duplex=True) as transceiver:
                protocol = TCP_Protocol(channel_freq=40000)
                fm_packet = protocol.syn
                embed(quiet=True)
            pass

if __name__ == "__main__":
    main()

