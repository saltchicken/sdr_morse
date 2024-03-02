from sdr_tools.classes import Lime_RX_TX, UHD_RX_TX, TCP_Protocol, logger

import sys
import argparse
from IPython import embed

def main():
    parser = argparse.ArgumentParser(description="A simple script to demonstrate argument parsing.")
    parser.add_argument('--device', '-d', type=str, required=True, help="Device string (lime | uhd)")
    parser.add_argument('--verbose', '-v', action='store_true', help="Enable verbose mode")

    args = parser.parse_args()

    if args.verbose:
        print("Verbose mode is enabled.") # TODO: Remove this print
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.remove()
        logger.add(sys.stderr, level="INFO")

    match args.device:
        case 'uhd':
            sample_rate = 2e6
            rx_freq = 434e6 # frequency
            rx_antenna = ''
            tx_antenna = ''

            rx_channel = 40000
            tx_channel = 25000

            tx_freq = 434e6 # center_freq

            with UHD_RX_TX(sample_rate, rx_freq, tx_freq, rx_antenna, tx_antenna, rx_channel, tx_channel, full_duplex=True) as transceiver:
                protocol = TCP_Protocol(channel_freq=25000)
                fm_packet = protocol.syn
                embed(quiet=True)
        case 'lime':
            # apply settings
            sample_rate = 2e6
            rx_freq = 434e6 # frequency
            antenna = 'LNAW'

            rx_channel = 25000
            tx_channel = 40000

            tx_freq = 434e6 # center_freq

            with Lime_RX_TX(sample_rate, rx_freq, tx_freq, antenna, 'BAND2', rx_channel, tx_channel, full_duplex=True) as transceiver:
                protocol = TCP_Protocol(channel_freq=40000)
                fm_packet = protocol.syn
                embed(quiet=True)
            pass

if __name__ == "__main__":
    main()

