from core.transceiver import Lime_RX_TX, UHD_RX_TX, NodeMessage
from core.display import Displayer

import sys
import argparse
from IPython import embed
from core.logging import logger

def main():
    parser = argparse.ArgumentParser(description="A simple script to demonstrate argument parsing.")
    parser.add_argument('--device', '-d', type=str, required=True, help="Device string (lime | uhd)")
    parser.add_argument('--sample_rate', type=float, required=True, help="Sample rate (Hz). Example: 2e6")
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
            rx_antenna = ''
            tx_antenna = ''

            with UHD_RX_TX(args.sample_rate, 
                           args.rx_center, 
                           args.tx_center, 
                           rx_antenna, 
                           tx_antenna, 
                           args.rx_channel, 
                           args.tx_channel, 
                           full_duplex=True) as transceiver:
                embed(quiet=True)
        
        case 'lime':
            rx_antenna = 'LNAW'
            tx_antenna = 'BAND2'

            with Lime_RX_TX(args.sample_rate, 
                            args.rx_center, 
                            args.tx_center, 
                            rx_antenna, 
                            tx_antenna, 
                            args.rx_channel, 
                            args.tx_channel, 
                            full_duplex=True) as transceiver:
                embed(quiet=True)


if __name__ == "__main__":
    main()

