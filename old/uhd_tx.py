from sdr_tools.classes import UHD_TX

from IPython import embed

if __name__ == "__main__":  
    sample_rate = 2e6
    center_freq = 434e6
    antenna = ''
    with UHD_TX(sample_rate, center_freq, antenna) as transmitter:
    
        freq = 40000
        freq_deviation = 10000
        symbol_length = 10000

        fm_packet = transmitter.generate_fm_packet('10100010', freq, freq_deviation, symbol_length)
        # bpsk = transmitter.generateBPSK('10100010')

        # streamer.send(transmission_signal)
        
        transmitter.set_gain(20)
        embed()