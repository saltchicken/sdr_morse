from sdr_tools import classes
from sdr_tools import utils

def setup():
    sample_rate = 1e6
    center_freq = 434e6
    streamer = classes.UHD_TX_Streamer(sample_rate, center_freq)
    return sample_rate, center_freq, streamer

if __name__ == "__main__":
    sample_rate, center_freq, streamer = setup()

    # freq = 50000
    # freq_deviation = 10000
    # duration = 1
    # symbol_size = 250000

    # transmission_signal = utils.generate_fm_packet(streamer, '1010001', freq, freq_deviation, symbol_size / sample_rate)
    
    freq = 50000
    freq_deviation = 10000
    duration = 3

    transmission_signal = utils.generate_fm_packet_update('10100010', freq - freq_deviation, freq + freq_deviation, duration, sample_rate)

    # plt.plot(transmission_signal, label="Transmission")
    # plt.xlabel("Index")
    # plt.ylabel("Amplitude")
    # plt.legend()
    # plt.show()


    streamer.send(transmission_signal)
    # time.sleep(0.25)
    # streamer.send(deviation_signal)