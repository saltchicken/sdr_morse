from sdr_tools import classes
from sdr_tools import utils

# apply settings
sample_rate = 2e6
frequency = 433.5e6
antenna = 'LNAW'

buffer_size = 10000000

receiver = classes.Receiver(sample_rate, frequency, antenna, buffer_size)

frequency = -500000  # Adjust the frequency as needed
wave_gen = utils.cos_wave_generator(sample_rate, frequency, buffer_size)

received = receiver.read()

modulated = received * next(wave_gen)

output_file = 'samples.bin'
modulated.tofile(output_file)

print('closing time')
receiver.close()