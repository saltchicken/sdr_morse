import classes

class Displayer():
    def __init__(self):
        pass
    
    @staticmethod
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
    
    @staticmethod        
    def plot(self):
        plt.plot(self.data)
        plt.show()
        
    #TODO: This function needs a LOT of refactoring. Too much experimentation to figure out solution.
    @staticmethod
    def uhd_waterfall(self, iterations=1000, buffer_size=2000, fft_size=256, decimator=4):
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
        
        @staticmethod
        def lime_waterfall(self, iterations=1000, buffer_size=1024, fft_size=256, decimator=4):
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
            
        # TODO: This does not work for UHD_RX. Look for efficient TODO about iterating the read inside UHD_RX 
        # TODO: Remove hardcoded frequency_shift of 40000
        @staticmethod
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