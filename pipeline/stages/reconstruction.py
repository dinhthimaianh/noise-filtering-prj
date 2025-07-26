import logging
import numpy as np
from scipy.signal import butter, lfilter

logger = logging.getLogger(__name__)

class ReconstructionFilter:
    ''' Simulates reconstruction filter stage
    '''
    def __init__(self):
        logger.info("DAC Reconstruction initialized (BASE IMPLEMENTATION)")
        # Team member will add their algorithm parameters here
 
    def digital_to_analog(self, digital_signal, fs, cutoff_freq, order=5):
        """
        Chuyển đổi tín hiệu Digital sang Analog bằng mô phỏng DAC và lọc tín hiệu.
        - digital_signal: mảng tín hiệu số (numpy array)
        - fs: tần số lấy mẫu (Hz)
        - cutoff_freq: tần số cắt của lowpass filter (Hz)
        - order: bậc của bộ lọc
        Trả về: tín hiệu analog đã lọc
        """
        # Mô phỏng DAC: Giữ mẫu (Zero-Order Hold)
        t = np.arange(len(digital_signal)) / fs
        analog_signal = np.repeat(digital_signal, 1)
        t_analog = np.linspace(0, t[-1], len(analog_signal))

        # Thiết kế bộ lọc lowpass (Reconstruction Filter - Anti-imaging)
        nyq = 0.5 * fs
        normal_cutoff = cutoff_freq / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        filtered_signal = lfilter(b, a, analog_signal)

        return t_analog, filtered_signal

    def process(self, digital_signal, environment: str, sample_rate: int = 44100):
        logger.info(f"DAC reconstruction for {environment} environment")
        # Chọn tần số cắt phù hợp (ví dụ 0.45 Nyquist)
        cutoff =  int(0.45 * (sample_rate / 2))
        # Áp dụng hàm digital_to_analog cho tín hiệu audio
        t_audio_analog, audio_analog = self.digital_to_analog(digital_signal, sample_rate, cutoff)
        # Trả về chỉ tín hiệu analog (hoặc tuple nếu cần thời gian)
        return audio_analog
    