"""
Stage 2: DSP Processor - IMPLEMENTED NOISE FILTERING
Actual noise reduction algorithms
"""

import numpy as np
import scipy.signal as signal
import logging
from utils.config import ENVIRONMENT_CONFIGS
import librosa

logger = logging.getLogger(__name__)

class DSPProcessor:
    """
    DSP processor with actual noise filtering algorithms
    """
    
    def __init__(self):
        logger.info("DSP Processor initialized with noise filtering algorithms")
        
        # Spectral subtraction parameters
        self.alpha = 2.0  # Over-subtraction factor
        self.beta = 0.01  # Spectral floor
        self.frame_size = 1024
        self.hop_size = 512
        
        # Wiener filter parameters
        self.noise_estimation_frames = 10  # First N frames for noise estimation
        
    def process(self, noisy_signal: np.ndarray, environment: str, 
               sample_rate: int = 44100) -> np.ndarray:
        """
        Main DSP processing function with actual noise filtering
        
        Args:
            noisy_signal: Noisy audio from Stage 1
            environment: Environment type
            sample_rate: Audio sample rate
            
        Returns:
            Enhanced audio signal
        """
        logger.info(f"DSP processing for {environment} environment")
        
        final_signal = self.combined_noise_reduction(noisy_signal, sample_rate)

        return final_signal

    def combined_noise_reduction(self, noisy, sr, noise_start=0.0, noise_duration=0.005):
        frame_length = int(noise_duration * sr)
        if frame_length > len(noisy):
            frame_length = len(noisy)
        hop_length = frame_length // 2

        # ----- Noise detection (auto or manual) -----
        energy = librosa.feature.rms(y=noisy, frame_length=frame_length, hop_length=hop_length)[0]
        low_energy_frames = np.where(energy < 0.01)[0]

        if len(low_energy_frames) > 10:
            noise_start_sample = low_energy_frames[0] * hop_length
            noise_segment = noisy[noise_start_sample:noise_start_sample + sr // 2]
        else:
            start_sample = int(noise_start * sr)
            end_sample = int((noise_start + noise_duration) * sr)
            noise_segment = noisy[start_sample:end_sample]

        # ----- Step 1: Spectrum Subtraction -----
        stft_noisy = librosa.stft(noisy, n_fft=frame_length, hop_length=hop_length)
        noise_stft = librosa.stft(noise_segment, n_fft=frame_length, hop_length=hop_length)

        noise_mag = np.mean(np.abs(noise_stft), axis=1, keepdims=True)
        mag_noisy = np.abs(stft_noisy)
        phase = np.angle(stft_noisy)

        # Subtract noise magnitude spectrum
        mag_ss = mag_noisy - noise_mag
        mag_ss = np.maximum(mag_ss, 1e-10)

        # ISTFT to get intermediate signal
        stft_ss = mag_ss * np.exp(1j * phase)
        ss_result = librosa.istft(stft_ss, hop_length=hop_length)

        # ----- Step 2: Wiener Filtering on SS output -----
        stft_ss_again = librosa.stft(ss_result, n_fft=frame_length, hop_length=hop_length)
        mag_ss_again = np.abs(stft_ss_again)
        phase_ss = np.angle(stft_ss_again)
        psd_ss = mag_ss_again**2

        noise_psd = np.mean(np.abs(noise_stft)**2, axis=1, keepdims=True)

        # Wiener gain function
        H = psd_ss / (psd_ss + noise_psd)
        mag_wiener = H * mag_ss_again

        # Final signal
        stft_final = mag_wiener * np.exp(1j * phase_ss)
        final_signal = librosa.istft(stft_final, hop_length=hop_length)

        return final_signal
