"""
 Stage 1: Analog Input - OPTIMIZED VERSION
Fast microphone characteristics simulation
"""

import numpy as np
import scipy.signal as sp_signal
from typing import Dict
import logging

logger = logging.getLogger(__name__)

class AnalogInput:
    """
    Fast analog input simulation với pre-computed filters
    """
    
    def __init__(self):
        logger.info(" Khởi tạo Fast Analog Input Stage")
        
        # Pre-compute filter coefficients để tránh tính lại
        self.sample_rate = 44100  # Cache sample rate
        self._precompute_filters()
        
        # Environment-specific gains (lookup table cho speed)
        self.environment_gains = {
            'hospital': 1.5,
            'airport': 2.0, 
            'cafe': 1.8,
            'office': 1.2,
            'street': 2.2,
            'home': 1.0
        }
        
        logger.info(" Fast Analog Input Stage sẵn sàng")
    
    def _precompute_filters(self):
        """Pre-compute filter coefficients - CHỈ 1 LẦN"""
        sr = self.sample_rate
        
        try:
            # Essential filters only
            self.highpass_sos = sp_signal.butter(2, 50, btype='high', fs=sr, output='sos')
            self.lowpass_sos = sp_signal.butter(2, 15000, btype='low', fs=sr, output='sos')
            
            # Simple presence boost
            self.presence_sos = sp_signal.butter(1, [2000, 4000], btype='band', fs=sr, output='sos')
            
            logger.debug(" Pre-computed filters ready")
            
        except Exception as e:
            logger.warning(f" Filter pre-computation failed: {e}")
            # Fallback - will use quick_bandpass
            self.highpass_sos = None
            self.lowpass_sos = None
            self.presence_sos = None
    
    def apply_mic_characteristics(self, noisy_input: np.ndarray, environment: str, 
                                sample_rate: int = 44100) -> np.ndarray:
        """
        FAST mic characteristics processing
        
        Args:
            noisy_input: Input signal với noise
            environment: Environment type cho gain lookup
            sample_rate: Sample rate (prefer 44100 cho pre-computed filters)
            
        Returns:
            Processed signal với mic characteristics
        """
        logger.debug(f"🎤 Fast mic processing: {environment}")
        
        # 1. Quick environment-based gain (O(1) lookup)
        gain = self.environment_gains.get(environment, 1.5)
        processed = noisy_input * gain
        
        # 2. Fast filtering
        if sample_rate == self.sample_rate and self.highpass_sos is not None:
            # Use pre-computed filters - FASTEST path
            processed = sp_signal.sosfilt(self.highpass_sos, processed)
            processed = sp_signal.sosfilt(self.lowpass_sos, processed)
            
            # Optional presence boost cho speech environments
            if environment in ['hospital', 'office', 'cafe']:
                presence = sp_signal.sosfilt(self.presence_sos, processed)
                processed = processed + presence * 0.1  # Subtle boost
                
        else:
            # Fallback cho different sample rates
            processed = self._quick_bandpass(processed, sample_rate)
        
        # 3. Simple clipping (fastest limiting)
        processed = np.clip(processed, -0.95, 0.95)
        
        logger.debug("✅ Fast mic processing complete")
        return processed
    
    def _quick_bandpass(self, signal: np.ndarray, sr: int) -> np.ndarray:
        """Quick bandpass fallback cho different sample rates"""
        try:
            # Simple 1st order highpass only
            sos = sp_signal.butter(1, 100, btype='high', fs=sr, output='sos')
            return sp_signal.sosfilt(sos, signal)
        except Exception as e:
            logger.warning(f"Quick bandpass failed: {e}")
            return signal
    
