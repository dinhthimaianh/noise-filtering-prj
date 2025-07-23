"""
Audio Utilities - OPTIMIZED VERSIONS
Fast audio processing helper functions
"""

import numpy as np
import logging
from typing import Tuple, Dict
from scipy.fft import fft

logger = logging.getLogger(__name__)

def compute_frequency_response_fast(signal: np.ndarray, sample_rate: int, 
                                  max_points: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    """
    FAST frequency response calculation với limited resolution cho speed
    
    Args:
        signal: Input signal
        sample_rate: Sample rate
        max_points: Maximum frequency points (trade-off: resolution vs speed)
        
    Returns:
        (frequencies, magnitude_db)
    """
    try:
        # ============ DOWNSAMPLE LONG SIGNALS ============
        if len(signal) > max_points * 4:
            # Downsample để reduce FFT size
            step = len(signal) // (max_points * 2)
            signal = signal[::step]
            logger.debug(f"Downsampled signal: {len(signal)} points")
        
        # ============ FAST FFT ============
        fft_result = fft(signal)
        frequencies = np.fft.fftfreq(len(signal), 1/sample_rate)
        
        # Take only positive frequencies
        positive_freqs = frequencies[:len(frequencies)//2]
        magnitude_db = 20 * np.log10(np.abs(fft_result[:len(fft_result)//2]) + 1e-10)
        
        # ============ LIMIT OUTPUT RESOLUTION ============
        if len(positive_freqs) > max_points:
            # Downsample frequency response
            indices = np.linspace(0, len(positive_freqs)-1, max_points, dtype=int)
            positive_freqs = positive_freqs[indices]
            magnitude_db = magnitude_db[indices]
        
        logger.debug(f"Fast frequency response: {len(positive_freqs)} points")
        return positive_freqs, magnitude_db
        
    except Exception as e:
        logger.warning(f"Fast frequency response failed: {e}")
        # Fallback - return minimal data
        freqs = np.linspace(0, sample_rate//2, max_points)
        mags = np.zeros(max_points) - 60  # -60dB default
        return freqs, mags

def analyze_audio_properties_fast(signal: np.ndarray, sample_rate: int) -> Dict:
    """
    UPDATED: Sử dụng core functions
    """
    try:
        duration = len(signal) / sample_rate
        rms_db = calculate_rms_db(signal)
        peak_db = calculate_peak_db(signal)
        
        # Dynamic range
        if rms_db > -60 and peak_db > -60:
            dynamic_range = peak_db - rms_db
        else:
            dynamic_range = 0.0
        
        # Quick frequency estimate (existing code)
        if len(signal) > 1000:
            sample_portion = signal[:1000]
            zero_crossings = np.sum(np.diff(np.sign(sample_portion)) != 0)
            estimated_frequency = (zero_crossings / 2) * (sample_rate / 1000)
        else:
            estimated_frequency = 0.0
        
        return {
            'duration_s': duration,
            'sample_rate': sample_rate,
            'num_samples': len(signal),
            'rms_level_db': rms_db,
            'peak_level_db': peak_db,
            'dynamic_range_db': dynamic_range,
            'estimated_fundamental_hz': estimated_frequency,
            'quality_grade': _get_audio_quality_grade(rms_db, peak_db, dynamic_range),
            'analysis_type': 'fast'
        }
        
    except Exception as e:
        logger.warning(f" Audio analysis failed: {e}")
        return {
            'duration_s': len(signal) / sample_rate if sample_rate > 0 else 0,
            'sample_rate': sample_rate,
            'num_samples': len(signal),
            'rms_level_db': -60.0,
            'peak_level_db': -60.0,
            'dynamic_range_db': 0.0,
            'estimated_fundamental_hz': 0.0,
            'quality_grade': 'Unknown',
            'analysis_type': 'fallback'
        }

def estimate_noise_level_fast(signal: np.ndarray, percentile: float = 10.0) -> float:
    """
    FAST noise level estimation using percentile method
    
    Args:
        signal: Input signal
        percentile: Percentile for noise floor estimation
        
    Returns:
        Estimated noise level
    """
    try:
        # Use percentile of absolute values - much faster than complex analysis
        noise_level = np.percentile(np.abs(signal), percentile)
        return noise_level
        
    except Exception as e:
        logger.warning(f"Fast noise estimation failed: {e}")
        return 1e-6  # Very small default



def estimate_processing_improvement(input_signal: np.ndarray, output_signal: np.ndarray) -> Dict:
    """
    Ước lượng cải thiện xử lý mà KHÔNG cần clean reference
    
    Args:
        input_signal: Tín hiệu đầu vào (có nhiễu)
        output_signal: Tín hiệu đầu ra (đã xử lý)
        
    Returns:
        Dict với các metrics cải thiện
    """
    try:
        # ============ ƯỚC LƯỢNG NOISE LEVELS ============
        input_noise_level = estimate_noise_level_fast(input_signal, percentile=10)
        output_noise_level = estimate_noise_level_fast(output_signal, percentile=10)
        
        # ============ TÍNH SIGNAL LEVELS ============
        input_rms = np.sqrt(np.mean(input_signal**2))
        output_rms = np.sqrt(np.mean(output_signal**2))
        
        # ============ ƯỚC LƯỢNG SNR ============
        if input_noise_level > 0 and output_noise_level > 0:
            input_snr_est = 20 * np.log10((input_rms + 1e-10) / (input_noise_level + 1e-10))
            output_snr_est = 20 * np.log10((output_rms + 1e-10) / (output_noise_level + 1e-10))
            snr_improvement = output_snr_est - input_snr_est
        else:
            input_snr_est = 0.0
            output_snr_est = 0.0
            snr_improvement = 0.0
        
        # ============ NOISE REDUCTION ============
        if input_noise_level > 0:
            noise_reduction = 20 * np.log10(input_noise_level / (output_noise_level + 1e-10))
        else:
            noise_reduction = 0.0
        
        # ============ LEVEL CHANGE ============
        if input_rms > 0:
            level_change = 20 * np.log10((output_rms + 1e-10) / (input_rms + 1e-10))
        else:
            level_change = 0.0
        
        return {
            'estimated_snr_improvement_db': snr_improvement,
            'estimated_noise_reduction_db': noise_reduction,
            'level_change_db': level_change,
            'input_snr_estimate_db': input_snr_est,
            'output_snr_estimate_db': output_snr_est,
            'input_noise_level': input_noise_level,
            'output_noise_level': output_noise_level
        }
        
    except Exception as e:
        logger.warning(f"Processing improvement estimation failed: {e}")
        return {
            'estimated_snr_improvement_db': 0.0,
            'estimated_noise_reduction_db': 0.0,
            'level_change_db': 0.0,
            'input_snr_estimate_db': 0.0,
            'output_snr_estimate_db': 0.0,
            'input_noise_level': 1e-6,
            'output_noise_level': 1e-6
        }
        
def _get_audio_quality_grade(rms_db: float, peak_db: float, dynamic_range: float) -> str:
    """
    Grade audio quality based on measurements
    
    Args:
        rms_db: RMS level in dB
        peak_db: Peak level in dB  
        dynamic_range: Dynamic range in dB
        
    Returns:
        Quality grade string
    """
    # Simple grading based on levels and dynamic range
    if peak_db > -6 and dynamic_range > 20:
        return "Excellent"
    elif peak_db > -12 and dynamic_range > 15:
        return "Very Good"
    elif peak_db > -20 and dynamic_range > 10:
        return "Good"
    elif peak_db > -30:
        return "Fair"
    else:
        return "Poor"

def estimate_noise_level_fast(signal: np.ndarray, percentile: float = 10.0) -> float:
    """
    CORE FUNCTION: Ước lượng noise level bằng percentile method
    
    Args:
        signal: Input signal
        percentile: Percentile cho noise floor estimation (default 10%)
        
    Returns:
        Estimated noise level
    """
    try:
        noise_level = np.percentile(np.abs(signal), percentile)
        return noise_level
    except Exception as e:
        logger.warning(f"Noise estimation failed: {e}")
        return 1e-6

def calculate_rms_fast(signal: np.ndarray) -> float:
    """
    CORE FUNCTION: Tính RMS nhanh
    """
    try:
        return np.sqrt(np.mean(signal**2))
    except Exception as e:
        logger.warning(f"RMS calculation failed: {e}")
        return 1e-6

def calculate_peak_db(signal: np.ndarray) -> float:
    """
    CORE FUNCTION: Tính peak level trong dB
    """
    try:
        peak = np.max(np.abs(signal))
        return 20 * np.log10(peak + 1e-10)
    except Exception as e:
        logger.warning(f"Peak calculation failed: {e}")
        return -60.0

def calculate_rms_db(signal: np.ndarray) -> float:
    """
    CORE FUNCTION: Tính RMS level trong dB
    """
    try:
        rms = calculate_rms_fast(signal)
        return 20 * np.log10(rms + 1e-10)
    except Exception as e:
        logger.warning(f"RMS dB calculation failed: {e}")
        return -60.0
