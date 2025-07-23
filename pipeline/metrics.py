"""
 Audio Processing Metrics - HIGH-LEVEL ANALYSIS
Comprehensive quality assessment for audio processing
"""

import numpy as np
import logging
from typing import Dict
from utils.audio_utils import estimate_noise_level_fast, calculate_rms_fast, calculate_peak_db

logger = logging.getLogger(__name__)

def calculate_snr_improvement(input_signal: np.ndarray, output_signal: np.ndarray) -> Dict:
    """
    Tính cải thiện SNR 
    
    Args:
        input_signal: Tín hiệu đầu vào (có nhiễu)
        output_signal: Tín hiệu đầu ra (đã xử lý)
        
    Returns:
        Dict chứa SNR metrics
    """
    try:
        # ============ NOISE LEVEL ESTIMATION ============
        input_noise_level = estimate_noise_level_fast(input_signal, percentile=10)
        output_noise_level = estimate_noise_level_fast(output_signal, percentile=10)
        
        # ============ SIGNAL LEVEL CALCULATION ============
        input_rms = calculate_rms_fast(input_signal)
        output_rms = calculate_rms_fast(output_signal)
        
        # ============ SNR ESTIMATION ============
        if input_noise_level > 0 and output_noise_level > 0:
            input_snr = 20 * np.log10((input_rms + 1e-10) / (input_noise_level + 1e-10))
            output_snr = 20 * np.log10((output_rms + 1e-10) / (output_noise_level + 1e-10))
            snr_improvement = output_snr - input_snr
        else:
            input_snr = 0.0
            output_snr = 0.0
            snr_improvement = 0.0
        
        return {
            'input_snr_db': input_snr,
            'output_snr_db': output_snr,
            'estimated_snr_improvement_db': snr_improvement,
            'input_noise_level': input_noise_level,
            'output_noise_level': output_noise_level
        }
        
    except Exception as e:
        logger.warning(f"⚠️ SNR improvement calculation failed: {e}")
        return {
            'input_snr_db': 0.0,
            'output_snr_db': 0.0,
            'estimated_snr_improvement_db': 0.0,
            'input_noise_level': 1e-6,
            'output_noise_level': 1e-6
        }

def calculate_noise_reduction(input_signal: np.ndarray, output_signal: np.ndarray) -> float:
    """
    Tính noise reduction trong dB
    """
    try:
        input_noise = estimate_noise_level_fast(input_signal)
        output_noise = estimate_noise_level_fast(output_signal)
        
        if input_noise > 0 and output_noise > 0:
            noise_reduction = 20 * np.log10(input_noise / output_noise)
        else:
            noise_reduction = 0.0
            
        return noise_reduction
        
    except Exception as e:
        logger.warning(f" Noise reduction calculation failed: {e}")
        return 0.0

def calculate_level_change(input_signal: np.ndarray, output_signal: np.ndarray) -> float:
    """
    Tính thay đổi level tổng thể
    """
    try:
        input_rms = calculate_rms_fast(input_signal)
        output_rms = calculate_rms_fast(output_signal)
        
        if input_rms > 0:
            level_change = 20 * np.log10((output_rms + 1e-10) / (input_rms + 1e-10))
        else:
            level_change = 0.0
            
        return level_change
        
    except Exception as e:
        logger.warning(f" Level change calculation failed: {e}")
        return 0.0

def generate_quality_report(input_signal: np.ndarray, output_signal: np.ndarray) -> Dict:
    """
    Tạo báo cáo chất lượng toàn diện
    
    Args:
        input_signal: Tín hiệu đầu vào
        output_signal: Tín hiệu đầu ra
        
    Returns:
        Dict chứa tất cả quality metrics
    """
    try:
        # ============ SNR ANALYSIS ============
        snr_metrics = calculate_snr_improvement(input_signal, output_signal)
        
        # ============ NOISE REDUCTION ============
        noise_reduction = calculate_noise_reduction(input_signal, output_signal)
        
        # ============ LEVEL ANALYSIS ============
        level_change = calculate_level_change(input_signal, output_signal)
        
        # ============ PEAK ANALYSIS ============
        input_peak_db = calculate_peak_db(input_signal)
        output_peak_db = calculate_peak_db(output_signal)
        
        # ============ COMPREHENSIVE REPORT ============
        return {
            # SNR Metrics
            'estimated_snr_improvement_db': snr_metrics['estimated_snr_improvement_db'],
            'input_snr_db': snr_metrics['input_snr_db'],
            'output_snr_db': snr_metrics['output_snr_db'],
            
            # Noise Metrics
            'estimated_noise_reduction_db': noise_reduction,
            
            # Level Metrics
            'level_change_db': level_change,
            'input_peak_db': input_peak_db,
            'output_peak_db': output_peak_db,
            
            # Quality Assessment
            'processing_quality': 'optimized'
        }
        
    except Exception as e:
        logger.warning(f" Quality report generation failed: {e}")
        return {
            'estimated_snr_improvement_db': 0.0,
            'input_snr_db': 0.0,
            'output_snr_db': 0.0,
            'estimated_noise_reduction_db': 0.0,
            'level_change_db': 0.0,
            'input_peak_db': -60.0,
            'output_peak_db': -60.0,
            'processing_quality': 'fallback'
        }