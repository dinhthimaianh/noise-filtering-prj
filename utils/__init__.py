"""
Configuration settings cho noise filtering project
Compatible với Python 3.11.x
"""

from typing import Dict, List, Union
import sys

# Check Python version
if sys.version_info < (3, 11):
    raise RuntimeError("This project requires Python 3.11 or higher")

# Environment configurations with type hints
ENVIRONMENT_CONFIGS: Dict[str, Dict[str, Union[str, float, int, bool, List[int]]]] = {

    'office': {
        'name': 'Văn phòng',
        'description': 'Lọc tiếng điều hòa, bàn phím',
        'filter_type': 'minimal',
        'highpass_cutoff': 80,
        'lowpass_cutoff': 8000,
        'compression_ratio': 3.0,
        'noise_gate_threshold': -40,
        'notch_frequencies': [50, 60, 120],
        'gain_reduction': 0.9,
        'gentle_processing': True
    },
    'meeting_room': {
        'name': 'Phòng họp',
        'description': 'Lọc tiếng nói, tiếng máy chiếu',
        'filter_type': 'balanced',
        'highpass_cutoff': 100,
        'lowpass_cutoff': 12000,
        'compression_ratio': 2.0,
        'noise_gate_threshold': -30,
        'notch_frequencies': [50, 60],
        'gain_reduction': 0.85,
        'natural_sound': False
    },

}

# ADC/DAC Settings
ADC_SETTINGS: Dict[str, Union[int, float]] = {
    'bit_depth': 16,
    'quantization_levels': 2**16,
    'full_scale_voltage': 1.0,
    'sample_rate': 44100,
    'dither_enabled': True
}

# Filter Settings
FILTER_SETTINGS: Dict[str, Union[int, float]] = {
    'analog_filter_order': 8,
    'reconstruction_filter_order': 6,
    'anti_aliasing_ratio': 0.45,
    'reconstruction_ratio': 0.4,
    'transition_bandwidth': 0.1
}

# Processing Settings
PROCESSING_SETTINGS: Dict[str, Union[int, float, str]] = {
    'frame_size': 2048,
    'overlap_ratio': 0.5,
    'window_type': 'hann',
    'fft_size': 4096,
    'spectral_floor': -80.0,
    'max_gain_db': 20.0
}

# Performance Settings
PERFORMANCE_SETTINGS: Dict[str, Union[int, float, bool]] = {
    'real_time_processing': True,
    'max_latency_ms': 50.0,
    'buffer_size': 512,
    'threading_enabled': True,
    'cache_filters': True
}

# File I/O Settings
FILE_SETTINGS: Dict[str, Union[str, int, List[str]]] = {
    'supported_formats': ['wav', 'mp3', 'flac', 'm4a', 'ogg'],
    'max_file_size_mb': 50,
    'default_sample_rate': 44100,
    'temp_dir': 'temp_audio',
    'output_format': 'wav'
}

def get_environment_config(environment: str) -> Dict:
    """Get configuration for specific environment"""
    return ENVIRONMENT_CONFIGS.get(environment, ENVIRONMENT_CONFIGS['office'])

def validate_config() -> bool:
    """Validate configuration settings"""
    try:
        # Check all environments have required keys
        required_keys = ['name', 'filter_type', 'highpass_cutoff', 'lowpass_cutoff']
        
        for env_id, config in ENVIRONMENT_CONFIGS.items():
            for key in required_keys:
                if key not in config:
                    raise ValueError(f"Missing required key '{key}' in environment '{env_id}'")
        
        return True
    except Exception as e:
        print(f"Configuration validation failed: {e}")
        return False

# Validate configuration on import
if not validate_config():
    raise RuntimeError("Invalid configuration detected")