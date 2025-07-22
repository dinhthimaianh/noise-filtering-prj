"""
Audio Processing Pipeline Package
6-Stage DSP Pipeline for Environment-Aware Noise Filtering
"""

from .main_pipeline import AudioProcessingPipeline
from .stages.analog_filter import AnalogFilter
from .stages.adc import ADC  
from .stages.dsp_processor import DSPProcessor
from .stages.dac import DAC
from .stages.reconstruction import ReconstructionFilter

__version__ = "1.0.0"
__author__ = "DSP Group 3"

__all__ = [
    'AudioProcessingPipeline',
    'AnalogFilter',
    'ADC',
    'DSPProcessor', 
    'DAC',
    'ReconstructionFilter'
]