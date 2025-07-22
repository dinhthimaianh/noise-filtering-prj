"""
Main Audio Processing Pipeline
Coordinates all 6 stages of processing with comprehensive error handling and monitoring
"""

import numpy as np
import logging
import time
from typing import Dict, Any, Optional, List
import warnings

from utils.config import ENVIRONMENT_CONFIGS


# Import all processing stages
from .stages.analog_filter import AnalogFilter
from .stages.adc import ADC
from .stages.dsp_processor import DSPProcessor
from .stages.dac import DAC
from .stages.reconstruction import ReconstructionFilter

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioProcessingPipeline:
    """
    Main pipeline coordinator for 6-stage audio processing
    Provides comprehensive processing with monitoring and error recovery
    """
    
    def __init__(self):
        logger.info("Initializing Audio Processing Pipeline v1.0")
        
        # Initialize all processing stages
        try:
            self.analog_filter = AnalogFilter()
            self.adc = ADC()
            self.dsp_processor = DSPProcessor()
            self.dac = DAC()
            self.reconstruction_filter = ReconstructionFilter()
            logger.info(" All processing stages initialized successfully")
        except Exception as e:
            logger.error(f" Stage initialization failed: {e}")
            raise
        

    def process_audio(self, audio_input: np.ndarray, environment: str, 
                     sample_rate: int = 44100, monitor_performance: bool = True) -> Dict[str, Any]:
        """
        Main processing function - coordinates all 6 stages
        
        Args:
            audio_input: Raw audio signal (from microphone/file)
            environment: Selected environment type
            sample_rate: Sample rate of input audio
            monitor_performance: Enable performance monitoring
            
        Returns:
            Dict containing processed audio and comprehensive metadata
        """
        
        processing_start_time = time.time()

        logger.info("="*60)
        logger.info("STARTING AUDIO PROCESSING PIPELINE")
        logger.info(f"Environment: {environment}")
        logger.info(f"Input: {len(audio_input)} samples at {sample_rate} Hz")
        logger.info(f"Duration: {len(audio_input)/sample_rate:.2f}s")
        logger.info("="*60)
        
        try:
            # Input validation
            self._validate_inputs(audio_input, environment, sample_rate)
           
            
        except Exception as e:
            logger.error(f"CRITICAL ERROR in audio processing pipeline: {str(e)}")
            logger.error(f"Processing failed after {time.time() - processing_start_time:.3f} seconds")
            raise
    
    def _validate_inputs(self, audio_input: np.ndarray, environment: str, sample_rate: int):
        """Comprehensive input validation"""
        
        # Audio validation
        if len(audio_input) == 0:
            raise ValueError("Empty audio input")
        
        if not np.isfinite(audio_input).all():
            raise ValueError("Audio contains invalid values (NaN or infinity)")
        
        if np.max(np.abs(audio_input)) == 0:
            logger.warning("Input signal is silent")
        
        # Environment validation
        if environment not in ENVIRONMENT_CONFIGS:
            raise ValueError(f"Unknown environment: {environment}")
        
        # Sample rate validation
        if sample_rate <= 0 or sample_rate > 192000:
            raise ValueError(f"Invalid sample rate: {sample_rate}")
        
        # Real-time constraint checking
        if self.real_time_mode:
            duration_ms = len(audio_input) / sample_rate * 1000
            if duration_ms > self.max_latency_ms * 10:  # Allow 10x latency for processing
                logger.warning(f"Input duration {duration_ms:.1f}ms may exceed real-time constraints")
    
   