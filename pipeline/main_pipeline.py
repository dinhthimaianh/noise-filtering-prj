"""
 Main Pipeline Controller - OPTIMIZED VERSION
Fast 3-stage processing với performance optimization
"""

import numpy as np
import logging
import time
from typing import Dict, Any
from pipeline.metrics import generate_quality_report
logger = logging.getLogger(__name__)

class AudioProcessingPipeline:
    """
    Fast audio processing pipeline với optimized performance
    """
    
    def __init__(self):
        logger.info(" Khởi tạo Audio Processing Pipeline")
        
        # Import stages once
        from .stages.dsp_processor import DSPProcessor
        
        # Initialize stages
        try:
            self.dsp_processor = DSPProcessor()
            logger.info(" Stage 2: Fast DSP Processor initialized")

            
        except Exception as e:
            logger.error(f" Stage initialization failed: {e}")
            raise
        
        # Performance tracking
        self.processing_times = {}
        self.total_processed = 0
        
        # Set logging level cho production performance
        logging.getLogger().setLevel(logging.INFO)  # Reduce logging overhead
        
        logger.info(" Pipeline ready for high-performance processing")
    
    def process_audio(self, noisy_input: np.ndarray, environment: str,
                     sample_rate: int = 44100) -> Dict[str, Any]:
        """
        OPTIMIZED 2-stage audio processing
        
        Args:
            noisy_input: Input signal với noise
            environment: Environment type
            sample_rate: Sample rate
            
        Returns:
            Complete processing results
        """
        processing_start = time.time()
        
        audio_duration = len(noisy_input) / sample_rate
        logger.info(f" FAST processing: {len(noisy_input)} samples ({audio_duration:.1f}s), {environment}")
        
        try:

            # ============ STAGE 2: FAST DSP PROCESSING ============ HIỂU CODE Ở ĐÂY
            
            
            
            
            
            
            
            stage2_start = time.time()
            final_output = self.dsp_processor.process(
                noisy_signal=noisy_input,
                environment=environment,
                sample_rate=sample_rate
            )
            stage2_time = time.time() - stage2_start
            self.processing_times['stage2_dsp'] = stage2_time
            
            # ============ FAST METRICS CALCULATION ============
            metrics_start = time.time()
            metrics =  generate_quality_report(noisy_input, final_output)
            metrics_time = time.time() - metrics_start
            
            # ============ PERFORMANCE CALCULATION ============
            total_time = time.time() - processing_start
            real_time_factor = audio_duration / total_time
            
            # ============ RESULTS ASSEMBLY ============
            results = {
                # Audio stages
                'input_noisy': noisy_input,
                'mic_processed': noisy_input,
                'final_output': final_output,
                'estimated_clean': final_output,  # Use DSP output as clean estimate
                
                # Metadata
                'environment': environment,
                'sample_rate': sample_rate,
                
                # Performance metrics
                'processing_metadata': {
                    'total_processing_time': total_time,
                    'stage_times': self.processing_times.copy(),
                    'metrics_time': metrics_time,
                    'real_time_factor': real_time_factor,
                    'latency_ms': total_time * 1000,
                    'audio_duration_s': audio_duration,
                    'processing_efficiency_percent': (audio_duration / total_time) * 100,
                },
                
                # Quality metrics
                'quality_metrics': metrics
            }
            
            # ============ PERFORMANCE LOGGING ============
            self.total_processed += 1
            
            logger.info(f" FAST processing complete:")
            logger.info(f"     Total time: {total_time:.3f}s")
            logger.info(f"    Real-time factor: {real_time_factor:.1f}x")
            logger.info(f"    SNR improvement: {metrics['estimated_snr_improvement_db']:.1f}dB")
            
            return results
            
        except Exception as e:
            logger.error(f" Fast pipeline processing failed: {e}")
            raise
