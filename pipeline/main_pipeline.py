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
import webrtcvad
import sys
from pydub import AudioSegment

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
    
    def process_audio(self, noisy_input: np.ndarray, noisy_input_original, environment: str,
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
            RATE = 16000
            FRAME_DURATION_MS = 20
            FRAMES_PER_BUFFER = int(RATE * FRAME_DURATION_MS / 1000)  # 16000 Hz, 20 ms frames, 320 samples per frame
            FORMAT_WIDTH = 2
            CHANNELS = 1
            vad = webrtcvad.Vad(2)

            noisy_input_original = noisy_input_original.set_channels(CHANNELS).set_frame_rate(RATE).set_sample_width(FORMAT_WIDTH)
            noisy_input_original = noisy_input_original.raw_data  # Get raw audio data as bytes like b'\x1a\x2b\x00\xff...'
            # WebRTC only accepts raw audio data in bytes, so we need to convert it to the right format, not accepting AudioSegment directly, or .wav files directly.
            # List to store ONLY the active speech chunks
            #   
            # final_frames = []
            # analog_frames = []
            # buffer = np.array([], dtype=np.float32)
            # frame_size = self.dsp_processor.frame_size
            # total_chunks = len(noisy_input_original) // (FRAMES_PER_BUFFER * FORMAT_WIDTH)

            stage2_start = time.time()

            # for i in range(total_chunks):
            #     start_byte = i * (FRAMES_PER_BUFFER * FORMAT_WIDTH)
            #     end_byte = start_byte + (FRAMES_PER_BUFFER * FORMAT_WIDTH)
            #     chunk = noisy_input_original[start_byte:end_byte]
            #     noisy_signal_chunk = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
            #     buffer = np.concatenate([buffer, noisy_signal_chunk])

                # is_active = vad.is_speech(chunk, sample_rate=RATE)
                # if is_active:
                #     while len(buffer) >= frame_size:
                #         process_chunk = buffer[:frame_size]
                #         final_output_chunk = self.dsp_processor.process(
                #             noisy_signal=process_chunk,
                #             environment=environment,
                #             sample_rate=RATE
                #         )
                #         final_output_bytes = (
                #             np.clip(final_output_chunk, -32768, 32767)
                #             .astype(np.int16)
                #             .tobytes()
                #         )
                #         final_frames.append(final_output_bytes)

                #         from pipeline.stages.reconstruction import ReconstructionFilter
                #         reconstructor = ReconstructionFilter()
                #         analog_output_chunk = reconstructor.process(final_output_chunk, environment=environment, sample_rate=RATE)
                #         analog_output_bytes = (
                #             np.clip(analog_output_chunk, -32768, 32767)
                #             .astype(np.int16)
                #             .tobytes()
                #         )
                #         analog_frames.append(analog_output_bytes)
                #         buffer = buffer[frame_size:]  # bỏ phần đã xử lý
                # else:
                #     final_frames.append(chunk)
                #     analog_frames.append(chunk)

            # Xử lý phần còn lại của buffer (nếu có)
            # if len(buffer) > 0:
            #     pad_length = frame_size - len(buffer)
            #     padded_chunk = np.pad(buffer, (0, pad_length), mode='constant')
            #     final_output_chunk = self.dsp_processor.process(
            #         noisy_signal=padded_chunk,
            #         environment=environment,
            #         sample_rate=RATE
            #     )
            #     final_output_bytes = (
            #         np.clip(final_output_chunk, -32768, 32767)
            #         .astype(np.int16)
            #         .tobytes()
            #     )
            #     final_frames.append(final_output_bytes)

            #     from pipeline.stages.reconstruction import ReconstructionFilter
            #     reconstructor = ReconstructionFilter()
            #     analog_output_chunk = reconstructor.process(final_output_chunk, environment=environment, sample_rate=RATE)
            #     analog_output_bytes = (
            #         np.clip(analog_output_chunk, -32768, 32767)
            #         .astype(np.int16)
            #         .tobytes()
            #     )
            #     analog_frames.append(analog_output_bytes)
            # final_audio_data = b''.join(final_frames)
            # final_output = AudioSegment(
            #     data=final_audio_data,
            #     sample_width=FORMAT_WIDTH,
            #     frame_rate=RATE,
            #     channels=1
            # )

            # analog_audio_data = b''.join(analog_frames)  # Use analog_frames, not final_frames!
            # analog_output = AudioSegment(
            #     data=analog_audio_data,
            #     sample_width=FORMAT_WIDTH,
            #     frame_rate=RATE,
            #     channels=1
            # )

            final_output = self.dsp_processor.process(
                noisy_signal=noisy_input,
                environment=environment,
                sample_rate=RATE
            )
            from pipeline.stages.reconstruction import ReconstructionFilter
            reconstructor = ReconstructionFilter()
            analog_output = reconstructor.process(final_output, environment=environment, sample_rate=RATE)
            
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
                'analog_output': analog_output,   # Thêm tín hiệu analog (DAC) để app.py lưu file

                # Metadata
                'environment': environment,
                'sample_rate': RATE,

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
