"""
Stage 2: DSP Processor - IMPLEMENTED NOISE FILTERING
Actual noise reduction algorithms
"""

import numpy as np
import scipy.signal as signal
import logging
from utils.config import ENVIRONMENT_CONFIGS

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
        logger.info(f"DSP processing for {environment} environment - VOICE-CENTRIC FILTERING")
        
        # Get environment-specific parameters
        env_config = ENVIRONMENT_CONFIGS.get(environment, ENVIRONMENT_CONFIGS['office'])
        target_snr = env_config['target_snr_db']
        
        # STEP 1: Voice Activity Detection and Extraction
        voice_mask, clean_voice = self._extract_human_voice(noisy_signal, sample_rate)
        
        # STEP 2: Apply aggressive noise suppression based on voice detection
        if environment == 'office':
            # Office: Suppress keyboard, paper shuffling, background chatter
            enhanced = self._voice_centric_processing(noisy_signal, voice_mask, clean_voice, sample_rate, 
                                                    suppress_level=0.2, voice_boost=1.5)
        elif environment == 'cafe':
            # Cafe: Very aggressive suppression of background voices and noise
            enhanced = self._voice_centric_processing(noisy_signal, voice_mask, clean_voice, sample_rate, 
                                                    suppress_level=0.1, voice_boost=2.0)
        elif environment == 'street':
            # Street: Extreme suppression of traffic, horns, sirens
            enhanced = self._voice_centric_processing(noisy_signal, voice_mask, clean_voice, sample_rate, 
                                                    suppress_level=0.05, voice_boost=2.5)
        elif environment == 'home':
            # Home: Ultra-aggressive animal sound suppression with simple but effective approach
            # Start with basic voice processing
            enhanced = self._voice_centric_processing(noisy_signal, voice_mask, clean_voice, sample_rate, 
                                                    suppress_level=0.4, voice_boost=1.3)
            # Apply simple but very aggressive animal suppression
            enhanced = self._simple_aggressive_animal_suppression(enhanced, sample_rate)
            # Apply speech continuity enhancement
            enhanced = self._enhance_speech_continuity(enhanced, voice_mask, sample_rate)
        else:
            # Default: Moderate voice-centric processing
            enhanced = self._voice_centric_processing(noisy_signal, voice_mask, clean_voice, sample_rate, 
                                                    suppress_level=0.2, voice_boost=1.5)
        
        # Final normalization and clipping prevention
        enhanced = self._normalize_audio(enhanced)
        
        logger.info(f"DSP processing complete - Signal enhanced for {environment}")
        return enhanced
    
    def _apply_spectral_subtraction(self, signal_in: np.ndarray, alpha: float = 2.0) -> np.ndarray:
        """
        Apply spectral subtraction noise reduction
        """
        # Add small epsilon to prevent divide by zero
        epsilon = 1e-10
        signal_in = signal_in + epsilon
        
        # STFT
        f, t, stft = signal.stft(signal_in, nperseg=self.frame_size, 
                                noverlap=self.frame_size - self.hop_size)
        
        # Estimate noise from first few frames
        noise_frames = min(self.noise_estimation_frames, stft.shape[1] // 4)
        noise_spectrum = np.mean(np.abs(stft[:, :noise_frames]), axis=1, keepdims=True)
        
        # Spectral subtraction
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Over-subtraction with spectral floor
        enhanced_magnitude = magnitude - alpha * noise_spectrum
        enhanced_magnitude = np.maximum(enhanced_magnitude, self.beta * magnitude)
        
        # Reconstruct signal
        enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
        _, enhanced = signal.istft(enhanced_stft, nperseg=self.frame_size,
                                  noverlap=self.frame_size - self.hop_size)
        
        return enhanced[:len(signal_in)]
    
    def _apply_wiener_filter(self, signal_in: np.ndarray) -> np.ndarray:
        """
        Apply Wiener filtering
        """
        # Simple Wiener filter approximation
        f, t, stft = signal.stft(signal_in, nperseg=self.frame_size,
                                noverlap=self.frame_size - self.hop_size)
        
        # Estimate noise and signal power
        noise_frames = min(self.noise_estimation_frames, stft.shape[1] // 4)
        noise_power = np.mean(np.abs(stft[:, :noise_frames])**2, axis=1, keepdims=True)
        signal_power = np.mean(np.abs(stft)**2, axis=1, keepdims=True)
        
        # Wiener gain
        wiener_gain = signal_power / (signal_power + noise_power + 1e-10)
        
        # Apply gain
        enhanced_stft = stft * wiener_gain
        _, enhanced = signal.istft(enhanced_stft, nperseg=self.frame_size,
                                  noverlap=self.frame_size - self.hop_size)
        
        return enhanced[:len(signal_in)]
    
    def _apply_bandpass_filter(self, signal_in: np.ndarray, sample_rate: int, 
                              low_freq: float, high_freq: float) -> np.ndarray:
        """
        Apply bandpass filter
        """
        nyquist = sample_rate / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        if high >= 1.0:
            high = 0.99
        if low <= 0:
            low = 0.01
            
        sos = signal.butter(4, [low, high], btype='band', output='sos')
        return signal.sosfilt(sos, signal_in)
    
    def _apply_highpass_filter(self, signal_in: np.ndarray, sample_rate: int, 
                              cutoff_freq: float) -> np.ndarray:
        """
        Apply highpass filter to remove low-frequency noise
        """
        nyquist = sample_rate / 2
        cutoff = cutoff_freq / nyquist
        
        if cutoff >= 1.0:
            cutoff = 0.99
            
        sos = signal.butter(4, cutoff, btype='high', output='sos')
        return signal.sosfilt(sos, signal_in)
    
    def _apply_smoothing_filter(self, signal_in: np.ndarray) -> np.ndarray:
        """
        Apply gentle smoothing filter
        """
        # Simple moving average smoothing
        window_size = 5
        kernel = np.ones(window_size) / window_size
        smoothed = np.convolve(signal_in, kernel, mode='same')
        
        # Blend with original (light processing)
        return 0.7 * signal_in + 0.3 * smoothed
    
    def _normalize_audio(self, signal_in: np.ndarray) -> np.ndarray:
        """
        Normalize audio to prevent clipping and maintain reasonable levels
        """
        # Remove DC offset
        signal_out = signal_in - np.mean(signal_in)
        
        # Gentle normalization to 70% of max range
        max_val = np.max(np.abs(signal_out))
        if max_val > 0:
            signal_out = signal_out * (0.7 / max_val)
        
        # Clip to prevent overflow
        signal_out = np.clip(signal_out, -1.0, 1.0)
        
        return signal_out
    
    def _apply_animal_sound_filter(self, signal_in: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Specifically target animal sounds like dog barking, cat meowing
        """
        # Animal sounds typically have these characteristics:
        # - Dog barks: 500-2000 Hz, sudden onset, short duration (0.1-0.5s)
        # - Cat meows: 300-2000 Hz, longer duration (0.5-2s)
        # - Both have harmonic structure and high energy
        
        f, t, stft = signal.stft(signal_in, nperseg=self.frame_size,
                                noverlap=self.frame_size - self.hop_size)
        
        # Frequency ranges for animal sounds (in Hz)
        animal_freq_low = 300   # Lower bound
        animal_freq_high = 2500 # Upper bound
        
        # Convert to frequency bin indices
        freq_bins = np.fft.fftfreq(self.frame_size, 1/sample_rate)
        freq_bins = freq_bins[:len(f)]
        
        animal_mask = (freq_bins >= animal_freq_low) & (freq_bins <= animal_freq_high)
        
        # Detect transient events (sudden energy increases)
        magnitude = np.abs(stft)
        energy_profile = np.sum(magnitude[animal_mask, :], axis=0)
        
        # Find sudden energy spikes (potential animal sounds)
        energy_diff = np.diff(energy_profile)
        energy_threshold = np.percentile(energy_diff, 85)  # Top 15% of energy changes
        
        # Create suppression mask
        enhanced_stft = stft.copy()
        
        for t_idx in range(stft.shape[1]):
            if t_idx > 0 and energy_diff[t_idx-1] > energy_threshold:
                # Potential animal sound - apply selective suppression
                frame_magnitude = magnitude[:, t_idx]
                frame_phase = np.angle(stft[:, t_idx])
                
                # Suppress animal frequency range more aggressively
                suppression_factor = np.ones_like(frame_magnitude)
                suppression_factor[animal_mask] = 0.3  # Reduce by 70%
                
                # Apply suppression
                suppressed_magnitude = frame_magnitude * suppression_factor
                enhanced_stft[:, t_idx] = suppressed_magnitude * np.exp(1j * frame_phase)
        
        # Reconstruct signal
        _, enhanced = signal.istft(enhanced_stft, nperseg=self.frame_size,
                                  noverlap=self.frame_size - self.hop_size)
        
        return enhanced[:len(signal_in)]
    
    def _apply_impulse_noise_filter(self, signal_in: np.ndarray) -> np.ndarray:
        """
        Remove sudden impulse noises (door slams, clicks, etc.)
        """
        # Detect sudden amplitude changes
        diff_signal = np.diff(signal_in)
        threshold = np.percentile(np.abs(diff_signal), 95)  # Top 5% of changes
        
        # Find impulse locations
        impulse_locations = np.where(np.abs(diff_signal) > threshold)[0]
        
        enhanced = signal_in.copy()
        
        # Suppress impulses by interpolation
        for loc in impulse_locations:
            # Define suppression window
            start = max(0, loc - 10)
            end = min(len(enhanced), loc + 10)
            
            # Linear interpolation to smooth the impulse
            if start > 0 and end < len(enhanced):
                enhanced[start:end] = np.linspace(enhanced[start-1], enhanced[end], end-start)
        
        return enhanced
    
    def _apply_transient_suppression(self, signal_in: np.ndarray) -> np.ndarray:
        """
        Suppress transient sounds (car horns, sirens, etc.)
        """
        # Use spectral flux to detect transients
        f, t, stft = signal.stft(signal_in, nperseg=self.frame_size,
                                noverlap=self.frame_size - self.hop_size)
        
        magnitude = np.abs(stft)
        
        # Calculate spectral flux (measure of spectral change)
        spectral_flux = np.sum(np.diff(magnitude, axis=1)**2, axis=0)
        flux_threshold = np.percentile(spectral_flux, 80)  # Top 20% of changes
        
        # Apply dynamic gain reduction to transient frames
        enhanced_stft = stft.copy()
        for t_idx in range(1, stft.shape[1]):
            if spectral_flux[t_idx-1] > flux_threshold:
                # Reduce gain for transient frame
                enhanced_stft[:, t_idx] *= 0.5
        
        # Reconstruct
        _, enhanced = signal.istft(enhanced_stft, nperseg=self.frame_size,
                                  noverlap=self.frame_size - self.hop_size)
        
        return enhanced[:len(signal_in)]
    
    def _apply_voice_isolation(self, signal_in: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Enhance human voice frequencies and suppress other sounds
        """
        # Human voice fundamental frequency range: 80-500 Hz
        # Human voice harmonic range: 80-4000 Hz
        
        # Apply bandpass filter for voice frequencies
        enhanced = self._apply_bandpass_filter(signal_in, sample_rate, 80, 4000)
        
        # Additional formant enhancement (typical formants: 800Hz, 1200Hz, 2500Hz)
        formant_freqs = [800, 1200, 2500]
        
        for formant in formant_freqs:
            # Create narrow bandpass around formant
            low_freq = formant - 100
            high_freq = formant + 100
            formant_signal = self._apply_bandpass_filter(signal_in, sample_rate, low_freq, high_freq)
            # Boost formant regions slightly
            enhanced += 0.2 * formant_signal
        
        # Normalize to prevent clipping
        enhanced = self._normalize_audio(enhanced)
        
        return enhanced
    
    def _extract_human_voice(self, signal_in: np.ndarray, sample_rate: int) -> tuple:
        """
        Advanced voice activity detection and voice extraction
        Returns: (voice_mask, extracted_voice_signal)
        """
        # STFT for frequency analysis
        f, t, stft = signal.stft(signal_in, nperseg=self.frame_size,
                                noverlap=self.frame_size - self.hop_size)
        
        magnitude = np.abs(stft)
        freq_bins = np.fft.fftfreq(self.frame_size, 1/sample_rate)[:len(f)]
        
        # Define human voice characteristics
        voice_freq_range = (80, 4000)   # Fundamental + harmonics
        formant_ranges = [(200, 1000), (800, 1800), (1500, 3500)]  # F1, F2, F3
        
        # Calculate voice likelihood for each time frame
        voice_scores = np.zeros(stft.shape[1])
        
        for t_idx in range(stft.shape[1]):
            frame_mag = magnitude[:, t_idx]
            
            # Score 1: Energy in voice frequency range
            voice_mask = (freq_bins >= voice_freq_range[0]) & (freq_bins <= voice_freq_range[1])
            voice_energy = np.sum(frame_mag[voice_mask])
            total_energy = np.sum(frame_mag) + 1e-10
            energy_ratio = voice_energy / total_energy
            
            # Score 2: Formant presence
            formant_score = 0
            for f_low, f_high in formant_ranges:
                formant_mask = (freq_bins >= f_low) & (freq_bins <= f_high)
                formant_energy = np.sum(frame_mag[formant_mask])
                if formant_energy > 0:
                    formant_score += 1
            formant_score /= len(formant_ranges)
            
            # Score 3: Harmonic structure detection
            harmonic_score = self._detect_harmonic_structure(frame_mag, freq_bins)
            
            # Score 4: Temporal consistency
            temporal_score = 1.0
            if t_idx > 0:
                prev_voice_energy = np.sum(magnitude[voice_mask, t_idx-1])
                consistency = min(voice_energy, prev_voice_energy) / (max(voice_energy, prev_voice_energy) + 1e-10)
                temporal_score = consistency
            
            # Combine scores (weighted)
            voice_scores[t_idx] = (0.3 * energy_ratio + 0.3 * formant_score + 
                                 0.2 * harmonic_score + 0.2 * temporal_score)
        
        # Create voice activity mask (threshold-based with smoothing)
        voice_threshold = np.percentile(voice_scores, 50)  # Lower threshold (top 50% as potential voice)
        initial_voice_mask = voice_scores > voice_threshold
        
        # Apply temporal smoothing to reduce fragmentation
        voice_mask = self._smooth_voice_mask(initial_voice_mask, min_voice_duration=5)
        
        # Extract clean voice using advanced spectral masking
        clean_voice_stft = stft.copy()
        
        for t_idx in range(stft.shape[1]):
            if voice_mask[t_idx]:
                # Enhance voice frequencies
                voice_freq_mask = (freq_bins >= 80) & (freq_bins <= 4000)
                clean_voice_stft[~voice_freq_mask, t_idx] *= 0.1  # Suppress non-voice frequencies
            else:
                # Non-voice frame - suppress heavily
                clean_voice_stft[:, t_idx] *= 0.05
        
        # Reconstruct clean voice signal
        _, clean_voice = signal.istft(clean_voice_stft, nperseg=self.frame_size,
                                     noverlap=self.frame_size - self.hop_size)
        
        return voice_mask, clean_voice[:len(signal_in)]
    
    def _detect_harmonic_structure(self, magnitude: np.ndarray, freq_bins: np.ndarray) -> float:
        """
        Detect harmonic structure typical of human voice
        """
        # Look for fundamental frequency and harmonics
        fundamental_range = (80, 400)  # Typical F0 range for human voice
        
        # Find peaks in fundamental range
        fund_mask = (freq_bins >= fundamental_range[0]) & (freq_bins <= fundamental_range[1])
        fund_spectrum = magnitude[fund_mask]
        
        if len(fund_spectrum) == 0:
            return 0.0
        
        # Find the strongest peak as potential fundamental
        fund_peak_idx = np.argmax(fund_spectrum)
        fund_freq = freq_bins[fund_mask][fund_peak_idx]
        
        # Look for harmonics (2f0, 3f0, 4f0, etc.)
        harmonic_score = 0
        harmonics_found = 0
        
        for harmonic_num in range(2, 6):  # Check 2nd to 5th harmonics
            harmonic_freq = fund_freq * harmonic_num
            if harmonic_freq > freq_bins[-1]:
                break
                
            # Find energy near harmonic frequency (±20 Hz tolerance)
            harmonic_mask = (freq_bins >= harmonic_freq - 20) & (freq_bins <= harmonic_freq + 20)
            if np.any(harmonic_mask):
                harmonic_energy = np.max(magnitude[harmonic_mask])
                fundamental_energy = magnitude[fund_mask][fund_peak_idx]
                
                # Good harmonic should have significant energy relative to fundamental
                if harmonic_energy > 0.1 * fundamental_energy:
                    harmonics_found += 1
                    harmonic_score += harmonic_energy / fundamental_energy
        
        # Normalize score
        if harmonics_found > 0:
            return min(1.0, harmonic_score / harmonics_found)
        return 0.0
    
    def _voice_centric_processing(self, signal_in: np.ndarray, voice_mask: np.ndarray, 
                                clean_voice: np.ndarray, sample_rate: int,
                                suppress_level: float = 0.2, voice_boost: float = 1.5) -> np.ndarray:
        """
        Voice-centric processing: Aggressively suppress non-voice, enhance voice
        """
        # STFT for processing
        f, t, stft = signal.stft(signal_in, nperseg=self.frame_size,
                                noverlap=self.frame_size - self.hop_size)
        
        # Apply time-frequency masking based on voice activity (with continuity)
        enhanced_stft = stft.copy()
        
        for t_idx in range(stft.shape[1]):
            if t_idx < len(voice_mask) and voice_mask[t_idx]:
                # Voice frame: gentle enhancement to preserve speech quality
                freq_bins = np.fft.fftfreq(self.frame_size, 1/sample_rate)[:len(f)]
                voice_freq_mask = (freq_bins >= 80) & (freq_bins <= 4000)
                
                # Gentle voice boost
                enhanced_stft[voice_freq_mask, t_idx] *= voice_boost
                # Less aggressive non-voice suppression to maintain continuity
                enhanced_stft[~voice_freq_mask, t_idx] *= max(0.3, suppress_level)
            else:
                # Non-voice frame: moderate suppression (not too aggressive)
                enhanced_stft[:, t_idx] *= max(0.1, suppress_level)
        
        # Reconstruct signal
        _, enhanced = signal.istft(enhanced_stft, nperseg=self.frame_size,
                                  noverlap=self.frame_size - self.hop_size)
        
        # Blend with clean voice for better quality
        enhanced = enhanced[:len(signal_in)]
        clean_voice = clean_voice[:len(signal_in)]
        
        # Adaptive blending based on voice confidence
        voice_confidence = np.sum(voice_mask) / len(voice_mask) if len(voice_mask) > 0 else 0
        blend_ratio = min(0.7, voice_confidence * 1.2)  # More clean voice if high confidence
        
        final_enhanced = (1 - blend_ratio) * enhanced + blend_ratio * clean_voice * voice_boost
        
        return self._normalize_audio(final_enhanced)
    
    def _aggressive_animal_suppression(self, signal_in: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Enhanced animal suppression with better voice protection
        """
        f, t, stft = signal.stft(signal_in, nperseg=self.frame_size,
                                noverlap=self.frame_size - self.hop_size)
        
        magnitude = np.abs(stft)
        freq_bins = np.fft.fftfreq(self.frame_size, 1/sample_rate)[:len(f)]
        
        enhanced_stft = stft.copy()
        
        # Detect voice activity first to protect it
        voice_mask, _ = self._extract_human_voice(signal_in, sample_rate)
        
        # Process each frame
        for t_idx in range(stft.shape[1]):
            frame_mag = magnitude[:, t_idx]
            
            # Check if this frame has voice activity
            has_voice = t_idx < len(voice_mask) and voice_mask[t_idx]
            
            # Detect animal sounds in this frame
            is_animal_sound = self._detect_animal_sound_in_frame(frame_mag, freq_bins)
            
            # Additional simple amplitude-based detection for very loud animal sounds
            frame_energy = np.sum(frame_mag)
            if not is_animal_sound and frame_energy > 0:
                # Check if this frame is much louder than average (potential loud bark/meow)
                avg_energy = np.mean(np.sum(magnitude, axis=0))
                if frame_energy > 2.0 * avg_energy:  # Much louder than average
                    # Check frequency distribution - animals often have energy spread
                    non_voice_mask = (freq_bins < 80) | (freq_bins > 3500)
                    non_voice_energy = np.sum(frame_mag[non_voice_mask]) if np.any(non_voice_mask) else 0
                    if non_voice_energy > 0.3 * frame_energy:  # Significant non-voice energy
                        is_animal_sound = True
            
            if is_animal_sound:
                if has_voice:
                    # Frame has both voice and animal sounds - surgical suppression
                    # Only suppress specific animal frequency ranges, preserve voice ranges
                    suppression_mask = np.ones_like(frame_mag)
                    
                    # Protect core voice frequencies strongly
                    voice_core_bands = [
                        (100, 300),   # Very low voice fundamentals
                        (300, 600),   # Voice fundamentals  
                        (600, 1000),  # Low formants
                        (1000, 1400), # F1 range
                        (1400, 2200), # F2 range
                        (2200, 3000), # F3 range (partial)
                    ]
                    
                    for low_f, high_f in voice_core_bands:
                        protect_mask = (freq_bins >= low_f) & (freq_bins <= high_f)
                        suppression_mask[protect_mask] = 0.9  # Minimal suppression
                    
                    # Moderate suppression of potential animal ranges
                    animal_ranges = [
                        (3000, 5000),   # High animal harmonics
                        (5000, 8000),   # Very high frequencies
                    ]
                    
                    for low_f, high_f in animal_ranges:
                        animal_mask = (freq_bins >= low_f) & (freq_bins <= high_f)
                        suppression_mask[animal_mask] = 0.3  # Moderate suppression
                        
                else:
                    # Frame has animal sounds but no voice - very aggressive suppression
                    suppression_mask = np.ones_like(frame_mag)
                    
                    # Protect only core voice frequencies (smaller safety margin)
                    voice_safety_bands = [
                        (100, 400),    # Core fundamental range only
                        (800, 1200),   # F1 safety (smaller range)
                        (1200, 1600),  # F2 safety (smaller range)
                    ]
                    
                    for low_f, high_f in voice_safety_bands:
                        protect_mask = (freq_bins >= low_f) & (freq_bins <= high_f)
                        suppression_mask[protect_mask] = 0.4  # More aggressive than before
                    
                    # Very aggressive suppression elsewhere
                    animal_target_bands = [
                        (50, 100),      # Very low frequencies
                        (400, 800),     # Mid-low animal sounds
                        (1600, 3000),   # Mid-high animal sounds
                        (3000, 10000),  # High frequency animal content
                    ]
                    
                    for low_f, high_f in animal_target_bands:
                        animal_mask = (freq_bins >= low_f) & (freq_bins <= high_f)
                        suppression_mask[animal_mask] = 0.05  # Very strong suppression (95% reduction)
                
                # Apply the suppression mask
                enhanced_stft[:, t_idx] = stft[:, t_idx] * suppression_mask
        
        # Reconstruct
        _, enhanced = signal.istft(enhanced_stft, nperseg=self.frame_size,
                                  noverlap=self.frame_size - self.hop_size)
        
        return enhanced[:len(signal_in)]
    
    def _simple_aggressive_animal_suppression(self, signal_in: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Aggressive but voice-preserving animal sound suppression.
        Suppress loud, non-voice sounds while keeping the human voice natural.
        """
        f, t, stft = signal.stft(signal_in, nperseg=self.frame_size,
                                noverlap=self.frame_size - self.hop_size)
        magnitude = np.abs(stft)
        freq_bins = np.fft.fftfreq(self.frame_size, 1/sample_rate)[:len(f)]
        enhanced_stft = stft.copy()

        # Calculate overall energy profile
        frame_energies = np.sum(magnitude, axis=0)
        energy_threshold = np.percentile(frame_energies, 60)


        # Diagnostic: Only pass 300–1200 Hz, mute everything else for all frames
        voice_band = (freq_bins >= 300) & (freq_bins <= 1200)

        for t_idx in range(stft.shape[1]):
            suppression_mask = np.zeros_like(magnitude[:, t_idx])
            suppression_mask[voice_band] = 1.0
            enhanced_stft[:, t_idx] = stft[:, t_idx] * suppression_mask

        _, enhanced = signal.istft(enhanced_stft, nperseg=self.frame_size,
                                  noverlap=self.frame_size - self.hop_size)
        return enhanced[:len(signal_in)]
    
    def _detect_animal_sound_in_frame(self, frame_magnitude: np.ndarray, freq_bins: np.ndarray) -> bool:
        """
        Very aggressive animal sound detection
        """
        # Multiple frequency bands for different animal sounds
        animal_bands = [
            (200, 600),    # Low growls, deep barks
            (600, 1200),   # Mid-low barks, meows  
            (1200, 2500),  # High barks, cat calls
            (2500, 5000),  # Very high frequency components
            (5000, 8000),  # Ultra-high frequency animal sounds
        ]
        
        total_energy = np.sum(frame_magnitude) + 1e-10
        animal_indicators = []
        
        # Check energy in each animal band with much lower thresholds
        for low_f, high_f in animal_bands:
            band_mask = (freq_bins >= low_f) & (freq_bins <= high_f)
            if np.any(band_mask):
                band_energy = np.sum(frame_magnitude[band_mask])
                band_ratio = band_energy / total_energy
                animal_indicators.append(band_ratio)
            else:
                animal_indicators.append(0.0)
        
        # Much more sensitive detection criteria:
        # 1. Any significant energy in animal frequency bands
        high_energy_bands = sum(1 for ratio in animal_indicators if ratio > 0.08)  # Very low threshold
        
        # 2. Energy concentration in mid-high frequencies (typical of animals)
        mid_high_mask = (freq_bins >= 800) & (freq_bins <= 4000)
        if np.any(mid_high_mask):
            mid_high_energy = np.sum(frame_magnitude[mid_high_mask])
            mid_high_ratio = mid_high_energy / total_energy
        else:
            mid_high_ratio = 0.0
        
        # 3. Spectral irregularity (much lower threshold)
        spectral_variance = np.var(frame_magnitude)
        spectral_mean = np.mean(frame_magnitude) + 1e-10
        irregularity = spectral_variance / spectral_mean
        
        # 4. Peak concentration (lower threshold)
        sorted_magnitude = np.sort(frame_magnitude)[::-1]
        top_15_percent = int(len(sorted_magnitude) * 0.15)
        peak_energy = np.sum(sorted_magnitude[:top_15_percent])
        peak_concentration = peak_energy / total_energy
        
        # 5. High frequency emphasis (much lower threshold)
        high_freq_mask = freq_bins >= 1500  # Lower frequency threshold
        if np.any(high_freq_mask):
            high_freq_energy = np.sum(frame_magnitude[high_freq_mask])
            high_freq_ratio = high_freq_energy / total_energy
        else:
            high_freq_ratio = 0.0
        
        # 6. Energy above noise floor
        energy_threshold = np.percentile(frame_magnitude, 70)  # 70th percentile
        above_threshold_count = np.sum(frame_magnitude > energy_threshold)
        energy_spread = above_threshold_count / len(frame_magnitude)
        
        # Very aggressive detection (OR logic - any indicator can trigger)
        is_animal = (
            high_energy_bands >= 1 or                    # Energy in at least 1 animal band (very low threshold)
            mid_high_ratio > 0.25 or                     # 25% energy in mid-high frequencies
            irregularity > 20 or                         # Much lower irregularity threshold  
            peak_concentration > 0.3 or                  # Lower peak concentration
            high_freq_ratio > 0.15 or                    # Lower high-frequency threshold
            energy_spread > 0.15                         # Significant energy spread
        )
        
        return is_animal
    
    def _smooth_voice_mask(self, voice_mask: np.ndarray, min_voice_duration: int = 5) -> np.ndarray:
        """
        Smooth voice activity mask to reduce fragmentation
        """
        smoothed_mask = voice_mask.copy()
        
        # Fill short gaps between voice segments
        gap_threshold = 3  # Fill gaps of 3 frames or less
        voice_regions = []
        current_start = None
        
        # Find voice regions
        for i, is_voice in enumerate(voice_mask):
            if is_voice and current_start is None:
                current_start = i
            elif not is_voice and current_start is not None:
                voice_regions.append((current_start, i))
                current_start = None
        
        # Add final region if needed
        if current_start is not None:
            voice_regions.append((current_start, len(voice_mask)))
        
        # Fill short gaps between regions
        for i in range(len(voice_regions) - 1):
            _, end1 = voice_regions[i]
            start2, _ = voice_regions[i + 1]
            gap_size = start2 - end1
            
            if gap_size <= gap_threshold:
                smoothed_mask[end1:start2] = True
        
        # Remove very short voice segments
        for start, end in voice_regions:
            if end - start < min_voice_duration:
                smoothed_mask[start:end] = False
        
        return smoothed_mask
    
    def _enhance_speech_continuity(self, signal_in: np.ndarray, voice_mask: np.ndarray, 
                                 sample_rate: int) -> np.ndarray:
        """
        Apply post-processing to enhance speech continuity and reduce artifacts
        """
        # Apply gentle smoothing to reduce processing artifacts
        smoothed = self._apply_gentle_smoothing(signal_in)
        
        # Enhance formant regions specifically
        formant_enhanced = self._enhance_formants(smoothed, sample_rate)
        
        return formant_enhanced
    
    def _apply_gentle_smoothing(self, signal_in: np.ndarray) -> np.ndarray:
        """
        Apply very gentle smoothing to reduce artifacts while preserving speech
        """
        # Use a small smoothing kernel
        kernel_size = 3
        kernel = np.ones(kernel_size) / kernel_size
        
        # Apply smoothing
        smoothed = np.convolve(signal_in, kernel, mode='same')
        
        # Blend with original (very light smoothing)
        return 0.9 * signal_in + 0.1 * smoothed
    
    def _enhance_formants(self, signal_in: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Enhance speech formants to improve intelligibility
        """
        # Define formant frequency ranges for speech clarity
        formant_bands = [
            (200, 800),   # F1 range
            (800, 1800),  # F2 range  
            (1800, 3500)  # F3 range
        ]
        
        enhanced = signal_in.copy()
        
        for low_freq, high_freq in formant_bands:
            # Create bandpass filter for formant
            formant_signal = self._apply_bandpass_filter(signal_in, sample_rate, low_freq, high_freq)
            # Gentle enhancement
            enhanced += 0.1 * formant_signal
        
        return self._normalize_audio(enhanced)
    
