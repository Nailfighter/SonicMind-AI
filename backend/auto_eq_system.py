#!/usr/bin/env python3
"""
ðŸŽ›ï¸ SonicMind AI - Auto-EQ System
Real-time audio processing with AI-enhanced EQ adjustments
"""

import numpy as np
import sounddevice as sd
import threading
import time
import math
from dataclasses import dataclass
from typing import List, Dict, Optional, Callable

# Optional dependencies
try:
    from scipy import signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ==================== AUDIO UTILITIES ==================== #


def db_to_lin(db):
    """Convert decibels to linear scale"""
    return 10 ** (db / 20.0)


def lin_to_db(x, eps=1e-12):
    """Convert linear to decibels"""
    return 20 * np.log10(np.maximum(np.abs(x), eps))


@dataclass
class EQBand:
    """EQ Band parameters"""
    freq: float     # Center frequency in Hz
    q: float        # Q factor (bandwidth)
    gain_db: float  # Gain in dB

# ==================== DIGITAL FILTERS ==================== #


class ParametricEQ:
    """5-band parametric EQ using biquad filters"""

    def __init__(self, sample_rate: int, bands: List[EQBand]):
        self.sample_rate = sample_rate
        self.bands = bands
        self.filters = []
        self._design_filters()

    def _design_filters(self):
        """Design biquad filters for each band"""
        self.filters = []

        for band in self.bands:
            if SCIPY_AVAILABLE:
                # Use SciPy for proper biquad design
                try:
                    # RBJ (Robert Bristow-Johnson) peaking EQ
                    A = 10 ** (band.gain_db / 40)
                    w0 = 2 * math.pi * band.freq / self.sample_rate
                    alpha = math.sin(w0) / (2 * max(band.q, 0.1))
                    cos_w0 = math.cos(w0)

                    # Biquad coefficients
                    b0 = 1 + alpha * A
                    b1 = -2 * cos_w0
                    b2 = 1 - alpha * A
                    a0 = 1 + alpha / A
                    a1 = -2 * cos_w0
                    a2 = 1 - alpha / A

                    # Normalize
                    b = np.array([b0, b1, b2]) / a0
                    a = np.array([1.0, a1 / a0, a2 / a0])

                    self.filters.append((b, a))

                except Exception as e:
                    print(
                        f"âš ï¸ Filter design failed for {band.freq}Hz: {e}")
                    self.filters.append(None)
            else:
                # Fallback - no filtering
                self.filters.append(None)

    def update_bands(self, bands: List[EQBand]):
        """Update EQ band parameters"""
        self.bands = bands
        self._design_filters()

    def process(self, audio: np.ndarray) -> np.ndarray:
        """Process audio through EQ filters"""
        if len(audio) == 0:
            return audio

        output = audio.astype(np.float32)

        if SCIPY_AVAILABLE:
            for filter_coef in self.filters:
                if filter_coef is not None:
                    b, a = filter_coef
                    try:
                        output = signal.lfilter(
                            b, a, output).astype(np.float32)
                    except Exception:
                        continue  # Skip problematic filters

        return output


class Compressor:
    """Simple digital compressor"""

    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
        self.threshold_db = -18.0
        self.ratio = 4.0
        self.attack_ms = 10.0
        self.release_ms = 100.0
        self.makeup_db = 6.0

        # Internal state
        self.envelope = 0.0
        self._attack_coef = math.exp(-1.0 /
                                     (self.attack_ms * 0.001 * sample_rate))
        self._release_coef = math.exp(-1.0 /
                                      (self.release_ms * 0.001 * sample_rate))

    def process(self, audio: np.ndarray) -> np.ndarray:
        """Apply compression to audio"""
        if len(audio) == 0:
            return audio

        output = np.zeros_like(audio)
        threshold_lin = db_to_lin(self.threshold_db)
        makeup_lin = db_to_lin(self.makeup_db)

        for i, sample in enumerate(audio):
            # Envelope follower
            abs_sample = abs(sample)
            if abs_sample > self.envelope:
                self.envelope = self._attack_coef * self.envelope + \
                    (1 - self._attack_coef) * abs_sample
            else:
                self.envelope = self._release_coef * self.envelope + \
                    (1 - self._release_coef) * abs_sample

            # Compression calculation
            if self.envelope > threshold_lin:
                over_db = lin_to_db(self.envelope) - self.threshold_db
                compressed_db = over_db / self.ratio
                gain_db = compressed_db - over_db
                gain_lin = db_to_lin(gain_db)
            else:
                gain_lin = 1.0

            output[i] = sample * gain_lin * makeup_lin

        return output

# ==================== AUDIO ANALYSIS ==================== #


class AudioAnalyzer:
    """Real-time audio feature extraction"""

    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate

    def analyze(self, audio: np.ndarray) -> Dict:
        """Extract audio features for EQ decisions"""
        if len(audio) == 0:
            return self._empty_analysis()

        # Basic level measurements
        rms = np.sqrt(np.mean(audio ** 2))
        peak = np.max(np.abs(audio))

        # Frequency analysis
        fft = np.fft.rfft(audio)
        magnitude = np.abs(fft)
        freqs = np.fft.rfftfreq(len(audio), 1.0 / self.sample_rate)

        # Frequency band energy
        bass_energy = self._band_energy(magnitude, freqs, 20, 250)
        mid_energy = self._band_energy(magnitude, freqs, 250, 4000)
        treble_energy = self._band_energy(magnitude, freqs, 4000, 20000)
        total_energy = bass_energy + mid_energy + treble_energy + 1e-12

        # Spectral features
        spectral_centroid = np.sum(
            freqs * magnitude) / (np.sum(magnitude) + 1e-12)
        spectral_rolloff = self._spectral_rolloff(magnitude, freqs, 0.85)

        return {
            'rms': float(rms),
            'peak': float(peak),
            'crest_factor': float(peak / (rms + 1e-12)),
            'bass_ratio': float(bass_energy / total_energy),
            'mid_ratio': float(mid_energy / total_energy),
            'treble_ratio': float(treble_energy / total_energy),
            'spectral_centroid': float(spectral_centroid),
            'spectral_rolloff': float(spectral_rolloff),
            'total_energy': float(total_energy)
        }

    def _band_energy(self, magnitude, freqs, low_freq, high_freq):
        """Calculate energy in frequency band"""
        mask = (freqs >= low_freq) & (freqs <= high_freq)
        return np.sum(magnitude[mask])

    def _spectral_rolloff(self, magnitude, freqs, threshold=0.85):
        """Calculate spectral rolloff frequency"""
        cumsum = np.cumsum(magnitude)
        total = cumsum[-1]
        rolloff_idx = np.where(cumsum >= threshold * total)[0]
        if len(rolloff_idx) > 0:
            return freqs[rolloff_idx[0]]
        return freqs[-1]

    def _empty_analysis(self):
        """Return empty analysis for silent audio"""
        return {
            'rms': 0.0, 'peak': 0.0, 'crest_factor': 1.0,
            'bass_ratio': 0.33, 'mid_ratio': 0.34, 'treble_ratio': 0.33,
            'spectral_centroid': 1000.0, 'spectral_rolloff': 8000.0,
            'total_energy': 0.0
        }

# ==================== RULE-BASED EQ ==================== #


class RuleBasedEQ:
    """Intelligent rule-based EQ adjustments"""

    def __init__(self):
        # Target frequency balance (ratios should sum to ~1.0)
        self.target_balance = {
            'bass_ratio': 0.25,
            'mid_ratio': 0.55,
            'treble_ratio': 0.20
        }

        # Instrument-specific EQ curves (dB adjustments per band)
        self.instrument_curves = {
            'acoustic_guitar': [0.0, 0.5, 1.0, 0.8, 0.3],      # Warm, present
            # Cut lows, boost mids
            'electric_guitar': [-0.5, 0.3, 1.5, 1.0, 0.0],
            # Boost lows, cut highs
            'bass_guitar': [2.0, 1.0, -0.5, -1.0, -1.5],
            # Balanced, bright
            'piano': [0.0, 0.3, 0.5, 0.8, 1.0],
            # Punch and sparkle
            'drums': [1.0, 0.5, 0.0, 1.5, 2.0],
            # Cut lows, boost highs
            'violin': [-1.0, 0.0, 0.5, 1.2, 1.5],
            'saxophone': [0.0, 0.8, 1.2, 0.5, 0.0],            # Mid-focused
            'vocal': [-0.5, 0.5, 1.0, 0.8, 0.0],               # Presence boost
            'none': [0.0, 0.0, 0.0, 0.0, 0.0]                  # No adjustment
        }

        self.max_adjustment_db = 0.3  # Maximum single adjustment

    def suggest_adjustments(self, analysis: Dict, current_instrument: str = 'none') -> List[float]:
        """Suggest EQ adjustments based on analysis and instrument"""
        adjustments = [0.0] * 5  # 5 EQ bands

        # Get current frequency balance
        current_balance = {
            'bass_ratio': analysis.get('bass_ratio', 0.33),
            'mid_ratio': analysis.get('mid_ratio', 0.33),
            'treble_ratio': analysis.get('treble_ratio', 0.33)
        }

        # Calculate balance corrections
        bass_error = self.target_balance['bass_ratio'] - \
            current_balance['bass_ratio']
        mid_error = self.target_balance['mid_ratio'] - \
            current_balance['mid_ratio']
        treble_error = self.target_balance['treble_ratio'] - \
            current_balance['treble_ratio']

        # Map errors to EQ bands (80Hz, 300Hz, 1kHz, 4kHz, 10kHz)
        adjustments[0] += bass_error * 3.0    # 80 Hz - deep bass
        adjustments[1] += bass_error * 2.0    # 300 Hz - bass/low-mid
        adjustments[2] += mid_error * 2.5     # 1 kHz - critical midrange
        adjustments[3] += treble_error * 2.0  # 4 kHz - presence
        adjustments[4] += treble_error * 3.0  # 10 kHz - air/sparkle

        # Add instrument-specific adjustments
        if current_instrument in self.instrument_curves:
            instrument_adj = self.instrument_curves[current_instrument]
            for i in range(len(adjustments)):
                adjustments[i] += instrument_adj[i] * \
                    0.3  # Scale down instrument influence

        # Limit adjustments
        adjustments = [np.clip(adj, -self.max_adjustment_db, self.max_adjustment_db)
                       for adj in adjustments]

        return adjustments

# ==================== MAIN AUTO-EQ SYSTEM ==================== #


class AutoEQSystem:
    """Main Auto-EQ system coordinator"""

    def __init__(self):
        self.sample_rate = 44100
        self.block_size = 256

        # EQ bands (80Hz, 300Hz, 1kHz, 4kHz, 10kHz)
        self.bands = [
            EQBand(80, 1.0, 0.0),
            EQBand(300, 1.2, 0.0),
            EQBand(1000, 1.5, 0.0),
            EQBand(4000, 2.0, 0.0),
            EQBand(10000, 1.0, 0.0)
        ]

        # Audio processing components
        self.eq = ParametricEQ(self.sample_rate, self.bands)
        self.compressor = Compressor(self.sample_rate)
        self.analyzer = AudioAnalyzer(self.sample_rate)
        self.rule_eq = RuleBasedEQ()

        # Audio streaming
        self.stream = None
        self.running = False

        # Auto-EQ control
        self.auto_eq_enabled = False
        self.auto_eq_thread = None
        self.current_instrument = 'none'

        # Callbacks
        self.update_callback = None

        # Audio buffer for analysis (thread-safe)
        self.audio_buffer = np.zeros(
            self.sample_rate * 2, dtype=np.float32)  # 2 seconds
        self.buffer_lock = threading.Lock()
        self.buffer_index = 0

    def start_audio(self, input_device=None, output_device=None) -> bool:
        """Start real-time audio processing"""
        if self.running:
            return False

        try:
            self.stream = sd.Stream(
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                channels=1,
                dtype='float32',
                callback=self._audio_callback,
                device=(input_device, output_device)
            )
            self.stream.start()
            self.running = True
            print("âœ… Audio engine started")
            return True

        except Exception as e:
            print(f"âŒ Audio start failed: {e}")
            return False

    def stop_audio(self):
        """Stop audio processing"""
        self.running = False
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            finally:
                self.stream = None
        print("ðŸ›‘ Audio engine stopped")

    def start_auto_eq(self) -> bool:
        """Start automatic EQ adjustments"""
        if self.auto_eq_enabled or not self.running:
            return False

        self.auto_eq_enabled = True
        self.auto_eq_thread = threading.Thread(
            target=self._auto_eq_loop, daemon=True)
        self.auto_eq_thread.start()
        print("âœ… Auto-EQ started")
        return True

    def stop_auto_eq(self):
        """Stop automatic EQ"""
        self.auto_eq_enabled = False
        print("ðŸ›‘ Auto-EQ stopped")

    def set_current_instrument(self, instrument: str):
        """Set the currently detected instrument"""
        self.current_instrument = instrument

    def apply_room_preset(self, room_analysis: Dict):
        """Apply EQ preset based on room acoustics"""
        acoustic_props = room_analysis.get('acoustic_properties', {})
        eq_adjustments = acoustic_props.get('eq_adjustments', {})

        # Map room adjustments to bands
        band_mapping = {'low': 0, 'low_mid': 1,
                        'mid': 2, 'high_mid': 3, 'high': 4}

        for freq_range, gain_db in eq_adjustments.items():
            if freq_range in band_mapping:
                band_idx = band_mapping[freq_range]
                if 0 <= band_idx < len(self.bands):
                    self.bands[band_idx].gain_db = float(
                        np.clip(gain_db, -6.0, 6.0))

        self.eq.update_bands(self.bands)
        print(
            f"ðŸ  Applied room preset: {acoustic_props.get('preset_name', 'unknown')}")

    def update_band(self, band_index: int, parameter: str, value: float) -> bool:
        """Update individual EQ band parameter"""
        if not (0 <= band_index < len(self.bands)):
            return False

        try:
            if parameter == 'freq':
                self.bands[band_index].freq = float(np.clip(value, 20, 20000))
            elif parameter == 'q':
                self.bands[band_index].q = float(np.clip(value, 0.1, 10.0))
            elif parameter == 'gain_db':
                self.bands[band_index].gain_db = float(
                    np.clip(value, -12.0, 12.0))
            else:
                return False

            self.eq.update_bands(self.bands)
            return True

        except Exception:
            return False

    def reset_eq(self):
        """Reset all EQ bands to flat response"""
        for band in self.bands:
            band.gain_db = 0.0
        self.eq.update_bands(self.bands)

    def get_bands_dict(self) -> List[Dict]:
        """Get EQ bands as dictionary list"""
        return [
            {
                'freq': band.freq,
                'q': band.q,
                'gain': band.gain_db
            }
            for band in self.bands
        ]

    def set_callback(self, callback: Callable):
        """Set callback for EQ updates"""
        self.update_callback = callback

    def get_available_devices(self) -> Dict:
        """Get available audio devices"""
        try:
            devices = sd.query_devices()
            input_devices = []
            output_devices = []

            for i, device in enumerate(devices):
                device_info = {
                    "index": i,
                    "name": device["name"],
                    "channels": device.get("max_input_channels", 0),
                    "hostapi": device.get("hostapi", 0)
                }

                if device.get("max_input_channels", 0) > 0:
                    input_devices.append(device_info)
                if device.get("max_output_channels", 0) > 0:
                    output_devices.append(device_info)

            return {
                "input_devices": input_devices,
                "output_devices": output_devices
            }

        except Exception as e:
            print(f"âš ï¸ Device query failed: {e}")
            return {
                "input_devices": [{"index": 0, "name": "Default Input", "channels": 1}],
                "output_devices": [{"index": 0, "name": "Default Output", "channels": 2}]
            }

    def cleanup(self):
        """Cleanup resources"""
        self.stop_auto_eq()
        self.stop_audio()

    def _audio_callback(self, indata, outdata, frames, time, status):
        """Real-time audio processing callback"""
        if status:
            print(f"âš ï¸ Audio callback status: {status}")

        # Get input audio (mono)
        input_audio = indata[:, 0].copy()

        # Store in analysis buffer
        with self.buffer_lock:
            buffer_len = len(self.audio_buffer)
            for sample in input_audio:
                self.audio_buffer[self.buffer_index] = sample
                self.buffer_index = (self.buffer_index + 1) % buffer_len

        # Process audio through EQ and compressor
        processed = self.eq.process(input_audio)
        processed = self.compressor.process(processed)

        # Output (limit to prevent clipping)
        output = np.clip(processed * 0.8, -1.0, 1.0)
        outdata[:, 0] = output

    def _auto_eq_loop(self):
        """Auto-EQ adjustment loop"""
        print("ðŸ¤– Auto-EQ loop started")

        # Initial warmup
        time.sleep(3.0)

        while self.auto_eq_enabled:
            try:
                # Get recent audio for analysis
                with self.buffer_lock:
                    # Get last 2 seconds of audio
                    analysis_audio = self.audio_buffer.copy()

                # Analyze audio
                analysis = self.analyzer.analyze(analysis_audio)

                # Get EQ suggestions
                adjustments = self.rule_eq.suggest_adjustments(
                    analysis, self.current_instrument)

                # Apply adjustments with smoothing
                for i, adjustment in enumerate(adjustments):
                    if abs(adjustment) > 0.05:  # Only apply significant adjustments
                        current_gain = self.bands[i].gain_db
                        new_gain = current_gain + adjustment * 0.5  # Smooth application
                        self.bands[i].gain_db = float(
                            np.clip(new_gain, -8.0, 8.0))

                # Update filters
                self.eq.update_bands(self.bands)

                # Notify callback
                if self.update_callback:
                    self.update_callback(
                        self.get_bands_dict(),
                        "auto_adjustment",
                        {
                            'analysis': analysis,
                            'adjustments': adjustments,
                            'instrument': self.current_instrument
                        }
                    )

                # Wait before next adjustment
                time.sleep(2.0)

            except Exception as e:
                print(f"âš ï¸ Auto-EQ error: {e}")
                time.sleep(1.0)
