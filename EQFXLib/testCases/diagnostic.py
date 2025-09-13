#!/usr/bin/env python3
"""
SonicMind-AI EQFXLib - Complete Audio Test with All Features
==========================================================
✅ 10-second recordings
✅ Extreme EQ effects  
✅ Reverb and delay
✅ Distortion effects
✅ Compression
✅ SciPy and NumPy support
"""

import sounddevice as sd
import numpy as np
import platform
import sys
from typing import Dict, List, Tuple

# Check for SciPy
SCIPY_AVAILABLE = False
try:
    import scipy.signal
    import scipy.io.wavfile
    SCIPY_AVAILABLE = True
    print("✅ SciPy loaded successfully")
except ImportError:
    print("⚠️  SciPy not available - using NumPy fallback")

class AudioTester:
    def __init__(self):
        self.sample_rate = 44100
        self.block_size = 512
        self.test_duration = 10.0  # 10 SECONDS
        self.pause_duration = 1000  # 1 second pause between effects
        
    def get_system_info(self) -> Dict[str, str]:
        """Get system information"""
        info = {
            'platform': platform.platform(),
            'machine': platform.machine(),
            'python_version': sys.version.split()[0],
            'sounddevice_version': sd.__version__,
            'numpy': "✅ Available",
            'scipy': "✅ Available" if SCIPY_AVAILABLE else "📦 pip install scipy",
            'pedalboard': "❌ DISABLED (avoiding issues)"
        }
        return info
    
    def create_effects(self) -> List[Dict]:
        """Create EXTREME effects with compression, reverb and delay"""
        effects = []
        
        if SCIPY_AVAILABLE:
            print("🔧 Creating SciPy effects with COMPRESSION, REVERB, DELAY, and EXTREME EQ...")
            
            effects = [
                {"name": "Original", "description": "Clean audio", "type": "bypass"},
                {"name": "🔊 MEGA Bass", "description": "INSANE low frequencies", "type": "scipy", "filter": "bass"},
                {"name": "✨ MEGA Treble", "description": "INSANE high frequencies", "type": "scipy", "filter": "treble"}, 
                {"name": "🎚️ Heavy Compressor", "description": "Dynamic range control", "type": "scipy", "filter": "compression"},
                {"name": "😊 MEGA Scoop", "description": "EXTREME V-shaped EQ", "type": "scipy", "filter": "mid_scoop"},
                {"name": "🌊 Cathedral Reverb", "description": "Massive spatial reverb", "type": "scipy", "filter": "reverb"},
                {"name": "🔄 Echo Delay", "description": "Rhythmic delay repeats", "type": "scipy", "filter": "delay"},
                {"name": "🎸 Tube Overdrive", "description": "Warm tube saturation", "type": "scipy", "filter": "soft_dist"},
                {"name": "🎸 INSANE Fuzz", "description": "Destructive distortion", "type": "scipy", "filter": "hard_dist"}
            ]
            
        else:
            print("🔧 Creating NumPy effects with COMPRESSION, REVERB, DELAY, and EXTREME EQ...")
            
            effects = [
                {"name": "Original", "description": "Clean audio", "type": "bypass"},
                {"name": "🔊 MEGA Bass", "description": "EXTREME low end", "type": "simple", "low": 4.0, "mid": 1.0, "high": 0.4},
                {"name": "✨ MEGA Treble", "description": "EXTREME high end", "type": "simple", "low": 0.4, "mid": 1.0, "high": 4.0},
                {"name": "🎚️ Simple Compressor", "description": "Volume leveling", "type": "compressor_simple"},
                {"name": "😊 MEGA Scoop", "description": "EXTREME V-shape", "type": "simple", "low": 3.0, "mid": 0.15, "high": 3.0},
                {"name": "🌊 Simple Reverb", "description": "Reverb simulation", "type": "reverb_simple"},
                {"name": "🔄 Simple Delay", "description": "Echo effect", "type": "delay_simple"},
                {"name": "🎸 Soft Overdrive", "description": "Warm tube saturation", "type": "distortion", "drive": 0.5},
                {"name": "🎸 INSANE Fuzz", "description": "Destructive distortion", "type": "distortion", "drive": 0.95}
            ]
        
        return effects
    
    def apply_effect(self, audio: np.ndarray, effect: Dict) -> np.ndarray:
        """Apply audio effect including compression, reverb and delay"""
        try:
            if effect["type"] == "bypass":
                return audio.copy()
            elif effect["type"] == "scipy" and SCIPY_AVAILABLE:
                return self.apply_scipy_effect(audio, effect["filter"])
            elif effect["type"] == "distortion":
                return self.apply_distortion(audio, effect["drive"])
            elif effect["type"] == "compressor_simple":
                return self.apply_simple_compressor(audio)
            elif effect["type"] == "reverb_simple":
                return self.apply_simple_reverb(audio)
            elif effect["type"] == "delay_simple":
                return self.apply_simple_delay(audio)
            else:
                return self.apply_simple_effect(audio, effect)
        except Exception as e:
            print(f"      ❌ Effect failed: {e}")
            return audio.copy()
    
    def apply_scipy_effect(self, audio: np.ndarray, filter_type: str) -> np.ndarray:
        """Apply EXTREME SciPy effects"""
        nyquist = self.sample_rate // 2
        
        try:
            if filter_type == "bass":
                # Stable bass boost with proper gain staging
                try:
                    # Use lower order filters to avoid instability
                    b1, a1 = scipy.signal.butter(2, 150/nyquist, btype='low')
                    bass1 = scipy.signal.filtfilt(b1, a1, audio)
                    
                    # Add a resonant peak at bass frequency  
                    b2, a2 = scipy.signal.butter(2, [100/nyquist, 300/nyquist], btype='band')
                    bass2 = scipy.signal.filtfilt(b2, a2, audio)
                    
                    # Gentle boost to avoid clipping/pops
                    result = audio + 0.8 * bass1 + 0.6 * bass2
                    
                    # Soft limiting to prevent pops
                    return np.tanh(result * 1.2) * 0.8
                    
                except Exception as e:
                    print(f"        Bass filter error: {e}")
                    return audio
                
            elif filter_type == "treble":
                # INSANE treble boost
                b, a = scipy.signal.butter(4, 2000/nyquist, btype='high')
                treble = scipy.signal.filtfilt(b, a, audio)
                return audio + 1.5 * treble
                
            elif filter_type == "compression":
                # Dynamic range compression
                threshold = 0.3  # Compression threshold
                ratio = 4.0      # 4:1 compression ratio
                
                # Simple peak detector
                abs_audio = np.abs(audio)
                
                # Apply compression where signal exceeds threshold
                compressed = audio.copy()
                over_threshold = abs_audio > threshold
                
                if np.any(over_threshold):
                    # Compress the overage
                    excess = abs_audio[over_threshold] - threshold
                    compressed_excess = threshold + (excess / ratio)
                    
                    # Apply to audio maintaining polarity
                    compressed[over_threshold] = np.sign(audio[over_threshold]) * compressed_excess
                
                # Add some makeup gain
                return compressed * 1.5
                
            elif filter_type == "mid_scoop":
                # INSANE V-shaped EQ
                b_low, a_low = scipy.signal.butter(4, 350/nyquist, btype='low')
                lows = scipy.signal.filtfilt(b_low, a_low, audio)
                b_high, a_high = scipy.signal.butter(4, 1800/nyquist, btype='high')
                highs = scipy.signal.filtfilt(b_high, a_high, audio)
                b_mid, a_mid = scipy.signal.butter(4, [400/nyquist, 1800/nyquist], btype='band')
                mids = scipy.signal.filtfilt(b_mid, a_mid, audio)
                return 0.1 * audio + 1.2 * lows + 1.2 * highs - 0.8 * mids
                
            elif filter_type == "reverb":
                return self.apply_scipy_reverb(audio)
                
            elif filter_type == "delay":
                return self.apply_scipy_delay(audio)
                
            elif filter_type == "soft_dist":
                # Tube overdrive
                gain = 5.0
                driven = audio * gain
                distorted = np.tanh(driven) * 0.8
                b, a = scipy.signal.butter(1, 8000/nyquist, btype='low')
                return scipy.signal.filtfilt(b, a, distorted)
                
            elif filter_type == "hard_dist":
                # INSANE fuzz
                gain = 8.0
                driven = audio * gain
                clipped = np.clip(driven, -0.6, 0.6)
                fuzz = np.sign(clipped) * (clipped ** 2)
                combined = 0.7 * clipped + 0.3 * fuzz
                b, a = scipy.signal.butter(1, 6000/nyquist, btype='low')
                return scipy.signal.filtfilt(b, a, combined) * 0.7
                
        except Exception as e:
            print(f"      SciPy error: {e}")
            return audio
            
        return audio
    
    def apply_scipy_reverb(self, audio: np.ndarray) -> np.ndarray:
        """Cathedral reverb using SciPy"""
        try:
            reverb_length = int(1.2 * self.sample_rate)  # Shorter to avoid memory issues
            decay = np.exp(-2.5 * np.arange(reverb_length) / reverb_length)
            
            impulse = np.zeros(reverb_length)
            early_reflections = [0.02, 0.05, 0.08, 0.12, 0.18, 0.25]
            
            for reflection_time in early_reflections:
                idx = int(reflection_time * self.sample_rate)
                if idx < reverb_length:
                    impulse[idx] += np.random.uniform(0.3, 0.6)
            
            noise = np.random.normal(0, 0.08, reverb_length)
            impulse += noise * decay
            
            reverbed = scipy.signal.convolve(audio, impulse, mode='same')
            return 0.75 * audio + 0.25 * reverbed
            
        except Exception as e:
            print(f"      Reverb error: {e}")
            return audio
    
    def apply_scipy_delay(self, audio: np.ndarray) -> np.ndarray:
        """Rhythmic delay using SciPy"""
        try:
            delay_times = [0.15, 0.3, 0.45]  # Fewer taps for stability
            delay_gains = [0.5, 0.3, 0.2]
            
            delayed = audio.copy()
            
            for delay_time, gain in zip(delay_times, delay_gains):
                delay_samples = int(delay_time * self.sample_rate)
                if delay_samples < len(audio):
                    delay_signal = np.zeros_like(audio)
                    delay_signal[delay_samples:] = audio[:-delay_samples] * gain
                    delayed += delay_signal
            
            return delayed
            
        except Exception as e:
            print(f"      Delay error: {e}")
            return audio
    
    def apply_simple_compressor(self, audio: np.ndarray) -> np.ndarray:
        """Simple compressor using NumPy"""
        try:
            threshold = 0.4
            ratio = 3.0
            
            abs_audio = np.abs(audio)
            compressed = audio.copy()
            
            # Find peaks above threshold
            over_threshold = abs_audio > threshold
            
            if np.any(over_threshold):
                excess = abs_audio[over_threshold] - threshold
                compressed_excess = threshold + (excess / ratio)
                compressed[over_threshold] = np.sign(audio[over_threshold]) * compressed_excess
            
            return compressed * 1.3  # Makeup gain
            
        except Exception as e:
            print(f"      Simple compressor error: {e}")
            return audio
    
    def apply_simple_reverb(self, audio: np.ndarray) -> np.ndarray:
        """Simple reverb using multiple delays"""
        try:
            delays = [0.03, 0.07, 0.12, 0.19, 0.28, 0.39]  # Fewer delays
            gains = [0.4, 0.3, 0.25, 0.2, 0.15, 0.1]
            
            reverbed = audio.copy()
            
            for delay, gain in zip(delays, gains):
                delay_samples = int(delay * self.sample_rate)
                if delay_samples < len(audio):
                    delayed = np.zeros_like(audio)
                    delayed[delay_samples:] = audio[:-delay_samples] * gain
                    reverbed += delayed
            
            return reverbed * 0.6
            
        except Exception as e:
            print(f"      Simple reverb error: {e}")
            return audio
    
    def apply_simple_delay(self, audio: np.ndarray) -> np.ndarray:
        """Simple delay effect"""
        try:
            delay_time = 0.25
            delay_samples = int(delay_time * self.sample_rate)
            
            if delay_samples < len(audio):
                delayed = np.zeros_like(audio)
                delayed[delay_samples:] = audio[:-delay_samples] * 0.5
                return audio + delayed
            
            return audio
            
        except Exception as e:
            print(f"      Simple delay error: {e}")
            return audio
    
    def apply_distortion(self, audio: np.ndarray, drive: float) -> np.ndarray:
        """Apply EXTREME distortion"""
        try:
            gain = 2.0 + drive * 6.0  # Reduced max gain for stability
            driven = audio * gain
            
            if drive < 0.5:
                distorted = np.tanh(driven * 2.5) * 0.8
                harmonics = np.tanh(driven * 4) * 0.2
                distorted += harmonics
            else:
                distorted = np.tanh(driven * 1.8)
                clipped = np.clip(distorted, -0.6, 0.6)
                fuzz = np.sign(clipped) * np.abs(clipped) ** 0.8
                distorted = 0.7 * clipped + 0.3 * fuzz
            
            return distorted * (0.7 - drive * 0.1)
            
        except Exception as e:
            print(f"      Distortion error: {e}")
            return audio
    
    def apply_simple_effect(self, audio: np.ndarray, effect: Dict) -> np.ndarray:
        """Apply EXTREME simple EQ"""
        fft = np.fft.fft(audio)
        freqs = np.fft.fftfreq(len(audio), 1/self.sample_rate)
        
        low_band = np.abs(freqs) < 500
        mid_band = (np.abs(freqs) >= 500) & (np.abs(freqs) < 3000)
        high_band = np.abs(freqs) >= 3000
        
        # TRIPLE the effect intensity
        low_mult = effect.get("low", 1.0)
        mid_mult = effect.get("mid", 1.0)
        high_mult = effect.get("high", 1.0)
        
        if low_mult != 1.0:
            low_mult = 1.0 + (low_mult - 1.0) * 2.5  # Reduced from 3.0 for stability
        if mid_mult != 1.0:
            mid_mult = 1.0 + (mid_mult - 1.0) * 2.5
        if high_mult != 1.0:
            high_mult = 1.0 + (high_mult - 1.0) * 2.5
        
        fft[low_band] *= low_mult
        fft[mid_band] *= mid_mult
        fft[high_band] *= high_mult
        
        return np.fft.ifft(fft).real.astype(np.float32)
    
    def test_audio_processing(self, input_device: int = None, output_device: int = None) -> None:
        """Test COMPLETE audio processing"""
        print("=== COMPLETE AUDIO PROCESSING TEST ===")
        
        effects = self.create_effects()
        
        print(f"🎤 Recording {self.test_duration} seconds...")
        print("Speak loudly and clearly - longer clips for better effect testing!")
        
        try:
            recording = sd.rec(
                frames=int(self.test_duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                device=input_device,
                dtype=np.float32
            )
            sd.wait()
            
            audio_data = recording.flatten()
            rms_level = np.sqrt(np.mean(audio_data**2))
            
            print(f"✅ Recorded {len(audio_data)} samples")
            print(f"📊 RMS Level: {rms_level:.4f}")
            
            print(f"\n🎛️  Processing with {len(effects)} effects:")
            print("=" * 60)
            
            for i, effect in enumerate(effects):
                print(f"\n[{i+1}/{len(effects)}] {effect['name']}")
                print(f"    📝 {effect['description']}")
                
                try:
                    processed = self.apply_effect(audio_data, effect)
                    
                    # Normalize safely
                    max_val = np.max(np.abs(processed))
                    if max_val > 0:
                        processed = processed / max_val * 0.7
                    
                    print("    🔊 Playing...")
                    sd.play(processed, samplerate=self.sample_rate, device=output_device)
                    sd.wait()
                    print("    ✅ Done!")
                    sd.sleep(self.pause_duration)
                    
                except Exception as e:
                    print(f"    ❌ Failed: {e}")
                    continue
            
            print("\n" + "=" * 60)
            print("🎉 COMPLETE AUDIO TEST FINISHED!")
            
            distortion_count = sum(1 for e in effects if "🎸" in e["name"])
            reverb_count = sum(1 for e in effects if "🌊" in e["name"])  
            delay_count = sum(1 for e in effects if "🔄" in e["name"])
            compressor_count = sum(1 for e in effects if "🎚️" in e["name"])
            
            print(f"🎸 {distortion_count} distortion effects")
            print(f"🌊 {reverb_count} reverb effects")
            print(f"🔄 {delay_count} delay effects")
            print(f"🎚️ {compressor_count} compressor effects")
            print(f"🔊 EXTREME EQ settings")
            print("🧠 Ready for AI parameter control!")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
    
    def run_test_suite(self):
        """Run complete test suite"""
        print("🎵 SonicMind-AI EQFXLib - COMPLETE FEATURE TEST 🎵")
        print("=" * 60)
        print("✅ 10-second recordings")
        print("✅ Extreme EQ effects")
        print("✅ Compression")
        print("✅ Reverb and delay")
        print("✅ Distortion effects")
        print("=" * 60)
        
        # System info
        print("=== SYSTEM INFO ===")
        info = self.get_system_info()
        for key, value in info.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
        print()
        
        # Audio devices
        try:
            devices = sd.query_devices()
            print(f"=== FOUND {len(devices)} AUDIO DEVICES ===")
            
            default_input = sd.default.device[0]
            default_output = sd.default.device[1]
            
            if default_input is not None:
                input_info = sd.query_devices(default_input)
                print(f"Input:  {input_info['name']}")
                
            if default_output is not None:
                output_info = sd.query_devices(default_output)
                print(f"Output: {output_info['name']}")
                
            print()
            
            # Test processing
            if default_input is not None and default_output is not None:
                response = input("🎵 Test COMPLETE audio processing (10s recording + 9 effects)? (Y/n): ").lower()
                if response != 'n':
                    self.test_audio_processing(default_input, default_output)
            else:
                print("❌ No default audio devices available")
                
        except Exception as e:
            print(f"❌ Audio device error: {e}")

def main():
    """Main function"""
    tester = AudioTester()
    
    try:
        tester.run_test_suite()
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")

if __name__ == "__main__":
    main()