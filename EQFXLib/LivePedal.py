#!/usr/bin/env python3
"""
SonicMind-AI Live Effects Pedal
===============================
Modular real-time audio passthrough with hot-swappable effects.
Mono input/output, low-latency processing, ARM64 compatible.

Usage:
    pedal = LiveEffectsPedal()
    pedal.add_effect("compressor", threshold=0.3, ratio=4.0)
    pedal.add_effect("reverb", room_size=0.7, decay=0.5)
    pedal.start()  # Begin live processing
"""

import sounddevice as sd
import numpy as np
import threading
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Check for SciPy availability
SCIPY_AVAILABLE = False
try:
    import scipy.signal
    SCIPY_AVAILABLE = True
except ImportError:
    pass

@dataclass
class EffectParameter:
    """Parameter definition for effects"""
    name: str
    min_val: float
    max_val: float
    default: float
    description: str = ""

class BaseEffect(ABC):
    """Base class for all effects"""
    
    def __init__(self, name: str, sample_rate: int = 44100):
        self.name = name
        self.sample_rate = sample_rate
        self.enabled = True
        self.parameters = {}
        self.parameter_definitions = {}
        self._initialize_parameters()
    
    @abstractmethod
    def _initialize_parameters(self):
        """Initialize effect-specific parameters"""
        pass
    
    @abstractmethod
    def process(self, audio: np.ndarray) -> np.ndarray:
        """Process audio through this effect"""
        pass
    
    def set_parameter(self, name: str, value: float):
        """Set effect parameter with bounds checking"""
        if name in self.parameter_definitions:
            param_def = self.parameter_definitions[name]
            value = np.clip(value, param_def.min_val, param_def.max_val)
            self.parameters[name] = value
        else:
            raise ValueError(f"Unknown parameter: {name}")
    
    def get_parameter(self, name: str) -> float:
        """Get current parameter value"""
        return self.parameters.get(name, 0.0)
    
    def reset_parameters(self):
        """Reset all parameters to defaults"""
        for name, param_def in self.parameter_definitions.items():
            self.parameters[name] = param_def.default

class BypassEffect(BaseEffect):
    """Clean passthrough - no processing"""
    
    def _initialize_parameters(self):
        pass
    
    def process(self, audio: np.ndarray) -> np.ndarray:
        return audio.copy()

class CompressorEffect(BaseEffect):
    """Dynamic range compressor"""
    
    def _initialize_parameters(self):
        self.parameter_definitions = {
            "threshold": EffectParameter("threshold", 0.1, 0.8, 0.4, "Compression threshold"),
            "ratio": EffectParameter("ratio", 1.0, 10.0, 3.0, "Compression ratio"),
            "makeup_gain": EffectParameter("makeup_gain", 0.5, 3.0, 1.3, "Makeup gain")
        }
        self.reset_parameters()
    
    def process(self, audio: np.ndarray) -> np.ndarray:
        threshold = self.get_parameter("threshold")
        ratio = self.get_parameter("ratio")
        makeup = self.get_parameter("makeup_gain")
        
        abs_audio = np.abs(audio)
        compressed = audio.copy()
        
        # Apply compression above threshold
        over_threshold = abs_audio > threshold
        if np.any(over_threshold):
            excess = abs_audio[over_threshold] - threshold
            compressed_excess = threshold + (excess / ratio)
            compressed[over_threshold] = np.sign(audio[over_threshold]) * compressed_excess
        
        return compressed * makeup

class EQEffect(BaseEffect):
    """3-band EQ with bass, mid, treble"""
    
    def _initialize_parameters(self):
        self.parameter_definitions = {
            "bass": EffectParameter("bass", 0.1, 4.0, 1.0, "Bass gain (< 500Hz)"),
            "mid": EffectParameter("mid", 0.1, 4.0, 1.0, "Mid gain (500Hz-3kHz)"),
            "treble": EffectParameter("treble", 0.1, 4.0, 1.0, "Treble gain (> 3kHz)")
        }
        self.reset_parameters()
    
    def process(self, audio: np.ndarray) -> np.ndarray:
        if SCIPY_AVAILABLE:
            return self._process_scipy(audio)
        else:
            return self._process_fft(audio)
    
    def _process_scipy(self, audio: np.ndarray) -> np.ndarray:
        """High-quality SciPy filtering"""
        try:
            nyquist = self.sample_rate // 2
            
            # Split into frequency bands
            b_low, a_low = scipy.signal.butter(2, 500/nyquist, btype='low')
            b_high, a_high = scipy.signal.butter(2, 3000/nyquist, btype='high')
            b_mid, a_mid = scipy.signal.butter(2, [500/nyquist, 3000/nyquist], btype='band')
            
            bass_band = scipy.signal.filtfilt(b_low, a_low, audio)
            mid_band = scipy.signal.filtfilt(b_mid, a_mid, audio)
            treble_band = scipy.signal.filtfilt(b_high, a_high, audio)
            
            # Apply gains
            bass_gain = self.get_parameter("bass")
            mid_gain = self.get_parameter("mid")
            treble_gain = self.get_parameter("treble")
            
            return bass_band * bass_gain + mid_band * mid_gain + treble_band * treble_gain
            
        except Exception:
            return self._process_fft(audio)
    
    def _process_fft(self, audio: np.ndarray) -> np.ndarray:
        """FFT-based EQ fallback"""
        fft = np.fft.fft(audio)
        freqs = np.fft.fftfreq(len(audio), 1/self.sample_rate)
        
        # Frequency bands
        low_band = np.abs(freqs) < 500
        mid_band = (np.abs(freqs) >= 500) & (np.abs(freqs) < 3000)
        high_band = np.abs(freqs) >= 3000
        
        # Apply gains
        fft[low_band] *= self.get_parameter("bass")
        fft[mid_band] *= self.get_parameter("mid")
        fft[high_band] *= self.get_parameter("treble")
        
        return np.fft.ifft(fft).real.astype(np.float32)

class ReverbEffect(BaseEffect):
    """Reverb with adjustable room size and decay"""
    
    def _initialize_parameters(self):
        self.parameter_definitions = {
            "room_size": EffectParameter("room_size", 0.1, 1.0, 0.5, "Room size"),
            "decay": EffectParameter("decay", 0.1, 0.9, 0.4, "Decay time"),
            "mix": EffectParameter("mix", 0.0, 1.0, 0.3, "Wet/dry mix")
        }
        self.reset_parameters()
        self._impulse_cache = None
        self._last_room_size = None
    
    def _generate_impulse(self, room_size: float) -> np.ndarray:
        """Generate reverb impulse response"""
        length = int(room_size * 1.5 * self.sample_rate)  # Max 1.5 seconds
        decay_rate = 3.0 * (1.0 - room_size + 0.2)
        
        decay_curve = np.exp(-decay_rate * np.arange(length) / length)
        
        # Early reflections
        impulse = np.zeros(length)
        reflections = [0.02, 0.04, 0.07, 0.11, 0.16, 0.23]
        
        for reflection_time in reflections:
            idx = int(reflection_time * self.sample_rate)
            if idx < length:
                impulse[idx] += np.random.uniform(0.2, 0.5)
        
        # Add diffuse reverb tail
        noise = np.random.normal(0, 0.05, length)
        impulse += noise * decay_curve
        
        return impulse * decay_curve
    
    def process(self, audio: np.ndarray) -> np.ndarray:
        room_size = self.get_parameter("room_size")
        decay = self.get_parameter("decay")
        mix = self.get_parameter("mix")
        
        # Cache impulse response for efficiency
        if self._impulse_cache is None or self._last_room_size != room_size:
            self._impulse_cache = self._generate_impulse(room_size)
            self._last_room_size = room_size
        
        try:
            if SCIPY_AVAILABLE:
                reverbed = scipy.signal.convolve(audio, self._impulse_cache, mode='same')
            else:
                # Simple multi-tap delay reverb
                reverbed = self._simple_reverb(audio, room_size)
            
            # Apply decay and mix
            reverbed *= decay
            return (1.0 - mix) * audio + mix * reverbed
            
        except Exception:
            return audio
    
    def _simple_reverb(self, audio: np.ndarray, room_size: float) -> np.ndarray:
        """Simple reverb using multiple delays"""
        delays = np.array([0.03, 0.07, 0.12, 0.19, 0.28]) * room_size
        gains = np.array([0.4, 0.3, 0.25, 0.2, 0.15])
        
        reverbed = audio.copy()
        
        for delay, gain in zip(delays, gains):
            delay_samples = int(delay * self.sample_rate)
            if delay_samples < len(audio):
                delayed = np.zeros_like(audio)
                delayed[delay_samples:] = audio[:-delay_samples] * gain
                reverbed += delayed
        
        return reverbed

class DelayEffect(BaseEffect):
    """Delay/echo effect"""
    
    def _initialize_parameters(self):
        self.parameter_definitions = {
            "time": EffectParameter("time", 0.05, 1.0, 0.25, "Delay time (seconds)"),
            "feedback": EffectParameter("feedback", 0.0, 0.8, 0.4, "Feedback amount"),
            "mix": EffectParameter("mix", 0.0, 1.0, 0.3, "Wet/dry mix")
        }
        self.reset_parameters()
        self._delay_buffer = None
        self._buffer_pos = 0
    
    def process(self, audio: np.ndarray) -> np.ndarray:
        delay_time = self.get_parameter("time")
        feedback = self.get_parameter("feedback")
        mix = self.get_parameter("mix")
        
        delay_samples = int(delay_time * self.sample_rate)
        
        # Initialize delay buffer if needed
        if self._delay_buffer is None or len(self._delay_buffer) < delay_samples:
            self._delay_buffer = np.zeros(delay_samples)
            self._buffer_pos = 0
        
        output = np.zeros_like(audio)
        
        for i in range(len(audio)):
            # Get delayed sample
            delayed_sample = self._delay_buffer[self._buffer_pos]
            
            # Mix input with feedback
            input_sample = audio[i] + delayed_sample * feedback
            
            # Store in delay buffer
            self._delay_buffer[self._buffer_pos] = input_sample
            
            # Output mix
            output[i] = (1.0 - mix) * audio[i] + mix * delayed_sample
            
            # Advance buffer position
            self._buffer_pos = (self._buffer_pos + 1) % delay_samples
        
        return output

class DistortionEffect(BaseEffect):
    """Overdrive/distortion effect"""
    
    def _initialize_parameters(self):
        self.parameter_definitions = {
            "drive": EffectParameter("drive", 0.0, 1.0, 0.3, "Drive amount"),
            "tone": EffectParameter("tone", 0.1, 1.0, 0.7, "Tone control"),
            "level": EffectParameter("level", 0.1, 2.0, 0.8, "Output level")
        }
        self.reset_parameters()
    
    def process(self, audio: np.ndarray) -> np.ndarray:
        drive = self.get_parameter("drive")
        tone = self.get_parameter("tone")
        level = self.get_parameter("level")
        
        # Apply drive
        gain = 1.0 + drive * 4.0
        driven = audio * gain
        
        # Soft clipping distortion
        if drive < 0.3:
            distorted = np.tanh(driven * 1.5)
        elif drive < 0.7:
            distorted = np.tanh(driven) * 0.9
            distorted += np.sign(driven) * (np.abs(driven) ** 0.7) * 0.1
        else:
            distorted = np.clip(driven, -0.7, 0.7)
            distorted = np.sign(distorted) * (np.abs(distorted) ** 0.8)
        
        # Tone control (simple high-frequency rolloff)
        if SCIPY_AVAILABLE and tone < 0.9:
            try:
                cutoff = 2000 + tone * 6000  # 2kHz to 8kHz
                nyquist = self.sample_rate // 2
                b, a = scipy.signal.butter(1, cutoff/nyquist, btype='low')
                distorted = scipy.signal.filtfilt(b, a, distorted)
            except Exception:
                pass
        
        return distorted * level

class LiveEffectsPedal:
    """Main live effects pedal class"""
    
    def __init__(self, sample_rate: int = 44100, block_size: int = 256):
        self.sample_rate = sample_rate
        self.block_size = block_size
        
        # Audio stream
        self.stream = None
        self.is_running = False
        
        # Effect chain
        self.effects_chain: List[BaseEffect] = []
        self.available_effects = {
            "bypass": BypassEffect,
            "compressor": CompressorEffect,
            "eq": EQEffect,
            "reverb": ReverbEffect,
            "delay": DelayEffect,
            "distortion": DistortionEffect
        }
        
        # Performance monitoring
        self.cpu_load = 0.0
        self.dropout_count = 0
        
        # Thread safety
        self.lock = threading.Lock()
        
        print("🎸 SonicMind Live Effects Pedal initialized")
        print(f"📊 Sample Rate: {sample_rate} Hz, Block Size: {block_size}")
        print(f"🔧 SciPy: {'Available' if SCIPY_AVAILABLE else 'Not available (using fallbacks)'}")
    
    def add_effect(self, effect_type: str, **parameters) -> int:
        """Add effect to the chain"""
        if effect_type not in self.available_effects:
            raise ValueError(f"Unknown effect: {effect_type}")
        
        effect_class = self.available_effects[effect_type]
        effect = effect_class(f"{effect_type}_{len(self.effects_chain)}", self.sample_rate)
        
        # Set parameters
        for param_name, value in parameters.items():
            try:
                effect.set_parameter(param_name, value)
            except ValueError as e:
                print(f"Warning: {e}")
        
        with self.lock:
            self.effects_chain.append(effect)
        
        effect_id = len(self.effects_chain) - 1
        print(f"➕ Added {effect_type} (ID: {effect_id})")
        return effect_id
    
    def remove_effect(self, effect_id: int):
        """Remove effect from chain"""
        with self.lock:
            if 0 <= effect_id < len(self.effects_chain):
                removed = self.effects_chain.pop(effect_id)
                print(f"➖ Removed {removed.name}")
            else:
                print(f"❌ Invalid effect ID: {effect_id}")
    
    def set_effect_parameter(self, effect_id: int, parameter: str, value: float):
        """Set effect parameter in real-time"""
        with self.lock:
            if 0 <= effect_id < len(self.effects_chain):
                try:
                    self.effects_chain[effect_id].set_parameter(parameter, value)
                except ValueError as e:
                    print(f"❌ {e}")
            else:
                print(f"❌ Invalid effect ID: {effect_id}")
    
    def toggle_effect(self, effect_id: int):
        """Enable/disable effect"""
        with self.lock:
            if 0 <= effect_id < len(self.effects_chain):
                effect = self.effects_chain[effect_id]
                effect.enabled = not effect.enabled
                status = "ON" if effect.enabled else "OFF"
                print(f"🔘 {effect.name}: {status}")
    
    def clear_chain(self):
        """Remove all effects"""
        with self.lock:
            self.effects_chain.clear()
        print("🗑️  Effects chain cleared")
    
    def get_chain_info(self) -> List[Dict]:
        """Get current effects chain info"""
        info = []
        with self.lock:
            for i, effect in enumerate(self.effects_chain):
                effect_info = {
                    "id": i,
                    "name": effect.name,
                    "type": effect.__class__.__name__,
                    "enabled": effect.enabled,
                    "parameters": effect.parameters.copy()
                }
                info.append(effect_info)
        return info
    
    def _audio_callback(self, indata: np.ndarray, outdata: np.ndarray, 
                       frames: int, time_info, status):
        """Real-time audio processing callback"""
        if status:
            self.dropout_count += 1
        
        try:
            start_time = time.perf_counter()
            
            # Get mono input
            audio = indata[:, 0] if indata.ndim > 1 else indata.flatten()
            
            # Process through effects chain
            with self.lock:
                for effect in self.effects_chain:
                    if effect.enabled:
                        audio = effect.process(audio)
            
            # Normalize and output
            max_val = np.max(np.abs(audio))
            if max_val > 1.0:
                audio = audio / max_val * 0.9
            
            # Mono output
            outdata[:, 0] = audio
            if outdata.shape[1] > 1:
                outdata[:, 1] = audio  # Duplicate to stereo if needed
            
            # Calculate CPU load
            process_time = time.perf_counter() - start_time
            self.cpu_load = (process_time / (frames / self.sample_rate)) * 100
            
        except Exception as e:
            print(f"❌ Audio processing error: {e}")
            outdata.fill(0)  # Silence on error
    
    def start(self, input_device=None, output_device=None):
        """Start live audio processing"""
        if self.is_running:
            print("⚠️  Already running")
            return
        
        try:
            self.stream = sd.Stream(
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                device=(input_device, output_device),
                channels=1,
                callback=self._audio_callback,
                dtype=np.float32
            )
            
            self.stream.start()
            self.is_running = True
            
            print("🎵 Live processing STARTED")
            print(f"🎤 Input device: {sd.query_devices(self.stream.device[0])['name']}")
            print(f"🔊 Output device: {sd.query_devices(self.stream.device[1])['name']}")
            print("🎸 Effects pedal is LIVE!")
            
        except Exception as e:
            print(f"❌ Failed to start: {e}")
    
    def stop(self):
        """Stop live audio processing"""
        if not self.is_running:
            print("⚠️  Not running")
            return
        
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        
        self.is_running = False
        print("⏹️  Live processing STOPPED")
    
    def get_status(self) -> Dict:
        """Get pedal status"""
        return {
            "running": self.is_running,
            "effects_count": len(self.effects_chain),
            "cpu_load": self.cpu_load,
            "dropout_count": self.dropout_count,
            "sample_rate": self.sample_rate,
            "block_size": self.block_size
        }

# Example usage and presets
def create_clean_preset(pedal: LiveEffectsPedal):
    """Clean preset - minimal processing"""
    pedal.clear_chain()
    pedal.add_effect("compressor", threshold=0.6, ratio=2.0, makeup_gain=1.1)

def create_rock_preset(pedal: LiveEffectsPedal):
    """Rock guitar preset"""
    pedal.clear_chain()
    pedal.add_effect("compressor", threshold=0.4, ratio=3.0, makeup_gain=1.2)
    pedal.add_effect("distortion", drive=0.6, tone=0.7, level=0.8)
    pedal.add_effect("eq", bass=1.2, mid=0.8, treble=1.3)

def create_ambient_preset(pedal: LiveEffectsPedal):
    """Ambient/atmospheric preset"""
    pedal.clear_chain()
    pedal.add_effect("compressor", threshold=0.3, ratio=4.0, makeup_gain=1.3)
    pedal.add_effect("eq", bass=0.9, mid=1.0, treble=1.1)
    pedal.add_effect("delay", time=0.375, feedback=0.45, mix=0.25)
    pedal.add_effect("reverb", room_size=0.8, decay=0.6, mix=0.4)

def main():
    """Demo the live effects pedal"""
    pedal = LiveEffectsPedal(block_size=256)  # Lower latency
    
    print("\n=== LIVE EFFECTS PEDAL DEMO ===")
    
    # Show available devices
    devices = sd.query_devices()
    print(f"📱 Available devices: {len(devices)}")
    
    try:
        # Create a preset
        create_clean_preset(pedal)
        
        # Start processing
        pedal.start()
        
        print("\n🎛️  Live Controls:")
        print("Press Enter to stop, or use these commands:")
        print("  'rock' - Rock guitar preset")
        print("  'ambient' - Ambient preset") 
        print("  'clean' - Clean preset")
        print("  'status' - Show status")
        
        while True:
            cmd = input("> ").strip().lower()
            
            if cmd == "":
                break
            elif cmd == "rock":
                create_rock_preset(pedal)
                print("🎸 Rock preset loaded")
            elif cmd == "ambient":
                create_ambient_preset(pedal)
                print("🌊 Ambient preset loaded")
            elif cmd == "clean":
                create_clean_preset(pedal)
                print("✨ Clean preset loaded")
            elif cmd == "status":
                status = pedal.get_status()
                print(f"📊 CPU: {status['cpu_load']:.1f}%, Effects: {status['effects_count']}")
            else:
                print("❓ Unknown command")
    
    except KeyboardInterrupt:
        pass
    
    finally:
        pedal.stop()
        print("👋 Demo finished")

if __name__ == "__main__":
    main()