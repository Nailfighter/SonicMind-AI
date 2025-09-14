import sounddevice as sd
import numpy as np
import threading
import time
from typing import List, Dict, Any, Optional

# Check for SciPy
SCIPY_AVAILABLE = False
try:
    import scipy.signal
    SCIPY_AVAILABLE = True
    print("SciPy found, using for advanced effects.")
except ImportError:
    print("SciPy not found, using NumPy for effects. Some effects may be disabled or have lower quality.")


class BaseEffect:
    """
    Base class for all audio effects.
    """
    def __init__(self):
        self.parameters = {}
        self.lock = threading.RLock()

    def set_parameter(self, name: str, value: Any):
        """
        Set a parameter for the effect.
        """
        with self.lock:
            if name in self.parameters:
                self.parameters[name] = value
            else:
                raise ValueError(f"Unknown parameter: {name}")

    def get_parameter(self, name: str) -> Any:
        """
        Get a parameter of the effect.
        """
        return self.parameters.get(name)

    def process(self, audio: np.ndarray) -> np.ndarray:
        """
        Process the audio. This method should be overridden by subclasses.
        """
        raise NotImplementedError

class BypassEffect(BaseEffect):
    """
    An effect that does nothing, just passes the audio through.
    """
    def process(self, audio: np.ndarray) -> np.ndarray:
        return audio

class GainEffect(BaseEffect):
    """
    An effect that applies a simple gain.
    """
    def __init__(self, gain: float = 1.0):
        super().__init__()
        self.parameters['gain'] = gain

    def process(self, audio: np.ndarray) -> np.ndarray:
        with self.lock:
            return audio * self.parameters['gain']

class DistortionEffect(BaseEffect):
    """
    A simple distortion effect.
    """
    def __init__(self, drive: float = 0.5):
        super().__init__()
        self.parameters['drive'] = drive # 0.0 to 1.0

    def process(self, audio: np.ndarray) -> np.ndarray:
        with self.lock:
            drive = self.parameters['drive']
            gain = 1 + drive * 19
            return np.tanh(audio * gain)

class DelayEffect(BaseEffect):
    """
    A simple delay effect.
    """
    def __init__(self, delay_time: float = 0.5, feedback: float = 0.5, mix: float = 0.5, sample_rate: int = 44100):
        super().__init__()
        self.parameters['delay_time'] = delay_time
        self.parameters['feedback'] = feedback
        self.parameters['mix'] = mix
        self.sample_rate = sample_rate
        self.delay_buffer = np.zeros(int(self.sample_rate * 2)) # Max 2 seconds delay
        self.write_pos = 0

    def process(self, audio: np.ndarray) -> np.ndarray:
        with self.lock:
            delay_samples = int(self.parameters['delay_time'] * self.sample_rate)
            feedback = self.parameters['feedback']
            mix = self.parameters['mix']
            
            output = np.zeros_like(audio)
            for i in range(len(audio)):
                read_pos = (self.write_pos - delay_samples + len(self.delay_buffer)) % len(self.delay_buffer)
                delayed_sample = self.delay_buffer[read_pos]
                
                output[i] = audio[i] + delayed_sample * mix
                
                self.delay_buffer[self.write_pos] = audio[i] + delayed_sample * feedback
                self.write_pos = (self.write_pos + 1) % len(self.delay_buffer)
                
            return (1 - mix) * audio + mix * output

class ReverbEffect(BaseEffect):
    """
    A Schroeder reverberator using SciPy's lfilter for efficiency.
    """
    def __init__(self, room_size=0.5, decay=0.5, damping=0.5, mix=0.5, sample_rate=44100):
        super().__init__()
        if not SCIPY_AVAILABLE:
            # If SciPy is not available, we create a dummy effect that does nothing.
            print("Warning: ReverbEffect requires SciPy. It will act as a bypass effect.")
            self.process = lambda audio: audio
            self.parameters = {'room_size': room_size, 'decay': decay, 'damping': damping, 'mix': mix}
            return

        self.parameters = {'room_size': room_size, 'decay': decay, 'damping': damping, 'mix': mix}
        self.sample_rate = sample_rate

        # Schroeder reverberator parameters
        self.comb_delays_base = np.array([1687, 1601, 2053, 2251]) / 44100.0 # in seconds
        self.allpass_delays_base = np.array([556, 441, 341, 225]) / 44100.0 # in seconds
        self.allpass_feedback = 0.5

        self.comb_zi = [np.array([]) for _ in self.comb_delays_base]
        self.allpass_zi = [np.array([]) for _ in self.allpass_delays_base]
        
        self._update_filters()

    def _update_filters(self):
        room_size = self.parameters['room_size']
        decay = self.parameters['decay']
        
        # Scale delay times by room_size
        self.comb_delays = np.floor(self.comb_delays_base * self.sample_rate * room_size).astype(int)
        self.allpass_delays = np.floor(self.allpass_delays_base * self.sample_rate).astype(int) # All-pass delays are often fixed

        # Comb filters
        self.comb_filters = []
        for i, delay in enumerate(self.comb_delays):
            if delay == 0: # Avoid issues with zero delay
                b = np.array([1.0])
                a = np.array([1.0])
            else:
                b = np.array([1.0])
                a = np.zeros(delay + 1)
                a[0] = 1.0
                a[delay] = -decay
            
            if len(self.comb_zi[i]) != len(a) - 1:
                self.comb_zi[i] = np.zeros(len(a) - 1)
            self.comb_filters.append({'b': b, 'a': a})

        # All-pass filters
        self.allpass_filters = []
        for i, delay in enumerate(self.allpass_delays):
            if delay == 0:
                b = np.array([1.0])
                a = np.array([1.0])
            else:
                b = np.zeros(delay + 1)
                b[0] = -self.allpass_feedback
                b[delay] = 1.0
                a = np.zeros(delay + 1)
                a[0] = 1.0
                a[delay] = -self.allpass_feedback
            
            if len(self.allpass_zi[i]) != len(a) - 1:
                self.allpass_zi[i] = np.zeros(len(a) - 1)
            self.allpass_filters.append({'b': b, 'a': a})

    def set_parameter(self, name: str, value: Any):
        super().set_parameter(name, value)
        if name in ['room_size', 'decay']:
            with self.lock:
                self._update_filters()

    def process(self, audio: np.ndarray) -> np.ndarray:
        with self.lock:
            if not SCIPY_AVAILABLE:
                return audio

            mix = self.parameters['mix']

            comb_output = np.zeros_like(audio)
            # Parallel comb filters
            for i in range(len(self.comb_filters)):
                filt = self.comb_filters[i]
                y, self.comb_zi[i] = scipy.signal.lfilter(filt['b'], filt['a'], audio, zi=self.comb_zi[i])
                comb_output += y

            comb_output /= len(self.comb_filters)

            # Series all-pass filters
            allpass_output = comb_output
            for i in range(len(self.allpass_filters)):
                filt = self.allpass_filters[i]
                allpass_output, self.allpass_zi[i] = scipy.signal.lfilter(filt['b'], filt['a'], allpass_output, zi=self.allpass_zi[i])

            # Damping is not implemented here for simplicity, but could be added as a LPF
            # on the feedback path of the comb filters. The original implementation's damping
            # was just an attenuation factor.

            wet_signal = allpass_output
            dry_signal = audio
            
            return (1 - mix) * dry_signal + mix * wet_signal

class CompressorEffect(BaseEffect):
    """
    A simple compressor effect.
    """
    def __init__(self, threshold=-20.0, ratio=4.0, attack=0.01, release=0.1, sample_rate=44100):
        super().__init__()
        self.parameters = {'threshold': threshold, 'ratio': ratio, 'attack': attack, 'release': release}
        self.sample_rate = sample_rate
        self.envelope = 0.0

    def process(self, audio: np.ndarray) -> np.ndarray:
        with self.lock:
            threshold = 10 ** (self.parameters['threshold'] / 20)
            ratio = self.parameters['ratio']
            attack_coeff = np.exp(-1.0 / (self.parameters['attack'] * self.sample_rate))
            release_coeff = np.exp(-1.0 / (self.parameters['release'] * self.sample_rate))
            
            output = np.zeros_like(audio)
            for i in range(len(audio)):
                # Level detection
                level = abs(audio[i])
                if level > self.envelope:
                    self.envelope = attack_coeff * self.envelope + (1 - attack_coeff) * level
                else:
                    self.envelope = release_coeff * self.envelope + (1 - release_coeff) * level

                # Gain computation
                gain = 1.0
                if self.envelope > threshold:
                    gain = threshold / self.envelope * (1.0/ratio - 1.0) + 1.0

                output[i] = audio[i] * gain
            return output

class EQEffect(BaseEffect):
    """
    A 5-band equalizer.
    """
    def __init__(self, sample_rate=44100):
        super().__init__()
        self.parameters = {
            'sub_bass': 1.0, 'bass': 1.0, 'mid': 1.0, 'upper_mid': 1.0, 'treble': 1.0
        }
        self.sample_rate = sample_rate
        self.nyquist = sample_rate / 2
        
        if SCIPY_AVAILABLE:
            self._init_scipy_filters()
        else:
            self._init_fft_bands()

    def _init_scipy_filters(self):
        self.filters = {
            'sub_bass': scipy.signal.butter(2, 80 / self.nyquist, btype='low', output='sos'),
            'bass': scipy.signal.butter(2, [80 / self.nyquist, 250 / self.nyquist], btype='band', output='sos'),
            'mid': scipy.signal.butter(2, [250 / self.nyquist, 2000 / self.nyquist], btype='band', output='sos'),
            'upper_mid': scipy.signal.butter(2, [2000 / self.nyquist, 6000 / self.nyquist], btype='band', output='sos'),
            'treble': scipy.signal.butter(2, 6000 / self.nyquist, btype='high', output='sos')
        }
        self.z = {band: np.zeros((self.filters[band].shape[0], 2)) for band in self.filters}

    def _init_fft_bands(self):
        # Pre-calculate frequency bands for FFT-based EQ
        # This is a placeholder for a proper FFT implementation if SciPy is not available
        pass

    def process(self, audio: np.ndarray) -> np.ndarray:
        with self.lock:
            if SCIPY_AVAILABLE:
                return self._process_scipy(audio)
            else:
                return self._process_fft(audio)

    def _process_scipy(self, audio: np.ndarray) -> np.ndarray:
        output = np.zeros_like(audio)
        for band, sos in self.filters.items():
            filtered_signal, self.z[band] = scipy.signal.sosfilt(sos, audio, zi=self.z[band])
            output += filtered_signal * (self.parameters[band] - 1.0)
        return audio + output

    def _process_fft(self, audio: np.ndarray) -> np.ndarray:
        # Fallback to a simple FFT-based EQ if SciPy is not available
        fft_data = np.fft.rfft(audio)
        freqs = np.fft.rfftfreq(len(audio), 1.0 / self.sample_rate)

        fft_data[freqs < 80] *= self.parameters['sub_bass']
        fft_data[(freqs >= 80) & (freqs < 250)] *= self.parameters['bass']
        fft_data[(freqs >= 250) & (freqs < 2000)] *= self.parameters['mid']
        fft_data[(freqs >= 2000) & (freqs < 6000)] *= self.parameters['upper_mid']
        fft_data[freqs >= 6000] *= self.parameters['treble']

        return np.fft.irfft(fft_data, n=len(audio)).astype(np.float32)


class LiveAudioProcessor:
    """
    Handles live audio input, processing with a chain of effects, and output.
    """
    def __init__(self, sample_rate: int = 44100, block_size: int = 512):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.effects: List[BaseEffect] = [BypassEffect()]
        self.stream = None
        self.lock = threading.Lock()

    def audio_callback(self, indata: np.ndarray, outdata: np.ndarray, frames: int, time, status: sd.CallbackFlags):
        """
        The callback function for the audio stream.
        """
        if status:
            print(status, flush=True)
        
        processed_data = indata.copy()
        is_mono = processed_data.shape[1] == 1
        if is_mono:
            processed_data = processed_data.ravel()

        with self.lock:
            for effect in self.effects:
                processed_data = effect.process(processed_data)
        
        if is_mono:
            outdata[:] = processed_data.reshape(-1, 1)
        else:
            outdata[:] = processed_data

    def start(self):
        """
        Start the audio stream.
        """
        if self.stream is None:
            self.stream = sd.Stream(
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                channels=1,
                dtype=np.float32,
                callback=self.audio_callback
            )
            self.stream.start()

    def stop(self):
        """
        Stop the audio stream.
        """
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def add_effect(self, effect: BaseEffect, index: Optional[int] = None):
        """
        Add an effect to the effect chain.
        """
        with self.lock:
            if index is None:
                self.effects.append(effect)
            else:
                self.effects.insert(index, effect)

    def remove_effect(self, index: int):
        """
        Remove an effect from the effect chain.
        """
        with self.lock:
            if 0 <= index < len(self.effects):
                self.effects.pop(index)

    def list_effects(self):
        """
        List the current effects in the chain.
        """
        with self.lock:
            for i, effect in enumerate(self.effects):
                print(f"{i}: {effect.__class__.__name__}")

if __name__ == '__main__':
    pass