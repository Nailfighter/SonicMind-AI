import numpy as np
import sounddevice as sd
import threading
import json
from scipy import signal
import time

class TwelveBandEQ:
    """
    12-Band Equalizer with real-time audio processing
    Frequency bands: 60, 120, 250, 500, 1k, 2k, 4k, 8k, 12k, 16k, 20k Hz
    """
    
    def __init__(self, sample_rate=44100, block_size=256):
        self.sample_rate = sample_rate
        self.block_size = block_size
        
        # Define the 12 frequency bands (Hz)
        self.frequencies = [60, 120, 250, 500, 1000, 2000, 4000, 8000, 12000, 16000, 20000]
        
        # EQ gains for each band (-20dB to +20dB)
        self.gains_db = [0.0] * 11  # 11 bands (we'll add one more)
        self.gains_db.append(0.0)   # 12th band
        
        # Create band-pass filters for each frequency
        self.filters = []
        self._create_filters()
        
        # Thread safety for real-time parameter updates
        self.parameter_lock = threading.Lock()
        
        # Audio stream
        self.stream = None
        self.is_running = False
        
    def _create_filters(self):
        """Create band-pass filters for each frequency band"""
        self.filters = []
        
        for i, freq in enumerate(self.frequencies):
            # Calculate filter boundaries
            if i == 0:
                # Low-pass for first band
                low_freq = 20
                high_freq = (freq + self.frequencies[i+1]) / 2
            elif i == len(self.frequencies) - 1:
                # High-pass for last band
                low_freq = (self.frequencies[i-1] + freq) / 2
                high_freq = 22000
            else:
                # Band-pass for middle bands
                low_freq = (self.frequencies[i-1] + freq) / 2
                high_freq = (freq + self.frequencies[i+1]) / 2
            
            # Create second-order sections (SOS) filter
            if i == 0:
                # Low-pass filter
                sos = signal.butter(2, high_freq / (self.sample_rate / 2), 
                                  btype='low', output='sos')
            elif i == len(self.frequencies) - 1:
                # High-pass filter
                sos = signal.butter(2, low_freq / (self.sample_rate / 2), 
                                  btype='high', output='sos')
            else:
                # Band-pass filter
                sos = signal.butter(2, [low_freq / (self.sample_rate / 2), 
                                      high_freq / (self.sample_rate / 2)], 
                                  btype='band', output='sos')
            
            self.filters.append({
                'sos': sos,
                'zi': signal.sosfilt_zi(sos),  # Initial conditions for filtering
                'freq': freq,
                'low': low_freq,
                'high': high_freq
            })
    
    def set_band_gain(self, band_index, gain_db):
        """
        Set gain for a specific frequency band
        
        Args:
            band_index (int): Band index (0-11)
            gain_db (float): Gain in dB (-20 to +20)
        """
        if 0 <= band_index < len(self.gains_db):
            with self.parameter_lock:
                # Clamp gain to reasonable range
                self.gains_db[band_index] = max(-20.0, min(20.0, gain_db))
                print(f"Band {band_index} ({self.frequencies[band_index]}Hz): {gain_db:.1f}dB")
    
    def get_band_gain(self, band_index):
        """Get current gain for a band"""
        if 0 <= band_index < len(self.gains_db):
            with self.parameter_lock:
                return self.gains_db[band_index]
        return 0.0
    
    def reset_all_bands(self):
        """Reset all bands to 0dB (flat response)"""
        with self.parameter_lock:
            self.gains_db = [0.0] * len(self.gains_db)
        print("All EQ bands reset to 0dB")
    
    def process_audio_block(self, audio_block):
        """
        Process a single audio block through the EQ
        
        Args:
            audio_block (numpy.ndarray): Input audio samples
            
        Returns:
            numpy.ndarray: EQ-processed audio
        """
        # Make sure we're working with the right shape
        if len(audio_block.shape) == 1:
            # Mono audio
            processed = np.zeros_like(audio_block)
        else:
            # Stereo - process left channel only for simplicity
            audio_block = audio_block[:, 0]
            processed = np.zeros_like(audio_block)
        
        # Apply each frequency band
        with self.parameter_lock:
            current_gains = self.gains_db.copy()
        
        for i, filter_data in enumerate(self.filters):
            if i < len(current_gains):
                # Apply band-pass filter
                filtered, filter_data['zi'] = signal.sosfilt(
                    filter_data['sos'], 
                    audio_block, 
                    zi=filter_data['zi']
                )
                
                # Apply gain (convert dB to linear)
                gain_linear = 10 ** (current_gains[i] / 20.0)
                processed += filtered * gain_linear
        
        return processed
    
    def audio_callback(self, indata, outdata, frames, time, status):
        """Real-time audio callback function - optimized for mono"""
        if status:
            print(f"Audio status: {status}")
        
        try:
            # Simple mono processing
            audio_input = indata.flatten() if len(indata.shape) > 1 else indata
            processed = self.process_audio_block(audio_input)
            
            # Output mono
            if len(outdata.shape) > 1:
                outdata[:, 0] = processed
            else:
                outdata[:] = processed
            
        except Exception as e:
            print(f"Audio processing error: {e}")
            # Fallback: pass through unprocessed audio
            try:
                outdata[:] = indata
            except:
                outdata.fill(0)  # Silence if all else fails
    
    def start_audio_stream(self):
        """Start real-time audio processing"""
        try:
            self.stream = sd.Stream(
                callback=self.audio_callback,
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                channels=2,  # Stereo input/output
                dtype=np.float32
            )
            self.stream.start()
            self.is_running = True
            print(f"EQ started - Sample rate: {self.sample_rate}Hz, Block size: {self.block_size}")
            print("12-Band EQ Frequencies:")
            for i, freq in enumerate(self.frequencies):
                print(f"  Band {i}: {freq}Hz")
            
        except Exception as e:
            print(f"Failed to start audio stream: {e}")
    
    def stop_audio_stream(self):
        """Stop audio processing"""
        if self.stream and self.is_running:
            self.stream.stop()
            self.stream.close()
            self.is_running = False
            print("EQ stopped")
    
    def save_preset(self, filename):
        """Save current EQ settings to JSON file"""
        preset = {
            'gains_db': self.gains_db,
            'frequencies': self.frequencies,
            'sample_rate': self.sample_rate
        }
        
        with open(filename, 'w') as f:
            json.dump(preset, f, indent=2)
        print(f"Preset saved to {filename}")
    
    def load_preset(self, filename):
        """Load EQ settings from JSON file"""
        try:
            with open(filename, 'r') as f:
                preset = json.load(f)
            
            with self.parameter_lock:
                self.gains_db = preset['gains_db']
            
            print(f"Preset loaded from {filename}")
            for i, gain in enumerate(self.gains_db):
                if i < len(self.frequencies):
                    print(f"  {self.frequencies[i]}Hz: {gain:.1f}dB")
                    
        except Exception as e:
            print(f"Failed to load preset: {e}")
    
    def print_current_settings(self):
        """Print current EQ settings"""
        print("\nCurrent EQ Settings:")
        for i, (freq, gain) in enumerate(zip(self.frequencies, self.gains_db)):
            print(f"  Band {i:2d}: {freq:5d}Hz = {gain:+5.1f}dB")


# Example usage and test functions
def interactive_eq_demo():
    """Interactive demo of the 12-band EQ"""
    eq = TwelveBandEQ()
    
    print("12-Band EQ Demo")
    print("===============")
    print("Commands:")
    print("  'set <band> <gain>' - Set band gain (e.g., 'set 0 5.0')")
    print("  'reset' - Reset all bands to 0dB")
    print("  'save <filename>' - Save preset")
    print("  'load <filename>' - Load preset")
    print("  'show' - Show current settings")
    print("  'start' - Start audio processing")
    print("  'stop' - Stop audio processing")
    print("  'quit' - Exit")
    
    eq.print_current_settings()
    
    while True:
        try:
            command = input("\nEQ> ").strip().lower()
            
            if command == 'quit':
                break
            elif command == 'start':
                eq.start_audio_stream()
            elif command == 'stop':
                eq.stop_audio_stream()
            elif command == 'reset':
                eq.reset_all_bands()
            elif command == 'show':
                eq.print_current_settings()
            elif command.startswith('set '):
                parts = command.split()
                if len(parts) == 3:
                    band = int(parts[1])
                    gain = float(parts[2])
                    eq.set_band_gain(band, gain)
                else:
                    print("Usage: set <band_index> <gain_db>")
            elif command.startswith('save '):
                filename = command.split(' ', 1)[1] + '.json'
                eq.save_preset(filename)
            elif command.startswith('load '):
                filename = command.split(' ', 1)[1] + '.json'
                eq.load_preset(filename)
            else:
                print("Unknown command")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    eq.stop_audio_stream()
    print("EQ Demo ended")


if __name__ == "__main__":
    # Quick test
    print("Testing 12-Band EQ Library...")
    
    # Create EQ instance
    eq = TwelveBandEQ()
    
    # Test setting some bands
    eq.set_band_gain(0, 3.0)   # Boost bass (60Hz)
    eq.set_band_gain(6, -2.0)  # Cut mids (4kHz)
    eq.set_band_gain(9, 5.0)   # Boost treble (16kHz)
    
    eq.print_current_settings()
    
    # Run interactive demo
    interactive_eq_demo()