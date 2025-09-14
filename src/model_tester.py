#!/usr/bin/env python3
"""
Three-Way Model Tester
Tests original vs corrupted vs AI-enhanced audio
"""

import os
import numpy as np
import torch
import torch.nn as nn
import librosa
import soundfile as sf
import sounddevice as sd
from scipy import signal
import random

# =============================================================================
# AUDIO PROCESSOR
# =============================================================================

class AudioProcessor:
    def __init__(self, sample_rate=44100):
        self.sr = sample_rate
    
    def load_audio(self, filepath):
        """Load entire audio file"""
        try:
            audio, _ = librosa.load(filepath, sr=self.sr, mono=True, duration=None)
            print(f"Loaded audio: {len(audio)/self.sr:.1f} seconds")
            return audio
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def extract_features(self, audio):
        """Extract 12 audio features for AI model"""
        if len(audio) == 0:
            return np.zeros(12, dtype=np.float32)
        
        # Frequency analysis
        fft = np.fft.rfft(audio)
        magnitude = np.abs(fft)
        freqs = np.fft.rfftfreq(len(audio), 1/self.sr)
        
        # Energy in frequency bands
        bass = np.sum(magnitude[(freqs >= 20) & (freqs <= 250)])
        mids = np.sum(magnitude[(freqs >= 250) & (freqs <= 4000)])
        highs = np.sum(magnitude[(freqs >= 4000) & (freqs <= 20000)])
        total = bass + mids + highs + 1e-8
        
        # Audio characteristics
        rms = np.sqrt(np.mean(audio**2))
        peak = np.max(np.abs(audio))
        crest_factor = peak / (rms + 1e-8)
        spectral_centroid = np.sum(freqs * magnitude) / (np.sum(magnitude) + 1e-8)
        
        features = np.array([
            bass / total,             # Bass proportion
            mids / total,             # Mids proportion  
            highs / total,            # Highs proportion
            rms,                      # RMS level
            peak,                     # Peak level
            crest_factor,             # Dynamic range
            spectral_centroid / 1000, # Brightness
            np.mean(audio),           # DC offset
            np.std(audio),            # Overall level
            np.mean(np.diff(audio)),  # Rate of change
            np.std(np.diff(audio)),   # Variability
            len(audio) / self.sr      # Duration
        ], dtype=np.float32)
        
        return features
    
    def apply_eq(self, audio, eq_params):
        """Apply 5-band EQ - handles both formats"""
        result = audio.copy()
        
        for i in range(5):
            freq = np.clip(eq_params[i*3], 20, 20000)
            q = np.clip(eq_params[i*3 + 1], 0.1, 10.0)
            gain = np.clip(eq_params[i*3 + 2], -20, 20)
            
            if abs(gain) > 0.1:
                result = self._biquad_filter(result, freq, gain, q)
        
        return result
    
    def _biquad_filter(self, audio, freq, gain_db, q):
        """Apply biquad peaking EQ filter"""
        try:
            A = 10**(gain_db / 40.0)
            w0 = 2 * np.pi * freq / self.sr
            cos_w0 = np.cos(w0)
            sin_w0 = np.sin(w0)
            alpha = sin_w0 / (2 * q)
            
            b0 = 1 + alpha * A
            b1 = -2 * cos_w0
            b2 = 1 - alpha * A
            a0 = 1 + alpha / A
            a1 = -2 * cos_w0
            a2 = 1 - alpha / A
            
            if abs(a0) < 1e-10:
                return audio
            
            b = np.array([b0, b1, b2]) / a0
            a = np.array([1, a1/a0, a2/a0])
            
            return signal.lfilter(b, a, audio)
        except:
            return audio

# =============================================================================
# CORRUPTION GENERATOR (NEW VERSION)
# =============================================================================

class EQCorruptionGenerator:
    def __init__(self):
        # More realistic frequency bands covering full spectrum
        self.frequency_bands = [
            (20, 120),      # Sub-bass
            (120, 400),     # Bass 
            (400, 2000),    # Low-mids
            (2000, 8000),   # High-mids/presence
            (8000, 20000)   # Highs/air
        ]

    def generate_corruption_eq(self):
        # Always use all 5 frequency bands for complete coverage
        eq_params = []
        
        # Assign frequencies to cover full spectrum systematically
        base_frequencies = [80, 300, 1000, 4000, 10000]  # Professional standard bands
        
        for i in range(5):
            # Use base frequency with some variation
            freq_variation = random.uniform(0.7, 1.4)  # ±40% variation
            freq = base_frequencies[i] * freq_variation
            freq = np.clip(freq, 20, 20000)
            
            # Professional Q range
            q = random.uniform(0.3, 8.0)  # Much wider Q range
            
            # More moderate, musical gain ranges
            if random.random() < 0.3:  # 30% chance of no change
                gain = 0.0
            else:
                # Most professional EQ is in ±6dB range, occasionally more
                if random.random() < 0.8:  # 80% moderate adjustments
                    gain = random.uniform(-6, 6)
                else:  # 20% more aggressive adjustments
                    gain = random.uniform(-12, 12)
            
            eq_params.extend([freq, q, gain])

        return np.array(eq_params, dtype=np.float32)

# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

class DynamicEQModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Exact architecture matching your model
        self.network = nn.Sequential(
            nn.Linear(12, 512),      # network.0
            nn.ReLU(),               # network.1
            nn.BatchNorm1d(512),     # network.2
            nn.Dropout(0.3),         # network.3
            
            nn.Linear(512, 256),     # network.4
            nn.ReLU(),               # network.5
            nn.BatchNorm1d(256),     # network.6
            nn.Dropout(0.2),         # network.7
            
            nn.Linear(256, 128),     # network.8
            nn.ReLU(),               # network.9
            nn.BatchNorm1d(128),     # network.10
            nn.Dropout(0.1),         # network.11
            
            nn.Linear(128, 64),      # network.12
            nn.ReLU(),               # network.13
            nn.BatchNorm1d(64),      # network.14
            nn.Dropout(0.1),         # network.15
            
            nn.Linear(64, 32),       # network.16
            nn.ReLU(),               # network.17
            nn.BatchNorm1d(32),      # network.18
            nn.Dropout(0.05),        # network.19
            
            nn.Linear(32, 16),       # network.20
            nn.ReLU(),               # network.21
            nn.BatchNorm1d(16),      # network.22
            nn.Dropout(0.05),        # network.23
            
            nn.Linear(16, 8),        # network.24
            nn.ReLU(),               # network.25
            
            nn.Linear(8, 15)         # network.26 (final output)
        )
    
    def forward(self, features):
        raw = self.network(features)
        
        # Convert from interleaved to grouped format if needed
        params = raw.view(-1, 5, 3)  # [batch, 5_bands, 3_params]
        freqs = params[:, :, 0]      # Frequencies
        qs = params[:, :, 1]         # Q factors
        gains = params[:, :, 2]      # Gains
        
        # Return in interleaved format for apply_eq function
        return raw

# =============================================================================
# MODEL LOADER
# =============================================================================

def load_model(model_path):
    device = torch.device('mps' if torch.backends.mps.is_available() else 
                         'cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading model: {model_path}")
    print(f"Device: {device}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print(f"Epoch: {checkpoint.get('epoch', '?')}, Loss: {checkpoint.get('val_loss', '?'):.4f}")
    else:
        state_dict = checkpoint
    
    model = DynamicEQModel()
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {params:,} parameters")
    
    return model, device

# =============================================================================
# THREE-WAY TESTER
# =============================================================================

def test_three_way(audio_file, model_path="src/fma_eq_model_npy.pth"):
    print("Three-Way Audio Comparison Test")
    print("=" * 50)
    
    # Load model
    try:
        model, device = load_model(model_path)
    except Exception as e:
        print(f"Model loading failed: {e}")
        return None
    
    # Load audio
    processor = AudioProcessor()
    audio = processor.load_audio(audio_file)
    if audio is None:
        return None
    
    # Generate corruption
    corruptor = EQCorruptionGenerator()
    corruption_params = corruptor.generate_corruption_eq()
    corrupted_audio = processor.apply_eq(audio, corruption_params)
    
    print("Applied corruption:")
    for i in range(5):
        freq, q, gain = corruption_params[i*3], corruption_params[i*3+1], corruption_params[i*3+2]
        if abs(gain) > 0.1:
            print(f"  Band {i+1}: {freq:.0f}Hz, Q={q:.1f}, {gain:+.1f}dB")
    
    # Extract features from corrupted audio and predict correction
    features = processor.extract_features(corrupted_audio)
    
    with torch.no_grad():
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)
        correction_params = model(features_tensor).cpu().numpy()[0]
    
    # Apply AI enhancement to corrupted audio
    enhanced_audio = processor.apply_eq(corrupted_audio, correction_params)
    
    print("\nAI correction applied:")
    for i in range(5):
        freq, q, gain = correction_params[i*3], correction_params[i*3+1], correction_params[i*3+2]
        if abs(gain) > 0.1:
            print(f"  Band {i+1}: {freq:.0f}Hz, Q={q:.1f}, {gain:+.1f}dB")
    
    # Save all three versions
    os.makedirs('three_way_test', exist_ok=True)
    base_name = os.path.splitext(os.path.basename(audio_file))[0]
    
    def normalize(audio_data):
        peak = np.max(np.abs(audio_data))
        return audio_data / peak * 0.7 if peak > 0 else audio_data
    
    original_file = f'three_way_test/{base_name}_1_original.wav'
    corrupted_file = f'three_way_test/{base_name}_2_corrupted.wav'
    enhanced_file = f'three_way_test/{base_name}_3_ai_enhanced.wav'
    
    sf.write(original_file, normalize(audio), processor.sr)
    sf.write(corrupted_file, normalize(corrupted_audio), processor.sr)
    sf.write(enhanced_file, normalize(enhanced_audio), processor.sr)
    
    print(f"\nFiles saved:")
    print(f"  1. Original: {original_file}")
    print(f"  2. Corrupted: {corrupted_file}")
    print(f"  3. AI Enhanced: {enhanced_file}")
    
    # Three-way playback
    try:
        print("\nThree-way comparison:")
        
        input("Press ENTER to play ORIGINAL (reference)...")
        sd.play(normalize(audio), processor.sr)
        sd.wait()
        print("✓ Original playback complete")
        
        input("Press ENTER to play CORRUPTED (with problems)...")
        sd.play(normalize(corrupted_audio), processor.sr)
        sd.wait()
        print("✓ Corrupted playback complete")
        
        input("Press ENTER to play AI ENHANCED (corrected)...")
        sd.play(normalize(enhanced_audio), processor.sr)
        sd.wait()
        print("✓ Enhanced playback complete")
        
        print("\nEvaluation:")
        print("1. Does the corrupted version sound worse than original?")
        print("2. Does the AI enhanced version sound better than corrupted?")
        print("3. How close is the AI enhanced version to the original?")
        
    except Exception as e:
        print(f"Playback failed: {e}")
        print("But files were saved for manual listening.")
    
    return {
        'original': audio,
        'corrupted': corrupted_audio,
        'enhanced': enhanced_audio,
        'corruption_params': corruption_params,
        'correction_params': correction_params
    }

# =============================================================================
# RUN TEST
# =============================================================================

if __name__ == "__main__":
    # Your audio file
    audio_file = "fma_medium/149/149082.mp3"  # Change this to your test file
    
    result = test_three_way(audio_file)
    
    if result:
        print("\nTest complete!")
        print("Check the three_way_test/ folder for saved audio files")
        
        rating_corrupted = input("Rate corruption realism (1-5): ").strip()
        rating_enhancement = input("Rate AI enhancement quality (1-5): ").strip()
        
        rating_enhancement = input("Rate hybrid enhancement quality (1-5): ").strip()
        
        if rating_enhancement.isdigit():
            print(f"Thanks! Enhancement rating: {rating_enhancement}/5")
    else:
        print("Test failed!")