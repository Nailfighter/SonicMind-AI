#!/usr/bin/env python3
"""
Multithreaded feature extraction for FMA-Medium (25,000 tracks)
Uses all available CPU cores for maximum speed
"""

import os
import glob
import numpy as np
import librosa
import random
from scipy import signal
import tqdm
from multiprocessing import Pool, cpu_count
import functools

# -----------------------------
# Audio processing classes
# -----------------------------
class BiquadEQ:
    def __init__(self, sample_rate=44100):
        self.sr = sample_rate

    def apply_eq_band(self, audio, freq, gain_db, q=1.0):
        if abs(gain_db) < 0.1:
            return audio

        freq = np.clip(freq, 20, min(20000, self.sr//2 - 100))
        q = np.clip(q, 0.1, 10.0)
        gain_db = np.clip(gain_db, -20, 20)

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

        try:
            return signal.lfilter(b, a, audio)
        except:
            return audio

    def apply_eq(self, audio, eq_params):
        result = audio.copy()
        for i in range(5):
            freq = eq_params[i*3]
            q = eq_params[i*3 + 1]
            gain = eq_params[i*3 + 2]
            result = self.apply_eq_band(result, freq, gain, q)
        return result

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
        
        # Realistic EQ problems with moderate gains
        self.problems = [
            # Bass problems
            (60, 150, "muddy_bass", (-8, 8)),
            (80, 200, "boomy", (-6, 6)),
            
            # Midrange problems  
            (200, 500, "muddy_mids", (-6, 6)),
            (400, 1000, "boxy", (-8, 8)),
            (800, 2000, "honky", (-6, 6)),
            (1500, 4000, "harsh", (-8, 8)),
            
            # High frequency problems
            (3000, 6000, "sibilant", (-6, 6)),
            (5000, 10000, "brittle", (-6, 6)),
            (8000, 16000, "dull", (-8, 8)),
            (10000, 20000, "lack_air", (-6, 8))
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

class AudioFeatureExtractor:
    def __init__(self, sample_rate=44100):
        self.sr = sample_rate

    def extract_features(self, audio):
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=0)

        if len(audio) == 0:
            return np.zeros(12, dtype=np.float32)

        try:
            fft = np.fft.rfft(audio)
            magnitude = np.abs(fft)
            freq_bins = np.fft.rfftfreq(len(audio), 1/self.sr)

            # Energy in bands
            bass = np.sum(magnitude[(freq_bins >= 20) & (freq_bins <= 250)])
            mids = np.sum(magnitude[(freq_bins >= 250) & (freq_bins <= 4000)])
            highs = np.sum(magnitude[(freq_bins >= 4000) & (freq_bins <= 20000)])
            total = bass + mids + highs + 1e-8

            rms = np.sqrt(np.mean(audio**2))
            peak = np.max(np.abs(audio))
            crest_factor = peak / (rms + 1e-8)
            spectral_centroid = np.sum(freq_bins * magnitude) / (np.sum(magnitude) + 1e-8)

            features = [
                bass / total, mids / total, highs / total,
                rms, peak, crest_factor, spectral_centroid / 1000,
                np.mean(audio), np.std(audio),
                np.mean(np.diff(audio)), np.std(np.diff(audio)),
                len(audio) / self.sr
            ]

            return np.array(features, dtype=np.float32)
        except:
            return np.zeros(12, dtype=np.float32)


def process_audio_file(audio_file):
    """Process a single audio file and return features/targets"""
    try:
        segment_length = 44100 * 30  # 30 seconds
        eq_gen = EQCorruptionGenerator()
        eq_processor = BiquadEQ()
        feature_extractor = AudioFeatureExtractor()
        
        audio, _ = librosa.load(audio_file, sr=44100, mono=True, duration=None)
        if len(audio) < 1000:
            return [], []

        file_features = []
        file_targets = []

        for _ in range(2):  # 2 segments per file
            if len(audio) >= segment_length:
                if len(audio) > segment_length:
                    start = random.randint(0, len(audio) - segment_length)
                    segment = audio[start:start + segment_length]
                else:
                    segment = audio
            else:
                segment = np.pad(audio, (0, segment_length - len(audio)))

            if np.max(np.abs(segment)) > 1e-8:
                segment = segment / np.max(np.abs(segment)) * 0.8

            corruption_eq = eq_gen.generate_corruption_eq()
            corrupted_audio = eq_processor.apply_eq(segment, corruption_eq)
            feature_vector = feature_extractor.extract_features(corrupted_audio)

            target_eq = corruption_eq.copy()
            target_eq[2::3] *= -1

            file_features.append(feature_vector)
            file_targets.append(target_eq)

        return file_features, file_targets
    except:
        return [], []

def main():
    dataset_dir = "fma_medium"  # Changed to fma_medium
    output_dir = "./precomputed_features"
    os.makedirs(output_dir, exist_ok=True)

    print("Finding audio files in fma_medium...")
    audio_files = []
    for ext in ['*.mp3', '*.wav', '*.flac']:
        audio_files.extend(glob.glob(os.path.join(dataset_dir, "**", ext), recursive=True))

    print(f"Found {len(audio_files)} audio files")

    # Use all available CPU cores
    num_processes = cpu_count()
    print(f"Using all {num_processes} CPU cores for processing")

    print("Processing files in parallel...")
    with Pool(num_processes) as pool:
        results = list(tqdm.tqdm(
            pool.imap(process_audio_file, audio_files),
            total=len(audio_files),
            desc="Processing files"
        ))

    # Collect all features and targets
    all_features = []
    all_targets = []
    
    for features, targets in results:
        all_features.extend(features)
        all_targets.extend(targets)

    if not all_features:
        print("No valid features extracted!")
        return

    features_array = np.stack(all_features)
    targets_array = np.stack(all_targets)

    np.save(os.path.join(output_dir, "features.npy"), features_array)
    np.save(os.path.join(output_dir, "targets.npy"), targets_array)

    print(f"Completed! Features: {features_array.shape}, Targets: {targets_array.shape}")
    print(f"Total samples from {len(audio_files)} files: {len(features_array):,}")

if __name__ == "__main__":
    main()