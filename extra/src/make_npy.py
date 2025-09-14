#!/usr/bin/env python3
"""
Precompute features and targets for FMA-Small
Saves features and EQ targets to .npy for fast loading
Uses 30-second clips for better musical context
"""

import os
import glob
import numpy as np
import librosa
import random
from scipy import signal
import tqdm

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
        self.problems = [
            (80, 200, "muddy", (3, 12)),
            (300, 800, "boxy", (2, 8)),
            (1500, 3500, "harsh", (3, 10)),
            (4000, 8000, "sibilant", (2, 8)),
            (200, 600, "thin", (-10, -3)),
            (8000, 16000, "dull", (-12, -4))
        ]

    def generate_corruption_eq(self):
        eq_params = []
        num_problems = random.randint(3, 5)
        selected = random.sample(self.problems, num_problems)

        for freq_min, freq_max, problem_type, gain_range in selected:
            freq = random.uniform(freq_min, freq_max)
            q = random.uniform(0.7, 3.0)
            gain = random.uniform(gain_range[0], gain_range[1])
            eq_params.extend([freq, q, gain])

        while len(eq_params) < 15:
            eq_params.extend([1000.0, 1.0, 0.0])

        return np.array(eq_params[:15], dtype=np.float32)

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

# -----------------------------
# Precompute features
# -----------------------------
def main():
    dataset_dir = "./fma_small"  # replace with your dataset path
    output_dir = "./precomputed_features"
    os.makedirs(output_dir, exist_ok=True)

    segment_length = 44100 * 30  # 30 seconds
    eq_gen = EQCorruptionGenerator()
    eq_processor = BiquadEQ()
    feature_extractor = AudioFeatureExtractor()

    audio_files = []
    for ext in ['*.mp3', '*.wav', '*.flac']:
        audio_files.extend(glob.glob(os.path.join(dataset_dir, "**", ext), recursive=True))

    features_list = []
    targets_list = []

    for audio_file in tqdm.tqdm(audio_files, desc="Processing audio files"):
        try:
            # Load full 30-second track (FMA tracks are ~30 seconds)
            audio, _ = librosa.load(audio_file, sr=44100, mono=True, duration=None)
            if len(audio) < 1000:
                continue

            # Since FMA tracks are already ~30 seconds, use fewer segments per file
            for _ in range(2):  # 2 segments per file instead of 8
                if len(audio) >= segment_length:
                    # Take full track or a 30-second segment
                    if len(audio) > segment_length:
                        start = random.randint(0, len(audio) - segment_length)
                        segment = audio[start:start + segment_length]
                    else:
                        segment = audio
                else:
                    # Pad if shorter than 30 seconds
                    segment = np.pad(audio, (0, segment_length - len(audio)))

                # Normalize audio
                if np.max(np.abs(segment)) > 1e-8:
                    segment = segment / np.max(np.abs(segment)) * 0.8

                # Generate corruption
                corruption_eq = eq_gen.generate_corruption_eq()
                corrupted_audio = eq_processor.apply_eq(segment, corruption_eq)
                feature_vector = feature_extractor.extract_features(corrupted_audio)

                # Target: inverse gain (correction)
                target_eq = corruption_eq.copy()
                target_eq[2::3] *= -1  # Invert gains at positions 2, 5, 8, 11, 14

                features_list.append(feature_vector)
                targets_list.append(target_eq)
        except Exception as e:
            # Skip problematic files
            print(f"Skipping {audio_file}: {e}")
            continue

    # Convert to numpy arrays
    features_array = np.stack(features_list)
    targets_array = np.stack(targets_list)

    np.save(os.path.join(output_dir, "features.npy"), features_array)
    np.save(os.path.join(output_dir, "targets.npy"), targets_array)

    print(f"Saved features: {features_array.shape}, targets: {targets_array.shape}")
    print(f"Using 30-second audio segments for improved musical context")

if __name__ == "__main__":
    main()