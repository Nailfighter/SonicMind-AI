#!/usr/bin/env python3
"""
Self-Supervised EQ Training System - Compatible with all PyTorch versions
Trains neural network to reverse "bad" EQ applied to professional music
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import glob
import random
from scipy import signal
from tqdm import tqdm
import librosa
import warnings
warnings.filterwarnings('ignore', message='.*Illegal Audio-MPEG-Header.*')

class BiquadEQ:
    """NumPy-based parametric EQ for data generation"""
    
    def __init__(self, sample_rate=44100):
        self.sr = sample_rate
    
    def apply_eq_band(self, audio, freq, gain_db, q=1.0):
        """Apply single EQ band using biquad filter"""
        if abs(gain_db) < 0.1:
            return audio
        
        # Clamp parameters to safe ranges
        freq = np.clip(freq, 20, min(20000, self.sr//2 - 100))
        q = np.clip(q, 0.1, 10.0)
        gain_db = np.clip(gain_db, -20, 20)
        
        A = 10**(gain_db / 40.0)
        w0 = 2 * np.pi * freq / self.sr
        cos_w0 = np.cos(w0)
        sin_w0 = np.sin(w0)
        alpha = sin_w0 / (2 * q)
        
        # Peaking EQ coefficients
        b0 = 1 + alpha * A
        b1 = -2 * cos_w0
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * cos_w0
        a2 = 1 - alpha / A
        
        # Normalize and ensure stability
        if abs(a0) < 1e-10:
            return audio
        
        b = np.array([b0, b1, b2]) / a0
        a = np.array([1, a1/a0, a2/a0])
        
        try:
            if len(audio.shape) == 1:
                return signal.lfilter(b, a, audio)
            else:
                result = np.zeros_like(audio)
                for ch in range(audio.shape[0]):
                    result[ch] = signal.lfilter(b, a, audio[ch])
                return result
        except:
            return audio  # Return original if filter fails
    
    def apply_eq(self, audio, eq_params):
        """Apply 5-band parametric EQ safely"""
        result = audio.copy()
        
        for i in range(5):
            try:
                freq = eq_params[i*3]
                q = eq_params[i*3 + 1]
                gain = eq_params[i*3 + 2]
                result = self.apply_eq_band(result, freq, gain, q)
            except:
                continue  # Skip problematic bands
        
        return result

class EQCorruptionGenerator:
    """Generates realistic "bad" EQ settings"""
    
    def __init__(self):
        # Common audio problems and their frequency ranges
        self.problems = [
            (80, 200, "muddy", (3, 8)),      # Muddy bass - boost
            (300, 800, "boxy", (2, 6)),      # Boxy mids - boost  
            (1500, 3500, "harsh", (2, 7)),  # Harsh mids - boost
            (4000, 8000, "sibilant", (2, 5)), # Sibilance - boost
            (200, 600, "thin", (-6, -2)),    # Thin sound - cut
            (8000, 16000, "dull", (-5, -2))  # Dull highs - cut
        ]
    
    def generate_corruption_eq(self):
        """Generate realistic corruption parameters"""
        eq_params = []
        
        # Select 3-4 problems to simulate
        num_problems = random.randint(3, 4)
        selected = random.sample(self.problems, num_problems)
        
        for freq_min, freq_max, problem_type, gain_range in selected:
            freq = random.uniform(freq_min, freq_max)
            q = random.uniform(0.5, 2.5)
            gain = random.uniform(gain_range[0], gain_range[1])
            eq_params.extend([freq, q, gain])
        
        # Pad to exactly 15 parameters (5 bands)
        while len(eq_params) < 15:
            eq_params.extend([1000.0, 1.0, 0.0])  # Neutral band
        
        return np.array(eq_params[:15], dtype=np.float32)

class AudioFeatureExtractor:
    """Fast feature extraction"""
    
    def __init__(self, sample_rate=44100):
        self.sr = sample_rate
    
    def extract_features(self, audio):
        """Extract essential features quickly"""
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=0)
        
        if len(audio) == 0:
            return np.zeros(12, dtype=np.float32)
        
        try:
            # FFT-based frequency analysis
            fft = np.fft.rfft(audio)
            magnitude = np.abs(fft)
            freq_bins = np.fft.rfftfreq(len(audio), 1/self.sr)
            
            # Energy in frequency bands
            bass = np.sum(magnitude[(freq_bins >= 20) & (freq_bins <= 250)])
            mids = np.sum(magnitude[(freq_bins >= 250) & (freq_bins <= 4000)]) 
            highs = np.sum(magnitude[(freq_bins >= 4000) & (freq_bins <= 20000)])
            total = bass + mids + highs + 1e-8
            
            # Time domain features  
            rms = np.sqrt(np.mean(audio**2))
            peak = np.max(np.abs(audio))
            crest_factor = peak / (rms + 1e-8)
            
            # Spectral features
            spectral_centroid = np.sum(freq_bins * magnitude) / (np.sum(magnitude) + 1e-8)
            
            features = [
                bass / total,           # Bass energy ratio
                mids / total,           # Mids energy ratio  
                highs / total,          # Highs energy ratio
                rms,                    # RMS energy
                peak,                   # Peak level
                crest_factor,           # Dynamic range
                spectral_centroid / 1000, # Spectral centroid (kHz)
                np.mean(audio),         # DC offset
                np.std(audio),          # Standard deviation
                np.mean(np.diff(audio)), # Mean delta
                np.std(np.diff(audio)),  # Delta variation
                len(audio) / self.sr    # Duration
            ]
            
            return np.array(features, dtype=np.float32)
            
        except:
            return np.zeros(12, dtype=np.float32)

class EQDataset(Dataset):
    """Dataset for EQ training"""
    
    def __init__(self, audio_dir="./fma_small/", segment_length=44100*3, max_files=2000):
        self.segment_length = segment_length
        self.eq_processor = BiquadEQ()
        self.corruption_gen = EQCorruptionGenerator()
        self.feature_extractor = AudioFeatureExtractor()
        
        # Find audio files
        self.audio_files = []
        for ext in ['*.mp3', '*.wav', '*.flac']:
            pattern = os.path.join(audio_dir, '**', ext)
            self.audio_files.extend(glob.glob(pattern, recursive=True))
        
        # Limit files for faster training during development
        if max_files and len(self.audio_files) > max_files:
            self.audio_files = random.sample(self.audio_files, max_files)
        
        # Filter out tiny files
        self.audio_files = [f for f in self.audio_files if os.path.getsize(f) > 10000]
        
        print(f"Using {len(self.audio_files)} audio files")
    
    def __len__(self):
        return len(self.audio_files) * 5  # 5 segments per file
    
    def load_audio_robust(self, path):
        """Robust audio loading with fallbacks"""
        try:
            # Try librosa first
            audio, _ = librosa.load(path, sr=44100, mono=True, duration=30)  # Max 30 seconds
            return audio.astype(np.float32)
        except:
            try:
                # Try with different parameters
                audio, sr = librosa.load(path, sr=None, mono=True, duration=10)
                if sr != 44100:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=44100)
                return audio.astype(np.float32)
            except:
                # Return silence if all fails
                return np.zeros(44100, dtype=np.float32)
    
    def __getitem__(self, idx):
        file_idx = idx // 5
        
        if file_idx >= len(self.audio_files):
            file_idx = file_idx % len(self.audio_files)
        
        audio_file = self.audio_files[file_idx]
        
        try:
            # Load audio
            audio = self.load_audio_robust(audio_file)
            
            if len(audio) < 1000:  # Skip very short clips
                raise Exception("Audio too short")
            
            # Random segment
            if len(audio) > self.segment_length:
                start = random.randint(0, len(audio) - self.segment_length)
                audio = audio[start:start + self.segment_length]
            else:
                audio = np.pad(audio, (0, self.segment_length - len(audio)))
            
            # Normalize
            if np.max(np.abs(audio)) > 1e-8:
                audio = audio / np.max(np.abs(audio)) * 0.8  # Leave headroom
            
            # Generate and apply corruption
            corruption_eq = self.corruption_gen.generate_corruption_eq()
            corrupted_audio = self.eq_processor.apply_eq(audio, corruption_eq)
            
            # Extract features from corrupted audio
            features = self.feature_extractor.extract_features(corrupted_audio)
            
            # Target: inverse of corruption (negative gains)
            target_eq = corruption_eq.copy()
            target_eq[2::3] *= -1  # Invert gains
            
            return (
                torch.FloatTensor(features),
                torch.FloatTensor(target_eq),
                torch.FloatTensor(audio),
                torch.FloatTensor(corrupted_audio)
            )
        
        except:
            # Return dummy data for failed loads
            return (
                torch.zeros(12, dtype=torch.float32),
                torch.zeros(15, dtype=torch.float32),
                torch.zeros(self.segment_length, dtype=torch.float32),
                torch.zeros(self.segment_length, dtype=torch.float32)
            )

class EQNeuralNetwork(nn.Module):
    """Compact neural network"""
    
    def __init__(self, input_size=12):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.ReLU(), 
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(32, 15)  # 5 bands × 3 params
        )
        
        # Parameter scaling
        self.register_buffer('freq_min', torch.tensor(20.0))
        self.register_buffer('freq_max', torch.tensor(20000.0))
        self.register_buffer('q_min', torch.tensor(0.3))
        self.register_buffer('q_max', torch.tensor(5.0))
        self.register_buffer('gain_min', torch.tensor(-15.0))
        self.register_buffer('gain_max', torch.tensor(15.0))
    
    def forward(self, features):
        raw_output = self.network(features)
        
        # Reshape to [batch, 5_bands, 3_params]
        params = raw_output.view(-1, 5, 3)
        
        # Scale parameters to realistic ranges
        frequencies = self.freq_min + torch.sigmoid(params[:, :, 0]) * (self.freq_max - self.freq_min)
        q_factors = self.q_min + torch.sigmoid(params[:, :, 1]) * (self.q_max - self.q_min)  
        gains = self.gain_min + torch.sigmoid(params[:, :, 2]) * (self.gain_max - self.gain_min)
        
        # Recombine
        scaled = torch.stack([frequencies, q_factors, gains], dim=2)
        return scaled.view(-1, 15)

class EQTrainer:
    """Training orchestrator"""
    
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        self.eq_processor = BiquadEQ()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
    
    def train_epoch(self, dataloader, optimizer, epoch):
        self.model.train()
        total_loss = 0
        num_batches = len(dataloader)
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch_idx, (features, target_eq, original_audio, corrupted_audio) in enumerate(progress_bar):
            # Move to device
            features = features.to(self.device)
            target_eq = target_eq.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            predicted_eq = self.model(features)
            
            # Loss computation
            param_loss = self.l1_loss(predicted_eq, target_eq)
            
            # Simple spectral loss (computed on CPU for stability)
            spectral_loss = 0
            if batch_idx % 10 == 0:  # Compute less frequently
                try:
                    batch_size = min(4, features.shape[0])  # Limit batch for efficiency
                    for i in range(batch_size):
                        pred_eq_np = predicted_eq[i].detach().cpu().numpy()
                        orig_np = original_audio[i].numpy()
                        corr_np = corrupted_audio[i].numpy()
                        
                        corrected = self.eq_processor.apply_eq(corr_np, pred_eq_np)
                        spectral_loss += np.mean((corrected - orig_np)**2)
                    
                    spectral_loss /= batch_size
                except:
                    spectral_loss = 0
            
            # Combined loss
            total_batch_loss = param_loss + 0.01 * spectral_loss
            
            # Backward pass
            total_batch_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Update progress
            batch_loss = total_batch_loss.item()
            total_loss += batch_loss
            
            progress_bar.set_postfix({
                'Loss': f'{batch_loss:.4f}',
                'Avg': f'{total_loss/(batch_idx+1):.4f}'
            })
        
        return total_loss / num_batches
    
    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for features, target_eq, _, _ in dataloader:
                features = features.to(self.device)
                target_eq = target_eq.to(self.device)
                
                predicted_eq = self.model(features)
                loss = self.l1_loss(predicted_eq, target_eq)
                total_loss += loss.item()
        
        return total_loss / len(dataloader)

def main():
    print("Starting EQ Training...")
    
    # Configuration
    BATCH_SIZE = 96
    LEARNING_RATE = 0.002
    EPOCHS = 15
    MAX_FILES = 1500  # Limit files for development
    
    # Setup device - prioritize MPS if available
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # Create dataset
    print("Loading dataset...")
    dataset = EQDataset(audio_dir="./fma_small/", max_files=MAX_FILES)
    
    if len(dataset.audio_files) == 0:
        print("No audio files found in ./fma_small/")
        print("Please download FMA-small dataset first!")
        return
    
    # Split dataset
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=0  # Set to 0 for compatibility
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
    print(f"Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}")
    
    # Initialize model
    model = EQNeuralNetwork(input_size=12)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Optimizer with scheduler
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # Trainer
    trainer = EQTrainer(model, device)
    
    # Training loop
    print("Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        # Train
        train_loss = trainer.train_epoch(train_loader, optimizer, epoch+1)
        
        # Validate
        val_loss = trainer.validate(val_loader)
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }
            torch.save(checkpoint, 'best_eq_model.pth')
            print("Saved best model!")
    
    print(f"Training completed! Best validation loss: {best_val_loss:.4f}")
    print("Model saved as 'best_eq_model.pth'")

if __name__ == "__main__":
    main()