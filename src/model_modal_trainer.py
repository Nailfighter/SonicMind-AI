#!/usr/bin/env python3
"""
Modal GPU Training for EQ Neural Network
Scales to full FMA dataset with A100 GPU power
"""

import modal

# Create Modal app
app = modal.App("eq-training")

# Define the container image with all dependencies
image = modal.Image.debian_slim().pip_install([
    "torch>=2.0.0",
    "torchaudio", 
    "librosa>=0.10.0",
    "numpy",
    "scipy",
    "matplotlib",
    "tqdm",
    "soundfile",
    "requests"
])

# GPU configuration - A100 40GB for serious training
GPU_CONFIG = modal.gpu.A100(count=1)

@app.function(
    image=image,
    gpu=GPU_CONFIG,
    timeout=7200,  # 2 hours max
    memory=32*1024,  # 32GB RAM
)
def train_eq_model_gpu(
    dataset_url: str,
    max_files: int = 8000,
    batch_size: int = 512,
    epochs: int = 200,
    learning_rate: float = 0.005
):
    """Train EQ model on Modal A100 GPU"""
    
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import numpy as np
    import librosa
    import os
    import glob
    import random
    from scipy import signal
    from tqdm import tqdm
    import requests
    import tarfile
    import zipfile
    import time
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    # Classes (same as before but optimized)
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
                # Optimized feature extraction
                fft = np.fft.rfft(audio)
                magnitude = np.abs(fft)
                freq_bins = np.fft.rfftfreq(len(audio), 1/self.sr)
                
                # Energy in bands
                bass = np.sum(magnitude[(freq_bins >= 20) & (freq_bins <= 250)])
                mids = np.sum(magnitude[(freq_bins >= 250) & (freq_bins <= 4000)])
                highs = np.sum(magnitude[(freq_bins >= 4000) & (freq_bins <= 20000)])
                total = bass + mids + highs + 1e-8
                
                # Time domain features
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
    
    class EQDataset(Dataset):
        def __init__(self, audio_files, segment_length=44100*3):
            self.audio_files = audio_files
            self.segment_length = segment_length
            self.eq_processor = BiquadEQ()
            self.corruption_gen = EQCorruptionGenerator()
            self.feature_extractor = AudioFeatureExtractor()
            
            print(f"Dataset initialized with {len(audio_files)} files")
        
        def __len__(self):
            return len(self.audio_files) * 8  # More segments per file
        
        def __getitem__(self, idx):
            file_idx = idx // 8
            if file_idx >= len(self.audio_files):
                file_idx = file_idx % len(self.audio_files)
            
            audio_file = self.audio_files[file_idx]
            
            try:
                # Load audio with librosa
                audio, _ = librosa.load(audio_file, sr=44100, mono=True, duration=15)
                
                if len(audio) < 1000:
                    raise Exception("Audio too short")
                
                # Random segment
                if len(audio) > self.segment_length:
                    start = random.randint(0, len(audio) - self.segment_length)
                    audio = audio[start:start + self.segment_length]
                else:
                    audio = np.pad(audio, (0, self.segment_length - len(audio)))
                
                # Normalize
                if np.max(np.abs(audio)) > 1e-8:
                    audio = audio / np.max(np.abs(audio)) * 0.8
                
                # Generate and apply corruption
                corruption_eq = self.corruption_gen.generate_corruption_eq()
                corrupted_audio = self.eq_processor.apply_eq(audio, corruption_eq)
                
                # Extract features from corrupted audio
                features = self.feature_extractor.extract_features(corrupted_audio)
                
                # Target: inverse corruption
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
        def __init__(self, input_size=12):
            super().__init__()
            
            self.network = nn.Sequential(
                nn.Linear(input_size, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Dropout(0.3),
                
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Dropout(0.2),
                
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.BatchNorm1d(64),
                nn.Dropout(0.1),
                
                nn.Linear(64, 15)
            )
            
            self.register_buffer('freq_min', torch.tensor(20.0))
            self.register_buffer('freq_max', torch.tensor(20000.0))
            self.register_buffer('q_min', torch.tensor(0.3))
            self.register_buffer('q_max', torch.tensor(5.0))
            self.register_buffer('gain_min', torch.tensor(-15.0))
            self.register_buffer('gain_max', torch.tensor(15.0))
        
        def forward(self, features):
            raw_output = self.network(features)
            params = raw_output.view(-1, 5, 3)
            
            frequencies = self.freq_min + torch.sigmoid(params[:, :, 0]) * (self.freq_max - self.freq_min)
            q_factors = self.q_min + torch.sigmoid(params[:, :, 1]) * (self.q_max - self.q_min)
            gains = self.gain_min + torch.sigmoid(params[:, :, 2]) * (self.gain_max - self.gain_min)
            
            scaled = torch.stack([frequencies, q_factors, gains], dim=2)
            return scaled.view(-1, 15)
    
    # Download and prepare dataset
    print("Downloading FMA dataset...")
    os.makedirs("/tmp/fma_data", exist_ok=True)
    
    if dataset_url:
        print(f"Downloading from {dataset_url}")
        response = requests.get(dataset_url, stream=True)
        dataset_file = "/tmp/fma_data/fma_small.zip"

        with open(dataset_file, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)


        with zipfile.ZipFile(dataset_file, "r") as zip_ref:
            zip_ref.extractall("/tmp/fma_data/")

    
    # Find audio files
    audio_files = []
    for ext in ['*.mp3', '*.wav', '*.flac']:
        pattern = f"/tmp/fma_data/**/{ext}"
        audio_files.extend(glob.glob(pattern, recursive=True))
    
    if max_files and len(audio_files) > max_files:
        audio_files = random.sample(audio_files, max_files)
    
    print(f"Found {len(audio_files)} audio files")
    
    if len(audio_files) == 0:
        print("No audio files found!")
        return None
    
    # Create dataset and dataloader
    dataset = EQDataset(audio_files)
    dataset_size = len(dataset)
    train_size = int(0.85 * dataset_size)
    val_size = dataset_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Large batch sizes for A100
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}")
    
    # Initialize model
    model = EQNeuralNetwork(input_size=12).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Loss functions
    l1_loss = nn.L1Loss()
    
    # Training loop
    print(f"Starting training for {epochs} epochs...")
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        total_train_loss = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (features, target_eq, original_audio, corrupted_audio) in enumerate(train_pbar):
            features = features.to(device, non_blocking=True)
            target_eq = target_eq.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            predicted_eq = model(features)
            loss = l1_loss(predicted_eq, target_eq)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_train_loss += loss.item()
            train_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
            
            # Clear cache periodically
            if batch_idx % 100 == 0:
                torch.cuda.empty_cache()
        
        # Validation
        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for features, target_eq, _, _ in val_loader:
                features = features.to(device, non_blocking=True)
                target_eq = target_eq.to(device, non_blocking=True)
                
                predicted_eq = model(features)
                loss = l1_loss(predicted_eq, target_eq)
                total_val_loss += loss.item()
        
        # Compute averages
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        
        # Update scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.6f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'learning_rate': current_lr
            }
            
            # Save to /tmp first
            torch.save(checkpoint, '/tmp/best_eq_model.pth')
            print(f"New best model! Val loss: {avg_val_loss:.4f}")
    
    print(f"Training completed! Best validation loss: {best_val_loss:.4f}")
    
    # Return the model file for download
    with open('/tmp/best_eq_model.pth', 'rb') as f:
        model_data = f.read()
    
    return {
        'model_data': model_data,
        'final_train_loss': avg_train_loss,
        'best_val_loss': best_val_loss,
        'epochs_completed': epochs,
        'total_params': total_params
    }

# Local function to trigger the training
@app.local_entrypoint()
def main():
    """Run EQ training on Modal GPU with FMA-Small dataset"""
    
    print("Starting EQ training on Modal A100 with FMA-Small...")
    
    # FMA-Small dataset URL
    dataset_url = "https://os.unil.cloud.switch.ch/fma/fma_small.zip"
    
    # Launch training with optimized parameters for hackathon
    result = train_eq_model_gpu.remote(
        dataset_url=dataset_url,
        max_files=8000,       # Full FMA-Small dataset
        batch_size=1024,      # Large batch for A100
        epochs=50,            # Reduced epochs for faster training
        learning_rate=0.002   # Higher learning rate for faster convergence
    )
    
    if result:
        print(f"Training completed successfully!")
        print(f"Final train loss: {result['final_train_loss']:.4f}")
        print(f"Best validation loss: {result['best_val_loss']:.4f}")
        print(f"Model parameters: {result['total_params']:,}")
        print(f"Epochs completed: {result['epochs_completed']}")
        
        # Save the model locally
        with open('fma_eq_model.pth', 'wb') as f:
            f.write(result['model_data'])
        
        print("Model saved as 'fma_eq_model.pth'")
        print(f"Expected improvement over current model: {776.4350/result['best_val_loss']:.1f}x better")
    else:
        print("Training failed!")

if __name__ == "__main__":
    main()