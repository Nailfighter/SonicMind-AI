# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

**SonicMind-AI** is a Real Time Live Sound Engineering Assistant that combines rule-based and AI-powered automatic EQ correction for live audio processing. The system analyzes incoming audio in real-time and applies intelligent EQ adjustments to improve sound quality.

## Core Architecture

### Audio Processing Pipeline
The system uses a hybrid approach combining:
- **Rule-based EQ (70%)**: Traditional audio engineering rules targeting specific frequency balance (bass: 27%, mids: 53%, highs: 20%)
- **Neural Model (30%)**: Deep learning model trained on professional music datasets to predict EQ corrections
- **Real-time Processing**: Zero-latency audio path with optional effects (delay/reverb add latency when enabled)

### Key Components

#### Main Application (`src/gui.py`)
- **LiveAutoEQApp**: Primary Tkinter-based GUI application
- **AudioEngine**: Real-time audio processing using sounddevice
- **ParametricEQ**: 5-band peaking EQ using SciPy biquad filters (FFT fallback)
- **Dynamics**: Zero-lookahead compressor for dynamics processing
- **FX**: Delay, reverb, and saturation effects
- **RuleEQ**: Implements audio engineering rules for tonal balance
- **ModelWrapper**: Handles PyTorch neural model loading and inference

#### Training System
- **model_trainer.py**: Local training with FMA-small dataset
- **modal_trainer.py**: Cloud training using Modal platform with A100 GPUs
- **make_npy.py**: Preprocessing script to generate training features and targets
- **make_npy2.py**: Alternative preprocessing approach

#### Neural Architecture
The AI model is a deep feedforward network:
- Input: 12 audio features (spectral bands, dynamics, temporal characteristics)
- Architecture: 12→512→256→128→64→32→16→8→15
- Output: 15 parameters (5 EQ bands × 3 params: frequency, Q, gain)
- Training: Self-supervised on corrupted/clean audio pairs

## Common Development Commands

### Running the Application
```bash
# Run the main GUI application
python src/gui.py

# Run with custom model path
python src/gui.py path/to/model.pth
```

### Training and Data Preparation
```bash
# Generate training data from FMA-small dataset
python src/make_npy.py

# Train model locally
python src/model_trainer.py

# Train on Modal cloud platform (requires Modal setup)
modal run src/modal_trainer.py

# Test preprocessing output
python src/test.py
```

### Model Testing
```bash
# Test model inference
python src/model_tester.py
```

## Dependencies and Setup

### Core Dependencies
- **PyTorch**: Neural network training and inference
- **NumPy**: Numerical operations and audio processing  
- **SciPy**: Biquad filter implementations for EQ
- **sounddevice**: Real-time audio I/O
- **tkinter**: GUI framework (usually included with Python)
- **librosa**: Audio file loading and preprocessing
- **tqdm**: Progress bars for training

### Optional Dependencies
- **Modal**: Cloud training platform for GPU acceleration
- **CUDA/MPS**: GPU acceleration for neural network operations

### Installation
```bash
# Install core dependencies
pip install torch numpy scipy sounddevice librosa tqdm

# For Modal cloud training
pip install modal
```

## Audio Processing Details

### EQ Band Configuration
Default 5-band parametric EQ:
- Band 1: 80 Hz (bass cleanup)
- Band 2: 300 Hz (low-mids)  
- Band 3: 1000 Hz (presence)
- Band 4: 4000 Hz (clarity)
- Band 5: 10000 Hz (air/brightness)

### Signal Flow
1. **Input**: Mono audio via sounddevice
2. **Analysis**: 5-second sliding window for feature extraction
3. **EQ Prediction**: Rule-based + AI model blend (70:30 ratio)
4. **Processing Chain**: EQ → Drive → Compressor → [Delay/Reverb]
5. **Output**: Processed audio with adjustable trim gain

### Training Data Generation
- Uses FMA-small dataset (8,000 30-second tracks)
- Applies realistic "bad" EQ corruptions (muddy bass, harsh mids, etc.)
- Neural network learns to predict inverse corrections
- Self-supervised learning approach (no manual labeling)

## File Organization

- `src/gui.py`: Main application with GUI and real-time processing
- `src/model_trainer.py`: Local training pipeline
- `src/modal_trainer.py`: Cloud training with Modal platform
- `src/make_npy.py`: Data preprocessing for training
- `src/fma_eq_model_npy.pth`: Pre-trained model weights
- `precomputed_features/`: Cached training data (features.npy, targets.npy)

## Development Notes

### Audio Engine Considerations
- Default 128-sample blocksize for low latency
- Real-time EQ updates every 1 second after 5-second warmup
- Stability detection prevents excessive corrections on tonal content
- Thread-safe ring buffer for analysis without affecting playback latency

### Model Training Workflow
1. Download FMA-small dataset to `./fma_small/`
2. Run `make_npy.py` to precompute features (speeds up training)
3. Use `model_trainer.py` for local development or `modal_trainer.py` for production training
4. Trained model loads automatically as `src/fma_eq_model_npy.pth`

### Platform Support
- Cross-platform Python application
- GPU acceleration: CUDA (NVIDIA), MPS (Apple Silicon), CPU fallback
- Audio I/O via sounddevice (PortAudio backend)

### Performance Characteristics
- Real-time processing: ~3ms latency (128 samples @ 44.1kHz)
- AI inference: ~1-2ms per prediction
- Memory usage: ~100MB base + model weights (~50MB)
- CPU usage: 5-15% on modern processors