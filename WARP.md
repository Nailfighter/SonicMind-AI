# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

**SonicMind-AI** is a Real Time Live Sound Engineering Assistant with a modular architecture featuring:
- **Backend**: Python-based modular auto-EQ system with Socket.IO API
- **Frontend**: Electron application with React frontend
- **Detection Systems**: CLIP-based instrument and material detection
- **Audio Processing**: Real-time EQ with AI enhancement and room acoustics analysis

The system has evolved from a monolithic GUI application to a distributed client-server architecture with separate detection modules.

## Architecture Overview

### Modular Backend System (`backend/`)

The backend uses a **clean modular architecture** where each component is a separate, loosely-coupled module:

#### Core Server (`main.py`)
- **Socket.IO Server**: Handles real-time communication with frontend clients
- **System Coordination**: Orchestrates all backend modules through callback pattern
- **Flask Web Server**: Provides basic web interface for debugging
- **Event-Driven**: Uses Socket.IO events for all client-server communication

#### Audio Processing (`auto_eq_system.py`)
- **Real-Time Audio**: Uses sounddevice for low-latency I/O
- **Parametric EQ**: 5-band EQ with biquad filters (SciPy implementation)
- **Dynamic Processing**: Digital compressor with envelope following
- **Audio Analysis**: FFT-based frequency analysis and feature extraction
- **Rule-Based EQ**: Traditional audio engineering rules for tonal balance
- **Neural Network Integration**: PyTorch model wrapper for AI-enhanced corrections

#### Detection Systems
- **Instrument Detection** (`instrument_detection.py`): CLIP-based zero-shot instrument recognition
- **Material Detection** (`material_detection.py`): Room acoustics analysis using CLIP for surface material identification

#### Key Architectural Patterns
- **Callback Architecture**: All modules use callbacks to communicate with main server
- **Thread-Safe Design**: Each module runs in separate threads with proper synchronization  
- **Graceful Fallbacks**: System continues to work even if optional dependencies fail
- **Modular Initialization**: Each component can be started/stopped independently

### Frontend System (`frontend/`)

- **Electron + React**: Cross-platform desktop application
- **Vite Build System**: Modern development tooling with HMR
- **TailwindCSS**: Utility-first styling framework
- **Socket.IO Client**: Real-time communication with backend

### Legacy Components

The `extra/` directory contains the original monolithic implementation and additional research:
- **Original GUI**: Complete standalone application (`src/gui.py` referenced in old WARP.md)
- **Training Pipeline**: Neural network training scripts for the AI model
- **Artist Detection**: Hackathon project with YOLO-based person detection

## Common Development Commands

### Backend Development

```bash
# Quick setup and run (Windows/Unix compatible)
cd backend
python setup_and_run.py

# Install dependencies only
python setup_and_run.py --install-only

# Start server only (skip setup)
python setup_and_run.py --server-only

# Run tests only
python setup_and_run.py --test-only

# Interactive testing mode
python setup_and_run.py --interactive

# Full test suite (modules + integration)  
python setup_and_run.py --full-test

# Manual server start
python main.py

# Individual module testing
python test_system.py --modules
python test_system.py --integration
python test_system.py --interactive
```

### Frontend Development

```bash
cd frontend

# Install dependencies
npm install

# Development mode with hot reload
npm run dev

# Build for production
npm run build

# Build for specific platforms
npm run build:win    # Windows
npm run build:mac    # macOS
npm run build:linux  # Linux

# Code quality
npm run lint
npm run format
```

### Full System Development

```bash
# Terminal 1: Backend
cd backend && python main.py

# Terminal 2: Frontend  
cd frontend && npm run dev

# Production build
cd frontend && npm run build:win
```

## Dependencies and Setup

### Backend Requirements

**Core Dependencies:**
- `python-socketio[client]>=5.8.0,<6.0.0` - Real-time communication
- `eventlet>=0.33.0` - WSGI server
- `flask>=2.3.0` - Web framework
- `numpy>=1.24.0` - Numerical operations
- `scipy>=1.10.0` - Digital signal processing
- `sounddevice>=0.4.5` - Real-time audio I/O

**AI/Vision Dependencies:**
- `torch>=2.0.0` - Neural network inference
- `torchvision>=0.15.0` - Vision utilities
- `opencv-python>=4.8.0` - Computer vision
- `pillow>=9.0.0` - Image processing
- `open-clip-torch>=2.20.0` - CLIP model (alternative to git+https://github.com/openai/CLIP.git)

**Installation:**
```bash
cd backend
pip install -r requirements.txt

# Try official CLIP first
pip install git+https://github.com/openai/CLIP.git

# If official CLIP fails, fallback is already in requirements.txt
```

### Frontend Requirements

**Core Dependencies:**
- `electron^37.2.3` - Desktop app framework  
- `react^19.1.0` - UI framework
- `vite^5.4.0` - Build tool
- `tailwindcss^4.0.0` - CSS framework
- `tone^15.1.22` - Audio synthesis (for UI feedback)

**Development Dependencies:**
- `electron-vite^4.0.0` - Electron + Vite integration
- `eslint^9.31.0` - Code linting
- `prettier^3.6.2` - Code formatting

## System Architecture Details

### Socket.IO Event System

The backend-frontend communication uses event-driven architecture:

**Audio Control Events:**
- `start_audio` / `stop_audio` - Audio system control
- `audio_started` / `audio_stopped` - Status responses

**Detection Control Events:**
- `start_detection` / `stop_detection` - Camera detection control
- `detection_started` / `detection_stopped` - Status responses

**EQ Control Events:**
- `start_auto_eq` / `stop_auto_eq` - Automatic EQ control
- `manual_eq_update` - Manual EQ band adjustments
- `reset_eq` - Reset all EQ to flat

**Real-time Data Events:**
- `system_status` - Overall system state
- `eq_updated` - EQ band changes (with update type)
- `instrument_detected` - Detected instrument with confidence
- `room_analysis` - Material detection results

### Audio Processing Pipeline

1. **Audio Input**: sounddevice captures real-time audio
2. **Analysis Engine**: Extracts frequency domain features every 5 seconds
3. **Detection Integration**: Instrument/room detection influences EQ decisions
4. **EQ Processing**: 5-band parametric EQ with biquad filters
5. **Dynamics Processing**: Optional compression
6. **Audio Output**: Real-time playback with <3ms latency

### Detection Systems Integration

**Instrument Detection Pipeline:**
- CLIP ViT-B/32 model with zero-shot classification
- Multiple text prompts per instrument category for improved accuracy
- Pre-computed text embeddings for efficiency
- Real-time camera processing at 1-second intervals

**Material Detection Pipeline:**
- CLIP-based surface material classification
- Acoustic property mapping for EQ preset generation
- Room analysis affects automatic EQ adjustments

## Development Patterns

### Error Handling
- **Graceful Degradation**: System continues working if optional components fail
- **Fallback Systems**: Mock detection when CLIP unavailable, CPU fallback when no GPU
- **User Feedback**: Clear error messages via Socket.IO events

### Threading Architecture
- **Main Thread**: Socket.IO event handling and coordination
- **Audio Thread**: Real-time audio processing (separate from analysis)
- **Detection Threads**: Camera processing and AI inference
- **Analysis Thread**: Audio feature extraction

### Configuration Management
- **Environment Variables**: Support for deployment configuration
- **Command Line Args**: Development and testing options
- **Device Discovery**: Automatic audio device enumeration
- **Camera Selection**: Support for multiple cameras

## Testing Strategy

### Module Testing (`test_system.py`)
- Individual component validation
- Mock data for missing hardware
- Component integration verification

### Integration Testing
- Full system startup/shutdown cycles
- Socket.IO communication validation
- Audio pipeline testing with virtual devices

### Interactive Testing
- Real-time system monitoring
- Manual component testing
- Performance profiling

## Platform Support

**Operating Systems:**
- Windows (PowerShell compatible)
- macOS
- Linux

**Hardware Requirements:**
- **Audio Interface**: Any sounddevice-compatible device
- **Camera**: Optional for detection features
- **GPU**: Optional (CUDA/MPS acceleration for AI models)

**Performance Characteristics:**
- **Audio Latency**: ~3ms (128 samples @ 44.1kHz)
- **Detection Speed**: ~75ms per frame (13 FPS)
- **Memory Usage**: ~150MB base + model weights
- **CPU Usage**: 10-20% on modern processors

## File Organization

**Backend Structure:**
- `main.py` - Server coordination and Socket.IO handling
- `auto_eq_system.py` - Audio processing and EQ engine
- `instrument_detection.py` - CLIP-based instrument recognition
- `material_detection.py` - Room acoustics analysis
- `test_system.py` - Comprehensive testing suite
- `setup_and_run.py` - One-command setup and deployment
- `simple_server.py` - Minimal server for basic testing
- `requirements.txt` - All Python dependencies

**Frontend Structure:**
- `src/main/` - Electron main process
- `src/preload/` - Electron preload scripts
- `electron.vite.config.mjs` - Build configuration
- `package.json` - Dependencies and scripts

**Legacy/Research:**
- `extra/WARP.md` - Original monolithic implementation guide
- `extra/Artist_Detection/` - YOLO-based person detection research
- `src/` - Original GUI application and training scripts

## Development Tips

### Backend Development
- Use `setup_and_run.py --interactive` for development - provides real-time system monitoring
- Mock detection systems automatically activate when cameras unavailable
- Audio devices are auto-discovered - no manual configuration needed
- Test individual modules with `test_system.py --modules`

### Frontend Development  
- Development server automatically proxies to backend Socket.IO
- Use browser dev tools for Socket.IO debugging
- Electron auto-restart is enabled in development mode

### Full System Development
- Backend must be running before frontend can connect
- Use separate terminals for backend/frontend development
- Integration tests require both systems running
- Monitor Socket.IO events for debugging communication issues
