# SonicMind-AI

Real-time AI-powered sound engineering assistant with automatic EQ adjustment, instrument detection, and room acoustics analysis for Snapdragon X Elite.

## Architecture

### Backend (Python)

- `main.py` - Socket.IO server and system coordination
- `auto_eq_system.py` - Real-time audio processing and EQ management
- `instrument_detection.py` - CLIP-based instrument detection
- `material_detection.py` - Room acoustics analysis
- `test_system.py` - Comprehensive testing suite

### Frontend (Electron + React)

- `src/main/index.js` - Electron main process
- `src/renderer/src/App.jsx` - React application
- `src/renderer/src/components/` - UI components
- `src/renderer/src/services/` - Backend communication

## Technical Specifications

**Platform:** Snapdragon X Elite (ARM64)  
**OS:** Windows 11  
**Backend:** Python 3.11+, PyTorch 2.0+, OpenCV 4.8+  
**Frontend:** Node.js 18+, Electron 37.2.3, React 19.1.0  
**Audio:** sounddevice, scipy, numpy  
**AI Models:** CLIP, custom PyTorch models  
**Communication:** Socket.IO 5.8+, Flask 2.3+

## Setup Instructions

### Prerequisites

- Windows 11 (ARM64 or x64)
- Python 3.11+ (x86/ARM64)
- Node.js 18+ and npm
- Git
- Audio drivers
- Camera (optional)

### Backend Setup

```bash
git clone https://github.com/your-username/SonicMind-AI.git
cd SonicMind-AI/backend

# Create virtual environment
python -m venv sonicmind-venv
.\sonicmind-venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Run automated setup
python setup_and_run.py --install-only
```

### Frontend Setup

```bash
cd ../frontend
npm install
npm run build
```

## Usage Instructions

### Start Backend

```bash
cd backend
python main.py
```

Server: `http://localhost:8000`  
Socket.IO: `ws://localhost:8000/socket.io/`

### Start Frontend

```bash
cd frontend
npm run dev
```

### API Endpoints

- `GET /api/status` - System status
- `POST /api/audio/start` - Start audio processing
- `POST /api/audio/stop` - Stop audio processing
- `GET /api/devices` - Available audio devices
- `POST /api/eq/update` - Update EQ settings

### Socket.IO Events

- `start_audio` - Start real-time audio processing
- `start_detection` - Start instrument/room detection
- `start_auto_eq` - Enable automatic EQ adjustment
- `manual_eq_update` - Manual EQ band adjustment
- `get_system_status` - Get current system state

## Building Windows Executable

### Windows Executable (.exe)

```bash
# Backend
pip install pyinstaller
pyinstaller --onefile --name SonicMind-Backend --add-data "fma_eq_model_npy.pth;." main.py

# Frontend
cd frontend
npm run build:win
```

### Windows MSIX Package

```bash
cd frontend
npm run build:win -- --publish=always
```

### Distribution Package

Create complete Windows distribution:

```
SonicMind-AI-Distribution/
├── SonicMind-Backend.exe
├── SonicMind-AI-Setup.exe
├── models/
│   └── fma_eq_model_npy.pth
└── README.txt
```

## Testing Instructions

### Automated Testing

```bash
cd backend
python test_system.py --full-test
```

### Test Categories

- `--modules` - Individual module tests
- `--integration` - Integration tests
- `--interactive` - Interactive testing mode

### Performance Benchmarks

- Audio Latency: < 10ms
- CPU Usage: < 30%
- Memory Usage: < 2GB
- Detection Accuracy: > 85%

## Edge AI Implementation

### Local Processing

- 100% edge processing on Snapdragon X Elite
- No cloud dependency after setup
- ARM64 native optimization
- Real-time inference with PyTorch

### AI Models

- **EQ Optimization**: PyTorch neural network (~50MB)
- **Instrument Detection**: CLIP-based computer vision
- **Room Analysis**: Material detection and acoustics

### System Requirements

- Windows 11 (ARM64 or x64)
- RAM: 16GB minimum (32GB recommended)
- Storage: 5GB free space
- Audio: ASIO-compatible interface
- Camera: USB 2.0+ for computer vision

## License

MIT License - see [LICENSE](LICENSE) file for details.
