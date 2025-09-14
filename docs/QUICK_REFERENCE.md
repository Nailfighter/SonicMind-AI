# 📚 Quick Reference

## 🚀 Quick Start Options

### Windows/Universal (Recommended)
```bash
python artist_setup.py                    # Auto-setup + start
python artist_setup.py --gui              # Launch GUI
python artist_setup.py --camera internal  # Internal camera
```

### Linux/Mac
```bash
./start_apis.sh                           # Start both APIs
CAMERA_MODE=internal ./start_apis.sh       # Internal camera mode
```

## 🧪 Testing Commands
```bash
# Interactive mode
python3 artist_test.py

# Test specific image
python3 artist_test.py path/to/image.jpg

# Test only instruments
python3 artist_test.py image.jpg --instrument

# Test only artists
python3 artist_test.py image.jpg --artist

# Check API health
python3 artist_test.py --health

# Find sample images
python3 artist_test.py --samples
```

## 📷 Camera Control
```bash
# Artist API with external camera (default)
python3 artist_run.py

# Artist API with internal camera
python3 artist_run.py --camera internal

# Custom port
python3 artist_run.py --port 5002

# Environment variable method
CAMERA_MODE=internal python3 artist_detection_api.py
```

## 🌐 API Endpoints

### Instrument Detection (Port 5000)
- **Web Interface**: http://localhost:5000
- **API**: `POST /detect` - Send base64 image
- **Health**: `GET /health`

### Artist Detection (Port 5001)  
- **Web Interface**: http://localhost:5001
- **API**: `POST /detect` - Send base64 image
- **Camera Info**: `GET /camera-info` - Get current camera mode
- **Health**: `GET /health` - Includes camera mode info

## 📁 Key Files
- `artist_setup.py` - Universal setup script (Windows ARM64 compatible)
- `artist_run.py` - Artist API runner with camera modes
- `artist_test.py` - Universal testing interface (GUI + Terminal)
- `artist_detection_api.py` - Artist API server  
- `instrument_detection_api.py` - Instrument API server
- `instrument_detector_zero_shot.py` - Core CLIP detector
- `artist_detection/` - YOLO person detection
- `start_apis.sh` - Linux/Mac startup script

## 🔧 Troubleshooting

### APIs won't start
1. Check dependencies: `pip install -r requirements.txt`
2. Check ports: `lsof -i :5000` and `lsof -i :5001`
3. Check logs in terminal output

### Import errors
1. Ensure you're in the hackathon directory
2. CLIP installation: `pip install git+https://github.com/openai/CLIP.git`
3. PyTorch installation may need specific version

### No sample images
- Sample images are in `training_artifacts/instrument_dataset/test/`
- Use your own images: JPG, PNG, GIF formats supported

## 🎯 Demo Tips
1. **Universal start**: Use `python artist_setup.py` for any platform
2. **Camera demo**: Show both external/internal camera modes
3. **Web interfaces**: Open http://localhost:5000 and http://localhost:5001
4. **Quick testing**: Use `python3 artist_test.py` for interactive testing
5. **Key selling points**:
   - Zero training needed for instruments (CLIP)
   - Camera mode flexibility for different setups
   - Windows ARM64 compatibility
   - No virtual environment complexity
