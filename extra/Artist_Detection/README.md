# 🎯 Hackathon AI Detection Suite

**Dual AI System: Artist Detection + Instrument Recognition**

## 🚀 Quick Start (< 30 seconds!)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the APIs (in separate terminals)
python3 instrument_detection_api.py  # Port 5000
python3 artist_detection_api.py      # Port 5001

# 3. Test both systems
python3 test.py  # Interactive GUI mode
# OR
python3 test.py path/to/image.jpg  # Terminal mode
```

## 🎯 Project Overview

We built a **dual AI detection system** for hackathons:
1. **🎸 Instrument Detection**: Recognizes musical instruments with 63% accuracy using zero-shot CLIP
2. **👤 Artist Detection**: Detects people/musicians using fine-tuned YOLO models

### The Innovation 💡
- **Zero-shot CLIP**: No training needed - leverages pre-trained knowledge from 400M images
- **Smart prompting**: Multiple text prompts per instrument boost accuracy
- **Dual APIs**: Independent services that can run together or separately
- **Universal testing**: Single test interface for both detection systems

## 📊 Performance

| Metric | Value |
|--------|-------|
| **Accuracy** | 63.3% (up to 100% on some instruments) |
| **Speed** | 75ms per image (13 FPS) |
| **Training Time** | 0 seconds! |
| **Model Size** | ~340MB |
| **Hardware** | Runs on CPU/ARM |

### Per-Instrument Accuracy
- 🎸 **Banjo**: 100%
- 🎹 **Harmonica**: 100%
- 🎷 **Saxophone**: 100%
- 🪗 **Accordion**: 100%
- 🥁 **Tabla**: 67%
- 🎻 **Violin**: 33%

## 🛠️ Technical Approach

### Why Zero-Shot CLIP?
1. **Pre-trained Knowledge**: CLIP was trained on 400M image-text pairs
2. **No Training Needed**: It already knows what instruments look like
3. **Prompt Engineering**: Multiple text prompts per instrument for better accuracy
4. **Fast Inference**: Runs in real-time on CPU

### Key Innovation: Smart Prompting
Instead of just "guitar", we use:
- "a photo of a guitar"
- "someone playing guitar"
- "a musician with a guitar"
- "guitar being played"
- "guitar in a concert"

Then average the scores → Much better accuracy!

## 🏗️ Architecture

```
Input Image → CLIP Image Encoder → Image Features
                                         ↓
                                    Similarity Score → Top Predictions
                                         ↑
Text Prompts → CLIP Text Encoder → Text Features (cached)
```

## 📁 Clean Project Structure

```
hackathon/
├── 🛠️ artist_setup.py                   # Universal setup script (Windows ARM64)
├── 🎤 artist_run.py                     # Artist API runner with camera modes
├── 🧪 artist_test.py                    # Universal test interface (GUI + Terminal)
├── 👤 artist_detection_api.py          # Artist API server (port 5001)
├── 🎸 instrument_detection_api.py      # Instrument API server (port 5000)
├── 🎵 instrument_detector_zero_shot.py # Zero-shot CLIP detector
├── 📂 artist_detection/               # YOLO-based person detection
├── 🚀 start_apis.sh                    # Linux/Mac startup script
├── 📝 requirements.txt                 # All dependencies (no venv needed)
├── 🚫 .gitignore                      # Excludes training artifacts
└── 📦 training_artifacts/             # All training files (gitignored)
    ├── datasets/                      # Large training datasets  
    ├── modal_scripts/                 # Training scripts
    └── deprecated/                    # Old code
```

## 🌟 Features

### 🎸 Instrument Detection API (Port 5000)
- Beautiful drag & drop web interface
- Zero-shot CLIP detection (no training needed)
- 63% accuracy with smart prompting
- API: `POST /detect`, `GET /`, `GET /health`

### 👤 Artist Detection API (Port 5001)  
- YOLO-based person detection
- **Camera modes**: External (artists) or Internal (sound engineer)
- Bounding box coordinates for detected people
- Configurable via environment or command line
- API: `POST /detect`, `GET /`, `GET /health`, `GET /camera-info`

### 🧪 Universal Testing Interface (`artist_test.py`)
- **Interactive GUI**: `python artist_test.py` - menu-driven testing
- **Terminal mode**: `python artist_test.py image.jpg` - direct testing
- **Health checks**: `python artist_test.py --health` - verify APIs
- **Flexible**: Test one or both detection systems

### 🛠️ Setup & Deployment (`artist_setup.py`)
- **Windows ARM64 compatible** - auto-detects platform
- **One-command setup** - installs dependencies automatically
- **Camera configuration** - external (default) or internal
- **GUI launch option** - `--gui` flag for testing interface

## 💡 Lessons Learned

1. **Don't overthink it!** - We wasted hours on complex training when the simple solution worked better
2. **Pre-trained models are powerful** - CLIP already knows most things
3. **Prompt engineering > Fine-tuning** (for hackathons)
4. **Test the simplest approach first**

## 🚄 Why This Wins Hackathons

1. **Dual System**: Two AI capabilities in one clean package
2. **Zero Training**: Instrument detection works immediately with CLIP
3. **Production Ready**: Clean APIs, proper structure, comprehensive testing
4. **Easy Demo**: Single command setup, beautiful web interfaces
5. **Extensible**: Clean architecture makes it easy to add more detection types
6. **Professional**: Proper .gitignore, requirements, documentation

## 🔮 Future Improvements (Post-Hackathon)

1. Add more instrument categories
2. Implement confidence thresholds
3. Add batch processing
4. Create mobile app
5. Add real-time video stream support

## 📷 Camera Configuration

**Two-Camera System (Single Camera at a Time)**

### 🎥 External Camera (Default)
- **Purpose**: Faces the artists/performers
- **Use case**: Artist detection and instrument recognition
- **Default mode**: All scripts use this unless specified

### 📹 Internal Camera  
- **Purpose**: Faces the sound engineer
- **Use case**: Engineering workspace monitoring
- **Configuration**: Set via `--camera internal` flag

### ⚙️ Setup Examples
```bash
# External camera (default)
python artist_run.py
CAMERA_MODE=external ./start_apis.sh

# Internal camera
python artist_run.py --camera internal
CAMERA_MODE=internal ./start_apis.sh
python artist_setup.py --camera internal
```

## 🍥 Demo Commands

```bash
# Option 1: Windows/Universal setup (recommended)
python artist_setup.py                    # Auto-installs & starts APIs
python artist_setup.py --gui              # Launch GUI instead
python artist_setup.py --camera internal  # Internal camera mode

# Option 2: Quick start (Linux/Mac)
./start_apis.sh  # Starts both APIs automatically

# Option 3: Individual APIs
python artist_run.py                       # Artist detection only
python artist_run.py --camera internal     # Internal camera (sound engineer)
python instrument_detection_api.py         # Instrument detection only

# Testing
python artist_test.py                      # Interactive GUI mode
python artist_test.py image.jpg            # Test both detections
python artist_test.py --health             # Check API status
```

## 📈 The Journey

**Hour 1-4**: Set up Modal, uploaded datasets, tried to fine-tune CLIP  
**Hour 5-6**: Training kept producing NaN losses, 10% accuracy  
**Hour 7**: Realized CLIP already knows instruments!  
**Hour 7.5**: Switched to zero-shot, got 63% accuracy immediately  
**Hour 8**: Built web interface and demo  

## 🏆 Key Takeaways

> "The best solution is often the simplest one"

We learned that sometimes you don't need to train a model - you just need to use it correctly!

## 👥 Team

Built with ❤️ during the hackathon

---

**Ready for demo!** Just run `python3 serve_instrument_api.py` and open http://localhost:5000 🎸