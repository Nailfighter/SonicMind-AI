# SonicMind-AI Build Instructions

This guide explains how to build and distribute your Electron + Python application as a single Windows executable.

## Overview

Your project consists of:

- **Backend**: Python server with AI models (PyTorch, OpenCV, etc.)
- **Frontend**: Electron app with React UI
- **Communication**: Socket.IO between frontend and backend

## Prerequisites

### Required Software

- **Python 3.11+** (x64 or ARM64)
- **Node.js 18+** and npm
- **Git** (for cloning dependencies)
- **Windows 11** (ARM64 or x64)

### Python Dependencies

```bash
cd backend
pip install -r requirements.txt
pip install pyinstaller
```

### Node.js Dependencies

```bash
cd frontend
npm install
```

## Build Process

### Method 1: Automated Build (Recommended)

Run the complete build process with one command:

```bash
# From project root
build_all.bat
```

This will:

1. Build the Python backend executable
2. Build the Electron frontend
3. Package everything into a Windows installer

### Method 2: Step-by-Step Build

#### Step 1: Build Python Backend

```bash
# From project root
python build_backend.py
```

This creates: `backend/dist/SonicMind-Backend.exe`

#### Step 2: Build Electron Frontend

```bash
# From project root
build_frontend.bat
```

This creates: `frontend/dist/SonicMind-AI-1.0.0-setup.exe`

#### Step 3: Create Distribution Package

```bash
# From project root
python create_distribution.py
```

This creates a complete distribution package with all files.

### Method 3: Using npm Scripts

```bash
cd frontend

# Build backend only
npm run build:backend

# Build frontend only
npm run build:win

# Build everything
npm run build:full

# Create distribution package
npm run dist

# Clean build artifacts
npm run clean
```

## Output Files

### Backend Executable

- **Location**: `backend/dist/SonicMind-Backend.exe`
- **Size**: ~200-500MB (includes PyTorch, OpenCV, etc.)
- **Dependencies**: All Python libraries bundled

### Frontend Installer

- **Location**: `frontend/dist/SonicMind-AI-1.0.0-setup.exe`
- **Size**: ~100-200MB
- **Includes**: Electron app + Python backend executable

### Distribution Package

- **Location**: `SonicMind-AI-Distribution-YYYYMMDD_HHMMSS/`
- **Contents**:
  - `SonicMind-AI-Setup.exe` - Main installer
  - `SonicMind-Backend.exe` - Standalone backend
  - `models/fma_eq_model_npy.pth` - AI model file
  - `USER_GUIDE.txt` - User instructions
  - `SYSTEM_REQUIREMENTS.txt` - System requirements
  - `install.bat` - Installation script

## Configuration Files

### PyInstaller Configuration

- **File**: `backend/sonicmind_backend.spec`
- **Purpose**: Configures Python executable packaging
- **Key Settings**:
  - Includes all dependencies
  - Bundles model files
  - Console mode for debugging

### Electron Builder Configuration

- **File**: `frontend/electron-builder.yml`
- **Purpose**: Configures Electron app packaging
- **Key Settings**:
  - Includes Python backend executable
  - Windows NSIS installer
  - ARM64 and x64 support

### Installer Script

- **File**: `frontend/build/installer.nsh`
- **Purpose**: Custom installation steps
- **Features**:
  - Creates user data directories
  - Sets environment variables
  - Handles model file placement

## Troubleshooting

### Common Issues

#### Backend Build Fails

```bash
# Check Python installation
python --version

# Install missing dependencies
pip install pyinstaller

# Check model file exists
ls backend/fma_eq_model_npy.pth
```

#### Frontend Build Fails

```bash
# Check Node.js installation
node --version
npm --version

# Install dependencies
cd frontend
npm install

# Check backend executable exists
ls ../backend/dist/SonicMind-Backend.exe
```

#### Runtime Issues

- **Backend won't start**: Check Windows Defender/Antivirus
- **Audio issues**: Install ASIO drivers
- **Performance**: Close other audio applications

### Build Optimization

#### Reduce Backend Size

Edit `backend/sonicmind_backend.spec`:

```python
# Add to excludes list
excludes=['matplotlib', 'jupyter', 'notebook', 'pandas']
```

#### Reduce Frontend Size

Edit `frontend/electron-builder.yml`:

```yaml
# Add compression
compression: maximum
```

## Distribution

### Single File Distribution

The main installer (`SonicMind-AI-Setup.exe`) contains everything needed:

- Electron frontend
- Python backend executable
- AI model files
- All dependencies

### Portable Distribution

For portable installation:

1. Extract the distribution ZIP
2. Run `install.bat` as Administrator
3. Or manually copy files to desired location

### Update Mechanism

To update the application:

1. Build new version
2. Increment version in `package.json`
3. Run build process
4. Distribute new installer

## Advanced Configuration

### Custom Icons

Add to `backend/sonicmind_backend.spec`:

```python
icon='path/to/icon.ico'
```

Add to `frontend/electron-builder.yml`:

```yaml
win:
  icon: build/icon.ico
```

### Code Signing

Add to `frontend/electron-builder.yml`:

```yaml
win:
  certificateFile: path/to/certificate.p12
  certificatePassword: password
```

### Auto-Updates

Add to `frontend/electron-builder.yml`:

```yaml
publish:
  provider: github
  owner: your-username
  repo: SonicMind-AI
```

## Performance Notes

- **Backend**: ~200-500MB due to PyTorch and OpenCV
- **Frontend**: ~100-200MB for Electron runtime
- **Total**: ~300-700MB installed
- **Memory**: 2-4GB RAM usage during operation
- **CPU**: Optimized for ARM64 (Snapdragon X Elite)

## Support

For build issues:

1. Check this guide
2. Verify all prerequisites
3. Check error logs in build output
4. Ensure sufficient disk space (5GB+)

For runtime issues:

1. Check USER_GUIDE.txt
2. Verify system requirements
3. Check Windows Event Viewer
4. Run as Administrator if needed
