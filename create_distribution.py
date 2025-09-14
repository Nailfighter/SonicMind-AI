#!/usr/bin/env python3
"""
Distribution packaging script for SonicMind-AI
Creates a complete distribution package with all necessary files
"""

import os
import shutil
import zipfile
from pathlib import Path
from datetime import datetime


def create_distribution():
    print("📦 Creating SonicMind-AI Distribution Package...")
    print("=" * 50)

    # Get project root directory
    project_root = Path(__file__).parent.absolute()

    # Define paths
    backend_exe = project_root / "backend" / "dist" / "SonicMind-Backend.exe"
    frontend_installer = project_root / "frontend" / \
        "dist" / "SonicMind-AI-1.0.0-setup.exe"
    model_file = project_root / "backend" / "fma_eq_model_npy.pth"

    # Create distribution directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dist_dir = project_root / f"SonicMind-AI-Distribution-{timestamp}"
    dist_dir.mkdir(exist_ok=True)

    print(f"📁 Creating distribution in: {dist_dir}")

    # Copy main files
    files_to_copy = [
        (backend_exe, "SonicMind-Backend.exe"),
        (frontend_installer, "SonicMind-AI-Setup.exe"),
        (model_file, "models/fma_eq_model_npy.pth"),
        (project_root / "README.md", "README.txt"),
        (project_root / "LICENSE", "LICENSE.txt")
    ]

    for src, dst in files_to_copy:
        if src.exists():
            dst_path = dist_dir / dst
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst_path)
            print(f"✅ Copied: {dst}")
        else:
            print(f"⚠️  Missing: {src}")

    # Create user guide
    create_user_guide(dist_dir)

    # Create system requirements file
    create_system_requirements(dist_dir)

    # Create installation script
    create_installation_script(dist_dir)

    # Create ZIP archive
    create_zip_archive(dist_dir, project_root)

    print(f"\n🎉 Distribution package created successfully!")
    print(f"📁 Location: {dist_dir}")
    print(f"📦 ZIP file: {dist_dir}.zip")

    return dist_dir


def create_user_guide(dist_dir):
    """Create a user guide for the distribution"""
    guide_content = """# SonicMind-AI User Guide

## Quick Start

1. **Install the Application**
   - Run `SonicMind-AI-Setup.exe` as Administrator
   - Follow the installation wizard
   - The application will be installed to your Program Files

2. **System Requirements**
   - Windows 11 (ARM64 or x64)
   - 16GB RAM minimum (32GB recommended)
   - 5GB free disk space
   - Audio interface (ASIO-compatible recommended)
   - Camera (optional, for instrument detection)

3. **First Launch**
   - Launch SonicMind-AI from Start Menu or Desktop
   - The backend will start automatically
   - Connect your audio interface
   - Select input/output devices in the app

## Features

- **Real-time Auto-EQ**: Automatically adjusts EQ based on detected instruments
- **Instrument Detection**: Uses computer vision to identify musical instruments
- **Room Analysis**: Analyzes room acoustics for optimal sound
- **Manual Controls**: Fine-tune EQ settings manually
- **Live Audio Processing**: Real-time audio processing with low latency

## Troubleshooting

### Backend Won't Start
- Check Windows Defender/Antivirus settings
- Ensure audio drivers are installed
- Run as Administrator if needed

### Audio Issues
- Check audio device selection
- Ensure ASIO drivers are installed
- Try different sample rates (44.1kHz, 48kHz)

### Performance Issues
- Close other audio applications
- Increase buffer size in audio settings
- Check system resources (RAM, CPU)

## Support

For technical support, please check the README.txt file or visit the project repository.

## License

This software is provided under the MIT License. See LICENSE.txt for details.
"""

    guide_path = dist_dir / "USER_GUIDE.txt"
    with open(guide_path, 'w', encoding='utf-8') as f:
        f.write(guide_content)
    print("✅ Created: USER_GUIDE.txt")


def create_system_requirements(dist_dir):
    """Create system requirements file"""
    requirements = """# SonicMind-AI System Requirements

## Minimum Requirements
- Operating System: Windows 11 (Build 22000 or later)
- Architecture: x64 or ARM64
- RAM: 16GB
- Storage: 5GB free space
- Audio: Built-in or external audio interface
- Camera: USB 2.0+ (optional)

## Recommended Requirements
- Operating System: Windows 11 (Latest)
- Architecture: ARM64 (Snapdragon X Elite optimized)
- RAM: 32GB
- Storage: 10GB free space (SSD recommended)
- Audio: ASIO-compatible audio interface
- Camera: USB 3.0+ for better detection

## Audio Interface Compatibility
- ASIO drivers (recommended)
- DirectSound (fallback)
- WASAPI (Windows 10/11)
- Core Audio (if available)

## Network Requirements
- No internet required for operation
- Local network only (Socket.IO communication)

## Performance Notes
- Real-time processing requires adequate CPU
- ARM64 builds are optimized for Snapdragon X Elite
- GPU acceleration not required but may help
"""

    req_path = dist_dir / "SYSTEM_REQUIREMENTS.txt"
    with open(req_path, 'w', encoding='utf-8') as f:
        f.write(requirements)
    print("✅ Created: SYSTEM_REQUIREMENTS.txt")


def create_installation_script(dist_dir):
    """Create installation script"""
    install_script = """@echo off
REM SonicMind-AI Installation Script
REM Run this script to install SonicMind-AI

echo 🎵 SonicMind-AI Installation Script
echo ====================================
echo.

REM Check if running as administrator
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ This script must be run as Administrator
    echo Right-click and select "Run as administrator"
    pause
    exit /b 1
)

echo ✅ Running as Administrator
echo.

REM Install the main application
echo 🔧 Installing SonicMind-AI...
if exist "SonicMind-AI-Setup.exe" (
    start /wait SonicMind-AI-Setup.exe
    echo ✅ Installation completed
) else (
    echo ❌ Installer not found: SonicMind-AI-Setup.exe
    pause
    exit /b 1
)

echo.
echo 🎉 Installation completed successfully!
echo.
echo 📝 Next steps:
echo    1. Launch SonicMind-AI from Start Menu
echo    2. Connect your audio interface
echo    3. Select input/output devices
echo    4. Start making music!
echo.
pause
"""

    script_path = dist_dir / "install.bat"
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(install_script)
    print("✅ Created: install.bat")


def create_zip_archive(dist_dir, project_root):
    """Create a ZIP archive of the distribution"""
    zip_path = project_root / f"{dist_dir.name}.zip"

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in dist_dir.rglob('*'):
            if file_path.is_file():
                arcname = file_path.relative_to(dist_dir)
                zipf.write(file_path, arcname)

    print(f"✅ Created ZIP archive: {zip_path.name}")


if __name__ == "__main__":
    create_distribution()
