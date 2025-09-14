#!/usr/bin/env python3
"""
SonicMind AI - Artist Detection Setup Script
Vanilla PC setup for Windows ARM64 and other platforms
"""

import os
import sys
import subprocess
import platform
import time
import argparse

def print_banner():
    print("🎯" + "="*60)
    print("    SonicMind AI - Artist Detection Setup")
    print("    Real-time Live Sound Engineering Assistant")
    print("="*62)
    print()

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required. Current version:", sys.version)
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def detect_platform():
    """Detect the current platform and architecture"""
    system = platform.system()
    machine = platform.machine()
    
    print(f"🖥️  Platform: {system} {machine}")
    
    # Special handling for Windows ARM64
    if system == "Windows" and machine.lower() in ["arm64", "aarch64"]:
        print("🔧 Windows ARM64 detected - using compatible packages")
        return "windows_arm64"
    elif system == "Windows":
        return "windows"
    elif system == "Darwin":
        return "macos"
    else:
        return "linux"

def install_requirements(platform_type):
    """Install requirements with platform-specific handling"""
    print("\n📦 Installing dependencies...")
    
    # Base requirements that work everywhere
    base_packages = [
        "flask>=2.3.0",
        "flask-cors>=4.0.0", 
        "requests>=2.28.0",
        "numpy>=1.24.0",
        "pillow>=9.0.0",
        "opencv-python>=4.8.0",
        "pyyaml>=6.0.0",
        "tqdm>=4.64.0"
    ]
    
    # Platform-specific handling
    if platform_type == "windows_arm64":
        print("🔧 Windows ARM64: Installing PyTorch...")
        # For Windows ARM64, try pip first (since you said PyTorch is installed)
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "torch", "torchvision"], check=True)
            print("✅ PyTorch installed successfully")
        except subprocess.CalledProcessError:
            print("⚠️  PyTorch installation failed. Please install manually:")
            print("   pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
    else:
        # Standard PyTorch installation
        base_packages.extend([
            "torch>=2.0.0",
            "torchvision>=0.15.0"
        ])
    
    # Install CLIP
    print("🎵 Installing CLIP...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "git+https://github.com/openai/CLIP.git"], check=True)
        print("✅ CLIP installed successfully")
    except subprocess.CalledProcessError:
        print("⚠️  CLIP installation failed. Trying alternative...")
        subprocess.run([sys.executable, "-m", "pip", "install", "open-clip-torch"], check=False)
    
    # Install ultralytics for YOLO
    print("👤 Installing YOLO (ultralytics)...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "ultralytics==8.0.200"], check=True)
        print("✅ YOLO installed successfully")
    except subprocess.CalledProcessError:
        print("⚠️  YOLO installation failed, trying without version constraint...")
        subprocess.run([sys.executable, "-m", "pip", "install", "ultralytics"], check=False)
    
    # Install base packages
    print("📋 Installing base packages...")
    for package in base_packages:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", package], check=True, capture_output=True)
            print(f"✅ {package.split('>=')[0]}")
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Failed to install {package}: {e}")

def test_imports():
    """Test that key imports work"""
    print("\n🧪 Testing imports...")
    
    tests = [
        ("torch", "PyTorch"),
        ("cv2", "OpenCV"),
        ("flask", "Flask"),
        ("PIL", "Pillow"),
        ("requests", "Requests")
    ]
    
    failed = []
    for module, name in tests:
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} - import failed")
            failed.append(name)
    
    # Try CLIP import (might be clip or open_clip)
    try:
        import clip
        print("✅ CLIP")
    except ImportError:
        try:
            import open_clip
            print("✅ OpenCLIP (alternative)")
        except ImportError:
            print("❌ CLIP - import failed")
            failed.append("CLIP")
    
    # Try YOLO import
    try:
        from ultralytics import YOLO
        print("✅ YOLO (ultralytics)")
    except ImportError:
        print("❌ YOLO - import failed")
        failed.append("YOLO")
    
    if failed:
        print(f"\n⚠️  Some imports failed: {', '.join(failed)}")
        print("   The system may still work, but some features might be limited.")
    else:
        print("\n🎉 All imports successful!")
    
    return len(failed) == 0

def start_api_server(camera_mode="external"):
    """Start the API server"""
    print(f"\n🚀 Starting API server (camera: {camera_mode})...")
    
    # Check if artist_run.py exists, otherwise use direct API
    if os.path.exists("artist_run.py"):
        cmd = [sys.executable, "artist_run.py", "--camera", camera_mode]
    else:
        # Fallback to direct API start
        cmd = [sys.executable, "artist_detection_api.py"]
    
    print(f"💻 Command: {' '.join(cmd)}")
    print("🌐 API will be available at: http://localhost:5001")
    print("🛑 Press Ctrl+C to stop")
    print("-" * 50)
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n👋 API server stopped!")
    except FileNotFoundError as e:
        print(f"❌ Error starting API: {e}")
        print("   Make sure you're in the correct directory with the API files.")

def launch_gui():
    """Launch the GUI test interface"""
    print("\n🖥️  Launching GUI...")
    
    if os.path.exists("artist_test.py"):
        cmd = [sys.executable, "artist_test.py"]
        print(f"💻 Command: {' '.join(cmd)}")
        subprocess.run(cmd)
    else:
        print("❌ artist_test.py not found!")
        print("   Make sure you're in the correct directory.")

def main():
    parser = argparse.ArgumentParser(
        description="🎯 SonicMind AI Setup Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python artist_setup.py                    # Full setup + start API
  python artist_setup.py --skip-install    # Skip installation, just start
  python artist_setup.py --gui             # Launch GUI only
  python artist_setup.py --camera internal # Use internal camera
        """
    )
    
    parser.add_argument('--skip-install', action='store_true',
                       help='Skip package installation')
    parser.add_argument('--gui', action='store_true',
                       help='Launch GUI instead of starting API')
    parser.add_argument('--camera', choices=['external', 'internal'], 
                       default='external',
                       help='Camera mode: external (default) or internal')
    parser.add_argument('--test-only', action='store_true',
                       help='Only test imports, don\'t start anything')
    
    args = parser.parse_args()
    
    print_banner()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Detect platform
    platform_type = detect_platform()
    
    # Install requirements unless skipped
    if not args.skip_install:
        install_requirements(platform_type)
        
        # Test imports
        if not test_imports() and not args.test_only:
            print("\n⚠️  Some imports failed, but continuing anyway...")
            time.sleep(2)
    
    if args.test_only:
        test_imports()
        return
    
    # Launch GUI or start API
    if args.gui:
        launch_gui()
    else:
        start_api_server(args.camera)

if __name__ == "__main__":
    main()