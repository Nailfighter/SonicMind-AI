#!/usr/bin/env python3
"""
Build script for SonicMind-AI Python backend
Creates a standalone executable using PyInstaller
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


def main():
    print("🔧 Building SonicMind-AI Backend...")
    print("=" * 50)

    # Get project root directory
    project_root = Path(__file__).parent.absolute()
    backend_dir = project_root / "backend"
    dist_dir = backend_dir / "dist"

    # Clean previous builds
    if dist_dir.exists():
        print("🧹 Cleaning previous build...")
        shutil.rmtree(dist_dir)

    # Ensure we're in the backend directory
    os.chdir(backend_dir)

    # Install PyInstaller if not already installed
    print("📦 Installing PyInstaller...")
    try:
        subprocess.run([sys.executable, "-m", "pip",
                       "install", "pyinstaller"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install PyInstaller: {e}")
        return False

    # Build the executable
    print("🔨 Building executable...")
    try:
        subprocess.run([
            sys.executable, "-m", "PyInstaller",
            "--clean",
            "--noconfirm",
            "sonicmind_backend.spec"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to build executable: {e}")
        return False

    # Verify the build
    exe_path = dist_dir / "SonicMind-Backend.exe"
    if exe_path.exists():
        print(f"✅ Backend executable created: {exe_path}")
        print(f"📊 File size: {exe_path.stat().st_size / (1024*1024):.1f} MB")
        return True
    else:
        print("❌ Executable not found after build")
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Backend build completed successfully!")
        print("Next step: Run 'npm run build:win' in the frontend directory")
    else:
        print("\n💥 Backend build failed!")
        sys.exit(1)
