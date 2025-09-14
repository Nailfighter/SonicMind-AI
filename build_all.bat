@echo off
REM Complete build script for SonicMind-AI
REM Builds both backend and frontend into a single Windows executable

echo 🎵 SonicMind-AI Complete Build Script
echo =====================================
echo.

REM Check Python installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.11+ from https://python.org/
    pause
    exit /b 1
)

REM Check Node.js installation
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Node.js is not installed or not in PATH
    echo Please install Node.js from https://nodejs.org/
    pause
    exit /b 1
)

echo ✅ Prerequisites check passed
echo.

REM Step 1: Build Python backend
echo 🔧 Step 1: Building Python Backend...
echo ----------------------------------------
python build_backend.py
if %errorlevel% neq 0 (
    echo ❌ Backend build failed!
    pause
    exit /b 1
)

echo.
echo ✅ Backend build completed
echo.

REM Step 2: Build Electron frontend
echo 🔧 Step 2: Building Electron Frontend...
echo -----------------------------------------
call build_frontend.bat
if %errorlevel% neq 0 (
    echo ❌ Frontend build failed!
    pause
    exit /b 1
)

echo.
echo 🎉 Complete build finished successfully!
echo.
echo 📁 Output files:
echo    - Frontend installer: frontend\dist\SonicMind-AI-1.0.0-setup.exe
echo    - Backend executable: backend\dist\SonicMind-Backend.exe
echo.
echo 🚀 You can now distribute the installer to users!
echo.
pause
