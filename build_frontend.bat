@echo off
REM Build script for SonicMind-AI Frontend (Electron)
REM This script builds the complete Windows executable

echo 🔧 Building SonicMind-AI Frontend...
echo ================================================

REM Check if Node.js is installed
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Node.js is not installed or not in PATH
    echo Please install Node.js from https://nodejs.org/
    pause
    exit /b 1
)

REM Check if backend executable exists
if not exist "backend\dist\SonicMind-Backend.exe" (
    echo ❌ Backend executable not found!
    echo Please run 'python build_backend.py' first
    pause
    exit /b 1
)

REM Navigate to frontend directory
cd frontend

REM Install dependencies
echo 📦 Installing frontend dependencies...
call npm install
if %errorlevel% neq 0 (
    echo ❌ Failed to install dependencies
    pause
    exit /b 1
)

REM Build the frontend
echo 🔨 Building Electron application...
call npm run build:win
if %errorlevel% neq 0 (
    echo ❌ Failed to build frontend
    pause
    exit /b 1
)

echo.
echo 🎉 Frontend build completed successfully!
echo.
echo 📁 Output files are in: frontend\dist\
echo 🚀 Run the installer to install SonicMind-AI
echo.
pause
