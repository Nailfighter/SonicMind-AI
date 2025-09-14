# 🎤 Artist Detection WebSocket Server - Hackathon Startup Script
# Quick and easy server startup for Windows PowerShell

param(
    [string]$Host = "127.0.0.1",
    [int]$Port = 8765,
    [string]$CameraMode = "external"
)

Write-Host "🎤" -ForegroundColor Magenta -NoNewline
Write-Host "="*50 -ForegroundColor Cyan
Write-Host "🎤 ARTIST DETECTION WEBSOCKET SERVER STARTUP" -ForegroundColor Yellow
Write-Host "🎤" -ForegroundColor Magenta -NoNewline
Write-Host "="*50 -ForegroundColor Cyan

# Set environment variables
$env:SOCKET_HOST = $Host
$env:SOCKET_PORT = $Port
$env:CAMERA_MODE = $CameraMode

Write-Host "📋 Configuration:" -ForegroundColor Green
Write-Host "   🌐 Host: $Host" -ForegroundColor White
Write-Host "   🔌 Port: $Port" -ForegroundColor White
Write-Host "   📷 Camera Mode: $CameraMode" -ForegroundColor White
Write-Host ""

# Check if Python is available
Write-Host "🐍 Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "   ✅ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "   ❌ Python not found! Please install Python 3.8+" -ForegroundColor Red
    Write-Host "   💡 Download from: https://python.org/downloads" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if we're in the right directory
if (-not (Test-Path "socket_server.py")) {
    Write-Host "❌ socket_server.py not found!" -ForegroundColor Red
    Write-Host "💡 Please run this script from the Artist_Detection directory" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if virtual environment exists or create one
Write-Host "🔧 Setting up Python environment..." -ForegroundColor Yellow
if (-not (Test-Path "venv_socket")) {
    Write-Host "   📦 Creating virtual environment..." -ForegroundColor Cyan
    python -m venv venv_socket
    if ($LASTEXITCODE -ne 0) {
        Write-Host "   ❌ Failed to create virtual environment" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
    Write-Host "   ✅ Virtual environment created" -ForegroundColor Green
}

# Activate virtual environment
Write-Host "   🔄 Activating virtual environment..." -ForegroundColor Cyan
& ".\venv_socket\Scripts\Activate.ps1"
if ($LASTEXITCODE -ne 0) {
    Write-Host "   ❌ Failed to activate virtual environment" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Install requirements if needed
Write-Host "   📦 Checking dependencies..." -ForegroundColor Cyan
if (-not (Test-Path "venv_socket\.installed")) {
    Write-Host "   🔧 Installing socket server dependencies..." -ForegroundColor Yellow
    pip install -r socket_requirements.txt
    if ($LASTEXITCODE -eq 0) {
        New-Item -Path "venv_socket\.installed" -ItemType File | Out-Null
        Write-Host "   ✅ Dependencies installed successfully" -ForegroundColor Green
    } else {
        Write-Host "   ❌ Failed to install dependencies" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
} else {
    Write-Host "   ✅ Dependencies already installed" -ForegroundColor Green
}

Write-Host ""
Write-Host "🚀 Starting WebSocket Server..." -ForegroundColor Green
Write-Host "🔗 Server URL: ws://${Host}:${Port}" -ForegroundColor Yellow
Write-Host "📱 Test Client: file:///$PWD/test_socket_client.html" -ForegroundColor Yellow
Write-Host ""
Write-Host "💡 Tips for your hackathon:" -ForegroundColor Cyan
Write-Host "   • Use the test client to verify everything works" -ForegroundColor White
Write-Host "   • Connect your Electron app to ws://${Host}:${Port}" -ForegroundColor White
Write-Host "   • Send messages in JSON format with 'type' field" -ForegroundColor White
Write-Host "   • Press Ctrl+C to stop the server" -ForegroundColor White
Write-Host ""

try {
    # Start the server
    python socket_server.py
} catch {
    Write-Host "❌ Server startup failed: $_" -ForegroundColor Red
} finally {
    Write-Host ""
    Write-Host "👋 Server stopped. Good luck with your hackathon!" -ForegroundColor Yellow
}