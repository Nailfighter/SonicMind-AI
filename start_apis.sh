#!/bin/bash
# Quick Start Script for Hackathon AI Detection Suite

echo "🎯 Starting Hackathon AI Detection Suite..."
echo "========================================"

# Check if requirements are installed
if ! python3 -c "import torch, clip, cv2, flask" 2>/dev/null; then
    echo "⚠️  Installing requirements..."
    pip install -r requirements.txt
fi

echo ""
echo "🚀 Starting APIs in background..."

# Start instrument detection API
echo "🎸 Starting Instrument Detection API (port 5000)..."
python3 instrument_detection_api.py &
INSTRUMENT_PID=$!

# Wait a moment
sleep 2

# Start artist detection API  
echo "👤 Starting Artist Detection API (port 5001)..."
# Default to external camera unless specified
export CAMERA_MODE=${CAMERA_MODE:-external}
export API_PORT=5001
python3 artist_detection_api.py &
ARTIST_PID=$!

# Wait for APIs to start
sleep 5

echo ""
echo "✅ APIs Started Successfully!"
echo ""
echo "🌐 Web Interfaces:"
echo "   Instrument Detection: http://localhost:5000"
echo "   Artist Detection:     http://localhost:5001"
echo ""
echo "🧪 Testing:"
echo "   python3 test.py                  # Interactive mode"
echo "   python3 test.py image.jpg        # Test with image"
echo "   python3 test.py --health         # Check API health"
echo ""
echo "🛑 To stop APIs:"
echo "   kill $INSTRUMENT_PID $ARTIST_PID"
echo "   or press Ctrl+C"

# Store PIDs for cleanup
echo "$INSTRUMENT_PID $ARTIST_PID" > .api_pids

# Wait for user interrupt
trap "kill $INSTRUMENT_PID $ARTIST_PID; rm -f .api_pids; echo '👋 APIs stopped!'; exit" INT
wait