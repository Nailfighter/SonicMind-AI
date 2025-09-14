#!/usr/bin/env python3
"""
ðŸŽ¯ SonicMind AI - Main Server Entry Point
Coordinates all systems and handles Socket.IO communication
"""

import socketio
import eventlet
from flask import Flask
import threading
import time
import json

from auto_eq_system import AutoEQSystem
from instrument_detection import InstrumentDetector
from material_detection import MaterialDetector

# Create Socket.IO server
sio = socketio.Server(cors_allowed_origins="*", logger=True)

# Initialize all systems
auto_eq = AutoEQSystem()
instrument_detector = InstrumentDetector()
material_detector = MaterialDetector()

# Global state
system_state = {
    "audio_running": False,
    "detection_running": False,
    "auto_eq_running": False,
    "current_instrument": "none",
    "room_preset": "neutral",
    "eq_bands": auto_eq.get_bands_dict()
}

# ==================== SOCKET.IO EVENTS ==================== #


@sio.event
def connect(sid, environ):
    """Client connected to server"""
    print(f"ðŸ”Œ Client connected: {sid}")
    sio.emit('system_status', system_state, room=sid)


@sio.event
def disconnect(sid):
    """Client disconnected from server"""
    print(f"ðŸ”Œ Client disconnected: {sid}")

# ==================== AUDIO CONTROL ==================== #


@sio.event
def start_audio(sid, data):
    """Start audio processing"""
    input_device = data.get('input_device')
    output_device = data.get('output_device')

    success = auto_eq.start_audio(input_device, output_device)
    system_state["audio_running"] = success

    sio.emit('audio_started', {"success": success}, room=sid)
    sio.emit('system_status', system_state, room=sid)
    print(f"ðŸŽ§ Audio {'started' if success else 'failed to start'}")


@sio.event
def stop_audio(sid, data):
    """Stop audio processing"""
    auto_eq.stop_audio()
    system_state["audio_running"] = False

    sio.emit('audio_stopped', {"success": True}, room=sid)
    sio.emit('system_status', system_state, room=sid)
    print("ðŸŽ§ Audio stopped")

# ==================== DETECTION CONTROL ==================== #


@sio.event
def start_detection(sid, data):
    """Start camera-based detection systems"""
    camera_index = data.get('camera_index', 0)

    # Start instrument detection
    inst_success = instrument_detector.start_detection(camera_index)

    # Start material detection (room analysis)
    mat_success = material_detector.start_detection(camera_index)

    success = inst_success or mat_success
    system_state["detection_running"] = success

    if success:
        # Set up callbacks for detection results
        instrument_detector.set_callback(on_instrument_detected)
        material_detector.set_callback(on_material_analyzed)

    sio.emit('detection_started', {"success": success}, room=sid)
    sio.emit('system_status', system_state, room=sid)
    print(f"ðŸ“· Detection {'started' if success else 'failed to start'}")


@sio.event
def stop_detection(sid, data):
    """Stop detection systems"""
    instrument_detector.stop_detection()
    material_detector.stop_detection()
    system_state["detection_running"] = False

    sio.emit('detection_stopped', {"success": True}, room=sid)
    sio.emit('system_status', system_state, room=sid)
    print("ðŸ“· Detection stopped")

# ==================== AUTO-EQ CONTROL ==================== #


@sio.event
def start_auto_eq(sid, data):
    """Start automatic EQ adjustments"""
    success = auto_eq.start_auto_eq()
    system_state["auto_eq_running"] = success

    if success:
        # Set up callback for EQ updates
        auto_eq.set_callback(on_eq_updated)

    sio.emit('auto_eq_started', {"success": success}, room=sid)
    sio.emit('system_status', system_state, room=sid)
    print(f"ðŸ¤– Auto-EQ {'started' if success else 'failed to start'}")


@sio.event
def stop_auto_eq(sid, data):
    """Stop automatic EQ"""
    auto_eq.stop_auto_eq()
    system_state["auto_eq_running"] = False

    sio.emit('auto_eq_stopped', {"success": True}, room=sid)
    sio.emit('system_status', system_state, room=sid)
    print("ðŸ¤– Auto-EQ stopped")

# ==================== MANUAL EQ CONTROL ==================== #


@sio.event
def manual_eq_update(sid, data):
    """Manual EQ band adjustment"""
    band_index = data.get('band_index', 0)
    parameter = data.get('parameter', 'gain_db')
    value = data.get('value', 0.0)

    success = auto_eq.update_band(band_index, parameter, value)

    if success:
        system_state["eq_bands"] = auto_eq.get_bands_dict()
        sio.emit('eq_updated', {
            "type": "manual_adjustment",
            "bands": system_state["eq_bands"],
            "timestamp": time.time()
        })
        sio.emit('manual_eq_updated', {"success": True}, room=sid)
    else:
        sio.emit('manual_eq_updated', {
                 "success": False, "error": "Invalid parameters"}, room=sid)


@sio.event
def reset_eq(sid, data):
    """Reset all EQ bands to flat"""
    auto_eq.reset_eq()
    system_state["eq_bands"] = auto_eq.get_bands_dict()

    sio.emit('eq_updated', {
        "type": "reset",
        "bands": system_state["eq_bands"],
        "timestamp": time.time()
    })
    sio.emit('eq_reset', {"success": True}, room=sid)
    print("ðŸŽ›ï¸ EQ reset to flat")

# ==================== SYSTEM INFO ==================== #


@sio.event
def get_system_status(sid, data):
    """Get current system status"""
    system_state["eq_bands"] = auto_eq.get_bands_dict()
    sio.emit('system_status', system_state, room=sid)


@sio.event
def get_available_devices(sid, data):
    """Get available audio devices"""
    devices = auto_eq.get_available_devices()
    sio.emit('available_devices', devices, room=sid)

# ==================== DETECTION CALLBACKS ==================== #


def on_instrument_detected(instrument, confidence):
    """Called when instrument is detected"""
    system_state["current_instrument"] = instrument

    # Send to all connected clients
    sio.emit('instrument_detected', {
        "instrument": instrument,
        "confidence": confidence,
        "timestamp": time.time()
    })

    # Update Auto-EQ with detected instrument
    if system_state["auto_eq_running"]:
        auto_eq.set_current_instrument(instrument)

    print(f"ðŸŽµ Detected: {instrument} ({confidence:.1%})")


def on_material_analyzed(analysis):
    """Called when room material analysis is complete"""
    system_state["room_preset"] = analysis.get(
        "acoustic_properties", {}).get("preset_name", "neutral")

    # Send to all connected clients
    sio.emit('room_analysis', analysis)

    # Apply room preset to EQ
    auto_eq.apply_room_preset(analysis)
    system_state["eq_bands"] = auto_eq.get_bands_dict()

    sio.emit('eq_updated', {
        "type": "room_preset",
        "bands": system_state["eq_bands"],
        "timestamp": time.time()
    })

    print(
        f"ðŸ  Room analyzed: {analysis.get('dominant_material', 'Unknown')}")


def on_eq_updated(bands, update_type="auto_adjustment", details=None):
    """Called when Auto-EQ updates bands"""
    system_state["eq_bands"] = bands

    eq_data = {
        "type": update_type,
        "bands": bands,
        "timestamp": time.time()
    }

    if details:
        eq_data["details"] = details

    sio.emit('eq_updated', eq_data)

# ==================== FLASK APP SETUP ==================== #


def create_app():
    """Create Flask application"""
    app = Flask(__name__)
    sio.init_app(app)

    @app.route('/')
    def index():
        return '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>ðŸŽ¯ SonicMind AI - Auto-EQ Backend</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; background: #1a1a2e; color: white; }
                .container { max-width: 800px; margin: 0 auto; }
                .status { background: #16213e; padding: 20px; border-radius: 10px; margin: 20px 0; }
                .connected { background: #0f4c75; }
                h1 { color: #4fc3f7; }
                .emoji { font-size: 1.2em; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1><span class="emoji">ðŸŽ¯</span> SonicMind AI Auto-EQ Backend</h1>
                <div class="status">
                    <h3>Server Status</h3>
                    <p><span class="emoji">âœ…</span> Socket.IO server is running</p>
                    <p><span class="emoji">ðŸ”Œ</span> Endpoint: <code>ws://localhost:8000/socket.io/</code></p>
                    <p><span class="emoji">ðŸŽ§</span> Audio System: Ready</p>
                    <p><span class="emoji">ðŸ“·</span> Detection Systems: Ready</p>
                    <p><span class="emoji">ðŸ¤–</span> Auto-EQ: Ready</p>
                </div>
                <div class="status" id="connection-status">
                    <h3>Connection Status</h3>
                    <p id="status-text"><span class="emoji">â³</span> Waiting for client connection...</p>
                </div>
            </div>
            
            <script src="https://cdn.socket.io/4.7.2/socket.io.min.js"></script>
            <script>
                const socket = io();
                const statusDiv = document.getElementById('connection-status');
                const statusText = document.getElementById('status-text');
                
                socket.on('connect', () => {
                    statusDiv.className = 'status connected';
                    statusText.innerHTML = '<span class="emoji">âœ…</span> Client connected successfully!';
                });
                
                socket.on('disconnect', () => {
                    statusDiv.className = 'status';
                    statusText.innerHTML = '<span class="emoji">âŒ</span> Client disconnected';
                });
                
                socket.on('system_status', (data) => {
                    console.log('System Status:', data);
                });
            </script>
        </body>
        </html>
        '''

    return app

# ==================== MAIN ENTRY POINT ==================== #


def main():
    print("SONICMIND AI" + "="*50)
    print("    SonicMind AI - Modular Auto-EQ Backend")
    print("    Clean Architecture with Separate Modules")
    print("="*62)
    print()
    print("System Components:")
    print("   â€¢ main.py - Server coordination & Socket.IO")
    print("   â€¢ auto_eq_system.py - Audio processing & EQ")
    print("   â€¢ instrument_detection.py - Real-time instrument detection")
    print("   â€¢ material_detection.py - Room acoustics analysis")
    print()
    print("Server starting on: http://localhost:8000")
    print("Socket.IO endpoint: ws://localhost:8000/socket.io/")
    print("Press Ctrl+C to stop")
    print("="*62)

    try:
        app = create_app()

        # Try eventlet first, fallback to development server
        try:
            import eventlet
            import eventlet.wsgi
            eventlet.wsgi.server(eventlet.listen(('0.0.0.0', 8000)), app)
        except ImportError:
            print("Eventlet not available, using development server")
            app.run(host='0.0.0.0', port=8000, debug=False)

    except KeyboardInterrupt:
        print("\nShutting down...")
        auto_eq.cleanup()
        instrument_detector.cleanup()
        material_detector.cleanup()
        print("Server stopped!")

    except Exception as e:
        print(f"Server error: {e}")
        print("Make sure all dependencies are installed:")
        print("pip install python-socketio eventlet flask numpy sounddevice")


if __name__ == "__main__":
    main()
