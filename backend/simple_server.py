#!/usr/bin/env python3
"""
ðŸŽ¯ Simple SonicMind AI Server - Guaranteed to Work
Fallback server with basic Socket.IO setup that works on all systems
"""

import time
from flask import Flask
import socketio
import eventlet
eventlet.monkey_patch()


# Create a simple Socket.IO server
sio = socketio.Server(cors_allowed_origins="*")
app = Flask(__name__)

# Global state
system_state = {
    "audio_running": False,
    "detection_running": False,
    "auto_eq_running": False,
    "current_instrument": "none",
    "room_preset": "neutral",
    "eq_bands": [
        {"freq": 80, "q": 1.0, "gain": 0.0},
        {"freq": 300, "q": 1.0, "gain": 0.0},
        {"freq": 1000, "q": 1.2, "gain": 0.0},
        {"freq": 4000, "q": 1.2, "gain": 0.0},
        {"freq": 10000, "q": 1.0, "gain": 0.0}
    ]
}

# ==================== SOCKET.IO EVENTS ==================== #


@sio.event
def connect(sid, environ):
    print(f"Client connected: {sid}")
    sio.emit('system_status', system_state, room=sid)


@sio.event
def disconnect(sid):
    print(f"Client disconnected: {sid}")


@sio.event
def start_audio(sid, data):
    print("Audio start requested")
    system_state["audio_running"] = True
    sio.emit('audio_started', {"success": True}, room=sid)
    sio.emit('system_status', system_state, room=sid)


@sio.event
def stop_audio(sid, data):
    print("Audio stop requested")
    system_state["audio_running"] = False
    sio.emit('audio_stopped', {"success": True}, room=sid)
    sio.emit('system_status', system_state, room=sid)


@sio.event
def start_detection(sid, data):
    print("Detection start requested")
    system_state["detection_running"] = True
    sio.emit('detection_started', {"success": True}, room=sid)

    # Mock room analysis after 3 seconds
    def mock_room_analysis():
        time.sleep(3)
        sio.emit('room_analysis', {
            "dominant_material": "wood",
            "material_confidence": 0.75,
            "acoustic_properties": {
                "room_type": "balanced_room",
                "preset_name": "balanced_room"
            },
            "eq_preset": {
                "description": "Balanced room with mixed acoustic properties",
                "eq_adjustments": {
                    "low": 0.0, "low_mid": 0.2, "mid": 0.0, "high_mid": 0.3, "high": 0.0
                }
            }
        }, room=sid)

    eventlet.spawn(mock_room_analysis)
    sio.emit('system_status', system_state, room=sid)


@sio.event
def start_auto_eq(sid, data):
    print("Auto-EQ start requested")
    system_state["auto_eq_running"] = True
    sio.emit('auto_eq_started', {"success": True}, room=sid)
    sio.emit('system_status', system_state, room=sid)


@sio.event
def manual_eq_update(sid, data):
    band_index = data.get('band_index', 0)
    parameter = data.get('parameter', 'gain_db')
    value = data.get('value', 0.0)

    if 0 <= band_index < len(system_state["eq_bands"]):
        if parameter == 'gain_db':
            system_state["eq_bands"][band_index]["gain"] = float(value)

        sio.emit('eq_updated', {
            "type": "manual_adjustment",
            "bands": system_state["eq_bands"],
            "timestamp": time.time()
        }, room=sid)
        sio.emit('manual_eq_updated', {"success": True}, room=sid)


@sio.event
def reset_eq(sid, data):
    for band in system_state["eq_bands"]:
        band["gain"] = 0.0

    sio.emit('eq_updated', {
        "type": "reset",
        "bands": system_state["eq_bands"],
        "timestamp": time.time()
    }, room=sid)
    sio.emit('eq_reset', {"success": True}, room=sid)


@sio.event
def get_system_status(sid, data):
    sio.emit('system_status', system_state, room=sid)

# ==================== FLASK ROUTES ==================== #


@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>SonicMind AI - Simple Server</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #1a1a2e; color: white; }
            .container { max-width: 800px; margin: 0 auto; }
            .status { background: #16213e; padding: 20px; border-radius: 10px; margin: 20px 0; }
            .connected { background: #0f4c75; }
            h1 { color: #4fc3f7; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>SonicMind AI - Simple Server</h1>
            <div class="status">
                <h3>Server Status</h3>
                <p>âœ… Simple Socket.IO server running</p>
                <p>ðŸ”Œ Endpoint: ws://localhost:8000/socket.io/</p>
                <p>ðŸ§ª Mock mode - for testing Socket.IO communication</p>
            </div>
            <div class="status" id="connection-status">
                <h3>Connection Status</h3>
                <p id="status-text">â³ Waiting for client...</p>
            </div>
            <div class="status">
                <h3>Test Commands</h3>
                <button onclick="testAudio()">Test Audio</button>
                <button onclick="testDetection()">Test Detection</button>
                <button onclick="testAutoEQ()">Test Auto-EQ</button>
                <button onclick="testManualEQ()">Test Manual EQ</button>
            </div>
        </div>
        
        <script src="https://cdn.socket.io/4.7.2/socket.io.min.js"></script>
        <script>
            const socket = io();
            const statusDiv = document.getElementById('connection-status');
            const statusText = document.getElementById('status-text');
            
            socket.on('connect', () => {
                statusDiv.className = 'status connected';
                statusText.innerHTML = 'âœ… Connected to backend!';
            });
            
            socket.on('disconnect', () => {
                statusDiv.className = 'status';
                statusText.innerHTML = 'âŒ Disconnected';
            });
            
            socket.on('system_status', (data) => {
                console.log('System Status:', data);
            });
            
            socket.on('room_analysis', (data) => {
                console.log('Room Analysis:', data);
                alert('Room analyzed: ' + data.dominant_material);
            });
            
            socket.on('eq_updated', (data) => {
                console.log('EQ Updated:', data);
            });
            
            function testAudio() {
                socket.emit('start_audio', {});
            }
            
            function testDetection() {
                socket.emit('start_detection', {camera_index: 0});
            }
            
            function testAutoEQ() {
                socket.emit('start_auto_eq', {});
            }
            
            function testManualEQ() {
                socket.emit('manual_eq_update', {
                    band_index: 2,
                    parameter: 'gain_db', 
                    value: 2.0
                });
            }
        </script>
    </body>
    </html>
    '''

# ==================== MAIN SERVER ==================== #


if __name__ == '__main__':
    print("ðŸŽ¯" + "="*50)
    print("    SonicMind AI - Simple Server")
    print("    Guaranteed Socket.IO Communication")
    print("="*52)
    print()
    print("ðŸŒ Server: http://localhost:8000")
    print("ðŸ”Œ Socket.IO: ws://localhost:8000/socket.io/")
    print("ðŸ§ª Mode: Mock responses for testing")
    print("ðŸ›‘ Press Ctrl+C to stop")
    print("="*52)

    try:
        # Wrap Flask app with Socket.IO
        app = socketio.WSGIApp(sio, app)

        # Start eventlet server
        eventlet.wsgi.server(eventlet.listen(('0.0.0.0', 8000)), app)

    except KeyboardInterrupt:
        print("\nðŸ›‘ Server stopped!")
    except Exception as e:
        print(f"âŒ Error: {e}")
        print("Try: pip install eventlet python-socketio flask")
