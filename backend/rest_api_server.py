#!/usr/bin/env python3
"""
🌐 SonicMind AI - REST API Server
Simple HTTP REST API for audio processing without Socket.IO
"""

import json
import time
import threading
from flask import Flask, request, jsonify
from flask_cors import CORS

from auto_eq_system import AutoEQSystem
from instrument_detection import InstrumentDetector
from material_detection import MaterialDetector

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

# Initialize systems
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
    "eq_bands": auto_eq.get_bands_dict(),
    "last_update": time.time()
}

# Store recent events for polling
recent_events = []
max_events = 50

def add_event(event_type, data):
    """Add event to recent events list"""
    event = {
        "type": event_type,
        "data": data,
        "timestamp": time.time()
    }
    recent_events.append(event)
    if len(recent_events) > max_events:
        recent_events.pop(0)

# ==================== SYSTEM STATUS ENDPOINTS ==================== #

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get current system status"""
    system_state["eq_bands"] = auto_eq.get_bands_dict()
    system_state["last_update"] = time.time()
    return jsonify(system_state)

@app.route('/api/devices', methods=['GET'])
def get_devices():
    """Get available audio devices"""
    devices = auto_eq.get_available_devices()
    return jsonify(devices)

@app.route('/api/events', methods=['GET'])
def get_recent_events():
    """Get recent events for polling-based updates"""
    since = request.args.get('since', type=float, default=0)
    filtered_events = [e for e in recent_events if e['timestamp'] > since]
    return jsonify({
        "events": filtered_events,
        "server_time": time.time()
    })

# ==================== AUDIO CONTROL ENDPOINTS ==================== #

@app.route('/api/audio/start', methods=['POST'])
def start_audio():
    """Start audio processing"""
    data = request.get_json() or {}
    input_device = data.get('input_device')
    output_device = data.get('output_device')
    
    success = auto_eq.start_audio(input_device, output_device)
    system_state["audio_running"] = success
    
    add_event('audio_started', {"success": success})
    
    return jsonify({
        "success": success,
        "message": "Audio started" if success else "Failed to start audio"
    })

@app.route('/api/audio/stop', methods=['POST'])
def stop_audio():
    """Stop audio processing"""
    auto_eq.stop_audio()
    system_state["audio_running"] = False
    
    add_event('audio_stopped', {"success": True})
    
    return jsonify({
        "success": True,
        "message": "Audio stopped"
    })

# ==================== AUTO-EQ ENDPOINTS ==================== #

@app.route('/api/auto-eq/start', methods=['POST'])
def start_auto_eq():
    """Start automatic EQ adjustments"""
    success = auto_eq.start_auto_eq()
    system_state["auto_eq_running"] = success
    
    if success:
        auto_eq.set_callback(on_eq_updated)
    
    add_event('auto_eq_started', {"success": success})
    
    return jsonify({
        "success": success,
        "message": "Auto-EQ started" if success else "Failed to start Auto-EQ"
    })

@app.route('/api/auto-eq/stop', methods=['POST'])
def stop_auto_eq():
    """Stop automatic EQ"""
    auto_eq.stop_auto_eq()
    system_state["auto_eq_running"] = False
    
    add_event('auto_eq_stopped', {"success": True})
    
    return jsonify({
        "success": True,
        "message": "Auto-EQ stopped"
    })

# ==================== MANUAL EQ ENDPOINTS ==================== #

@app.route('/api/eq/bands', methods=['GET'])
def get_eq_bands():
    """Get current EQ band settings"""
    return jsonify({
        "bands": auto_eq.get_bands_dict(),
        "timestamp": time.time()
    })

@app.route('/api/eq/bands/<int:band_index>', methods=['PUT'])
def update_eq_band(band_index):
    """Update individual EQ band"""
    data = request.get_json()
    parameter = data.get('parameter', 'gain_db')
    value = data.get('value', 0.0)
    
    success = auto_eq.update_band(band_index, parameter, value)
    
    if success:
        system_state["eq_bands"] = auto_eq.get_bands_dict()
        add_event('eq_updated', {
            "type": "manual_adjustment",
            "bands": system_state["eq_bands"],
            "band_index": band_index,
            "parameter": parameter,
            "value": value
        })
    
    return jsonify({
        "success": success,
        "message": "EQ band updated" if success else "Invalid parameters",
        "bands": auto_eq.get_bands_dict() if success else None
    })

@app.route('/api/eq/reset', methods=['POST'])
def reset_eq():
    """Reset all EQ bands to flat"""
    auto_eq.reset_eq()
    system_state["eq_bands"] = auto_eq.get_bands_dict()
    
    add_event('eq_updated', {
        "type": "reset",
        "bands": system_state["eq_bands"]
    })
    
    return jsonify({
        "success": True,
        "message": "EQ reset to flat",
        "bands": system_state["eq_bands"]
    })

# ==================== DETECTION ENDPOINTS ==================== #

@app.route('/api/detection/start', methods=['POST'])
def start_detection():
    """Start camera-based detection systems"""
    data = request.get_json() or {}
    camera_index = data.get('camera_index', 0)
    
    # Start instrument detection
    inst_success = instrument_detector.start_detection(camera_index)
    
    # Start material detection
    mat_success = material_detector.start_detection(camera_index)
    
    success = inst_success or mat_success
    system_state["detection_running"] = success
    
    if success:
        instrument_detector.set_callback(on_instrument_detected)
        material_detector.set_callback(on_material_analyzed)
    
    add_event('detection_started', {"success": success})
    
    return jsonify({
        "success": success,
        "message": "Detection started" if success else "Failed to start detection",
        "instrument_detection": inst_success,
        "material_detection": mat_success
    })

@app.route('/api/detection/stop', methods=['POST'])
def stop_detection():
    """Stop detection systems"""
    instrument_detector.stop_detection()
    material_detector.stop_detection()
    system_state["detection_running"] = False
    
    add_event('detection_stopped', {"success": True})
    
    return jsonify({
        "success": True,
        "message": "Detection stopped"
    })

# ==================== INSTRUMENT ENDPOINTS ==================== #

@app.route('/api/instrument/current', methods=['GET'])
def get_current_instrument():
    """Get currently detected instrument"""
    return jsonify({
        "instrument": system_state["current_instrument"],
        "timestamp": time.time()
    })

@app.route('/api/instrument/set', methods=['POST'])
def set_instrument():
    """Manually set current instrument"""
    data = request.get_json()
    instrument = data.get('instrument', 'none')
    
    auto_eq.set_current_instrument(instrument)
    system_state["current_instrument"] = instrument
    
    return jsonify({
        "success": True,
        "message": f"Instrument set to {instrument}",
        "instrument": instrument
    })

@app.route('/api/instrument/supported', methods=['GET'])
def get_supported_instruments():
    """Get list of supported instruments"""
    return jsonify({
        "instruments": instrument_detector.get_supported_instruments()
    })

# ==================== CALLBACK HANDLERS ==================== #

def on_instrument_detected(instrument, confidence):
    """Called when instrument is detected"""
    system_state["current_instrument"] = instrument
    
    # Update Auto-EQ with detected instrument
    if system_state["auto_eq_running"]:
        auto_eq.set_current_instrument(instrument)
    
    add_event('instrument_detected', {
        "instrument": instrument,
        "confidence": confidence,
        "timestamp": time.time()
    })
    
    print(f"🎵 Detected: {instrument} ({confidence:.1%})")

def on_material_analyzed(analysis):
    """Called when room material analysis is complete"""
    system_state["room_preset"] = analysis.get(
        "acoustic_properties", {}).get("preset_name", "neutral")
    
    # Apply room preset to EQ
    auto_eq.apply_room_preset(analysis)
    system_state["eq_bands"] = auto_eq.get_bands_dict()
    
    add_event('room_analysis', analysis)
    add_event('eq_updated', {
        "type": "room_preset",
        "bands": system_state["eq_bands"],
        "timestamp": time.time()
    })
    
    print(f"🏠 Room analyzed: {analysis.get('dominant_material', 'Unknown')}")

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
    
    add_event('eq_updated', eq_data)

# ==================== HEALTH CHECK ==================== #

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": time.time(),
        "systems": {
            "auto_eq": "available",
            "instrument_detection": "available" if instrument_detector.available else "mock",
            "material_detection": "available" if material_detector.available else "mock"
        }
    })

# ==================== ROOT ENDPOINT ==================== #

@app.route('/')
def index():
    """API documentation"""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>🎯 SonicMind AI - REST API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #1a1a2e; color: white; }
            .container { max-width: 1000px; margin: 0 auto; }
            .endpoint { background: #16213e; padding: 15px; margin: 10px 0; border-radius: 8px; }
            .method { background: #0f4c75; padding: 5px 10px; border-radius: 4px; margin-right: 10px; }
            .get { background: #28a745; }
            .post { background: #007bff; }
            .put { background: #ffc107; color: black; }
            h1 { color: #4fc3f7; }
            code { background: #2d3748; padding: 2px 6px; border-radius: 4px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎯 SonicMind AI - REST API</h1>
            
            <h2>📡 System Status</h2>
            <div class="endpoint">
                <span class="method get">GET</span><code>/api/status</code> - Get system status
            </div>
            <div class="endpoint">
                <span class="method get">GET</span><code>/api/devices</code> - Get audio devices
            </div>
            <div class="endpoint">
                <span class="method get">GET</span><code>/api/events?since=timestamp</code> - Get recent events
            </div>
            
            <h2>🎧 Audio Control</h2>
            <div class="endpoint">
                <span class="method post">POST</span><code>/api/audio/start</code> - Start audio processing
            </div>
            <div class="endpoint">
                <span class="method post">POST</span><code>/api/audio/stop</code> - Stop audio processing
            </div>
            
            <h2>🤖 Auto-EQ</h2>
            <div class="endpoint">
                <span class="method post">POST</span><code>/api/auto-eq/start</code> - Start auto-EQ
            </div>
            <div class="endpoint">
                <span class="method post">POST</span><code>/api/auto-eq/stop</code> - Stop auto-EQ
            </div>
            
            <h2>🎛️ Manual EQ</h2>
            <div class="endpoint">
                <span class="method get">GET</span><code>/api/eq/bands</code> - Get EQ bands
            </div>
            <div class="endpoint">
                <span class="method put">PUT</span><code>/api/eq/bands/{band_index}</code> - Update EQ band
            </div>
            <div class="endpoint">
                <span class="method post">POST</span><code>/api/eq/reset</code> - Reset EQ to flat
            </div>
            
            <h2>📷 Detection</h2>
            <div class="endpoint">
                <span class="method post">POST</span><code>/api/detection/start</code> - Start detection
            </div>
            <div class="endpoint">
                <span class="method post">POST</span><code>/api/detection/stop</code> - Stop detection
            </div>
            
            <h2>🎵 Instruments</h2>
            <div class="endpoint">
                <span class="method get">GET</span><code>/api/instrument/current</code> - Get current instrument
            </div>
            <div class="endpoint">
                <span class="method post">POST</span><code>/api/instrument/set</code> - Set instrument
            </div>
            <div class="endpoint">
                <span class="method get">GET</span><code>/api/instrument/supported</code> - List instruments
            </div>
            
            <p><strong>Base URL:</strong> http://localhost:8001</p>
            <p><strong>Content-Type:</strong> application/json</p>
            <p><strong>CORS:</strong> Enabled for all origins</p>
        </div>
        
        <script>
            // Simple test function
            function testAPI() {
                fetch('/api/status')
                    .then(response => response.json())
                    .then(data => console.log('Status:', data))
                    .catch(err => console.error('Error:', err));
            }
            
            // Auto-test on load
            testAPI();
        </script>
    </body>
    </html>
    '''

# ==================== CLEANUP ==================== #

def cleanup():
    """Cleanup resources"""
    auto_eq.cleanup()
    instrument_detector.cleanup()
    material_detector.cleanup()

if __name__ == '__main__':
    print("🌐" + "="*50)
    print("    SonicMind AI - REST API Server")
    print("    Standard HTTP endpoints for easy integration")
    print("="*52)
    print()
    print("🌐 REST API: http://localhost:8001")
    print("📖 Documentation: http://localhost:8001")
    print("🔧 Health check: http://localhost:8001/api/health")
    print("📡 Events polling: http://localhost:8001/api/events")
    print("🛑 Press Ctrl+C to stop")
    print("="*52)
    
    try:
        app.run(host='localhost', port=8001, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        cleanup()
        print("Server stopped!")
    except Exception as e:
        print(f"❌ Server error: {e}")
        cleanup()