#!/usr/bin/env python3
"""
Artist Detection API Server
Detects people/artists in images using YOLO
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import base64
import io
import cv2
import numpy as np
from PIL import Image
import os
import time
import sys
import torch

# Add artist_detection to path
sys.path.append('artist_detection')
from artist_detection.detection.artist_detector import detect_artists

# Camera configuration
CAMERA_MODE = os.environ.get('CAMERA_MODE', 'external')  # external (default) or internal
API_PORT = int(os.environ.get('API_PORT', '5001'))

app = Flask(__name__)
CORS(app)

# Simple HTML interface
DEMO_HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>👤 Artist Detector - Hackathon Demo</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%); min-height: 100vh; }
        .container { background: white; border-radius: 20px; padding: 30px; box-shadow: 0 20px 60px rgba(0,0,0,0.3); }
        h1 { text-align: center; color: #333; margin-bottom: 10px; }
        .subtitle { text-align: center; color: #666; margin-bottom: 30px; }
        .upload-area { border: 3px dashed #e74c3c; border-radius: 10px; padding: 40px; text-align: center; cursor: pointer; transition: all 0.3s; background: #fff5f5; }
        .upload-area:hover { border-color: #c0392b; background: #fff0f0; }
        #preview { max-width: 100%; max-height: 400px; margin: 20px auto; display: none; border-radius: 10px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); }
        .results { margin-top: 30px; display: none; }
        .result-item { background: linear-gradient(90deg, #e74c3c 0%, #c0392b 100%); color: white; padding: 15px; margin: 10px 0; border-radius: 10px; animation: slideIn 0.3s ease; }
        @keyframes slideIn { from { transform: translateX(-20px); opacity: 0; } to { transform: translateX(0); opacity: 1; } }
        .loading { display: none; text-align: center; margin: 20px 0; }
        .spinner { border: 4px solid #f3f3f3; border-top: 4px solid #e74c3c; border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite; margin: 0 auto; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <div class="container">
        <h1>👤 Artist Detector</h1>
        <p class="subtitle">AI-Powered Person Detection in Images</p>
        <p style="text-align: center; color: #666; font-size: 14px; margin-bottom: 20px;">
            📷 Camera Mode: <strong id="cameraMode">Loading...</strong>
        </p>
        
        <div class="upload-area" id="uploadArea">
            <p style="font-size: 60px; margin: 0;">🎤</p>
            <p style="color: #e74c3c; font-weight: bold;">Click to upload or drag & drop</p>
            <p style="color: #999; font-size: 14px;">Supports JPG, PNG, GIF</p>
            <input type="file" id="fileInput" accept="image/*" style="display: none;">
        </div>
        
        <img id="preview">
        <div class="loading" id="loading"><div class="spinner"></div><p style="color: #e74c3c; margin-top: 10px;">Detecting artists...</p></div>
        <div class="results" id="results"></div>
    </div>

    <script>
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const preview = document.getElementById('preview');
        const loading = document.getElementById('loading');
        const results = document.getElementById('results');
        
        // Load camera mode
        fetch('/camera-info')
            .then(response => response.json())
            .then(data => {
                const mode = data.camera_mode === 'external' ? 'External (Artists)' : 'Internal (Sound Engineer)';
                document.getElementById('cameraMode').textContent = mode;
            })
            .catch(() => document.getElementById('cameraMode').textContent = 'Unknown');

        uploadArea.addEventListener('click', () => fileInput.click());
        uploadArea.addEventListener('dragover', (e) => { e.preventDefault(); uploadArea.classList.add('dragging'); });
        uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('dragging'));
        uploadArea.addEventListener('drop', handleFile);
        fileInput.addEventListener('change', (e) => handleFile(e));

        function handleFile(e) {
            e.preventDefault();
            uploadArea.classList.remove('dragging');
            
            const file = e.dataTransfer ? e.dataTransfer.files[0] : e.target.files[0];
            if (!file) return;

            const reader = new FileReader();
            reader.onload = function(event) {
                preview.src = event.target.result;
                preview.style.display = 'block';
                detectArtists(event.target.result);
            };
            reader.readAsDataURL(file);
        }

        async function detectArtists(imageData) {
            loading.style.display = 'block';
            results.style.display = 'none';

            try {
                const response = await fetch('/detect', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image: imageData })
                });

                const data = await response.json();
                
                loading.style.display = 'none';
                
                if (data.success) {
                    showResults(data);
                } else {
                    results.innerHTML = `<div class="result-item">❌ Error: ${data.error}</div>`;
                    results.style.display = 'block';
                }
            } catch (error) {
                loading.style.display = 'none';
                results.innerHTML = `<div class="result-item">❌ Error: ${error.message}</div>`;
                results.style.display = 'block';
            }
        }

        function showResults(data) {
            let html = `<h3>Found ${data.detections.length} person(s) in ${data.process_time}ms</h3>`;
            
            data.detections.forEach((detection, i) => {
                html += `<div class="result-item">Person ${i+1}: x=${detection[0]}, y=${detection[1]}, w=${detection[2]-detection[0]}, h=${detection[3]-detection[1]}</div>`;
            });
            
            results.innerHTML = html;
            results.style.display = 'block';
        }
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(DEMO_HTML)

@app.route('/detect', methods=['POST'])
def detect_artist():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'success': False, 'error': 'No image data provided'})

        # Decode base64 image
        image_data = data['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to OpenCV format
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Detect artists
        start_time = time.time()
        detections = detect_artists(frame)
        process_time = int((time.time() - start_time) * 1000)
        
        return jsonify({
            'success': True,
            'detections': detections,
            'count': len(detections),
            'process_time': process_time
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/camera-info')
def camera_info():
    return jsonify({
        'camera_mode': CAMERA_MODE,
        'description': 'External (facing artists)' if CAMERA_MODE == 'external' else 'Internal (facing sound engineer)'
    })

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy', 
        'service': 'artist_detection',
        'camera_mode': CAMERA_MODE
    })

if __name__ == '__main__':
    camera_desc = 'External (Artists)' if CAMERA_MODE == 'external' else 'Internal (Sound Engineer)'
    print("🎤 Starting Artist Detection API...")
    print(f"📷 Camera Mode: {camera_desc}")
    print(f"📋 API will be available at: http://localhost:{API_PORT}")
    app.run(host='0.0.0.0', port=API_PORT, debug=False)
