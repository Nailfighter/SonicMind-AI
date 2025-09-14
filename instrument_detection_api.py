#!/usr/bin/env python3
"""
Fast API Server for Instrument Detection
Ready for hackathon demo!
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import base64
import io
from PIL import Image
import os
import time
from instrument_detector_zero_shot import ZeroShotInstrumentDetector

app = Flask(__name__)
CORS(app)  # Enable CORS for web demos

# Initialize detector once (singleton pattern)
print("Initializing instrument detector...")
detector = ZeroShotInstrumentDetector()
print("✅ Ready to serve!")

# Simple HTML interface for quick demos
DEMO_HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>🎸 Instrument Detector - Hackathon Demo</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            background: white;
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 10px;
        }
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
        }
        .upload-area {
            border: 3px dashed #667eea;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
            background: #f8f9ff;
        }
        .upload-area:hover {
            border-color: #764ba2;
            background: #f0f2ff;
        }
        .upload-area.dragging {
            background: #e8ebff;
            border-color: #764ba2;
        }
        #preview {
            max-width: 100%;
            max-height: 400px;
            margin: 20px auto;
            display: none;
            border-radius: 10px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        .results {
            margin-top: 30px;
            display: none;
        }
        .result-item {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            margin: 10px 0;
            border-radius: 10px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            animation: slideIn 0.3s ease;
        }
        @keyframes slideIn {
            from { transform: translateX(-20px); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
        .confidence-bar {
            background: rgba(255,255,255,0.3);
            height: 8px;
            border-radius: 4px;
            overflow: hidden;
            flex: 1;
            margin: 0 15px;
        }
        .confidence-fill {
            height: 100%;
            background: white;
            transition: width 0.5s ease;
        }
        .instrument-name {
            font-weight: bold;
            font-size: 18px;
            text-transform: capitalize;
        }
        .confidence-text {
            font-size: 20px;
            font-weight: bold;
        }
        .loading {
            display: none;
            text-align: center;
            margin: 20px 0;
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .stats {
            display: flex;
            justify-content: space-around;
            margin-top: 20px;
            padding-top: 20px;
            border-top: 1px solid #eee;
        }
        .stat {
            text-align: center;
        }
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: #667eea;
        }
        .stat-label {
            color: #999;
            font-size: 12px;
            text-transform: uppercase;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎸 Instrument Detector</h1>
        <p class="subtitle">AI-Powered Musical Instrument Recognition</p>
        
        <div class="upload-area" id="uploadArea">
            <p style="font-size: 60px; margin: 0;">🎵</p>
            <p style="color: #667eea; font-weight: bold;">Click to upload or drag & drop</p>
            <p style="color: #999; font-size: 14px;">Supports JPG, PNG, GIF</p>
            <input type="file" id="fileInput" accept="image/*" style="display: none;">
        </div>
        
        <img id="preview">
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p style="color: #667eea; margin-top: 10px;">Analyzing instrument...</p>
        </div>
        
        <div class="results" id="results"></div>
        
        <div class="stats" id="stats" style="display: none;">
            <div class="stat">
                <div class="stat-value" id="processTime">-</div>
                <div class="stat-label">Process Time</div>
            </div>
            <div class="stat">
                <div class="stat-value" id="confidence">-</div>
                <div class="stat-label">Confidence</div>
            </div>
            <div class="stat">
                <div class="stat-value">CLIP</div>
                <div class="stat-label">Model</div>
            </div>
        </div>
    </div>

    <script>
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const preview = document.getElementById('preview');
        const results = document.getElementById('results');
        const loading = document.getElementById('loading');
        const stats = document.getElementById('stats');
        
        uploadArea.addEventListener('click', () => fileInput.click());
        
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragging');
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragging');
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragging');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleFile(files[0]);
            }
        });
        
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFile(e.target.files[0]);
            }
        });
        
        function handleFile(file) {
            if (!file.type.startsWith('image/')) {
                alert('Please upload an image file');
                return;
            }
            
            const reader = new FileReader();
            reader.onload = (e) => {
                preview.src = e.target.result;
                preview.style.display = 'block';
                analyzeImage(e.target.result);
            };
            reader.readAsDataURL(file);
        }
        
        async function analyzeImage(base64Image) {
            loading.style.display = 'block';
            results.style.display = 'none';
            stats.style.display = 'none';
            
            const startTime = Date.now();
            
            try {
                const response = await fetch('/detect', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({image: base64Image})
                });
                
                const data = await response.json();
                const elapsed = ((Date.now() - startTime) / 1000).toFixed(2);
                
                loading.style.display = 'none';
                displayResults(data.predictions, elapsed);
                
            } catch (error) {
                loading.style.display = 'none';
                alert('Error analyzing image: ' + error);
            }
        }
        
        function displayResults(predictions, elapsed) {
            results.innerHTML = '<h3 style="color: #333;">Detected Instruments:</h3>';
            
            predictions.forEach((pred, index) => {
                const div = document.createElement('div');
                div.className = 'result-item';
                div.style.animationDelay = `${index * 0.1}s`;
                
                const name = pred[0].replace(/_/g, ' ');
                const confidence = (pred[1] * 100).toFixed(1);
                
                div.innerHTML = `
                    <span class="instrument-name">${name}</span>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: ${confidence}%"></div>
                    </div>
                    <span class="confidence-text">${confidence}%</span>
                `;
                
                results.appendChild(div);
            });
            
            results.style.display = 'block';
            stats.style.display = 'flex';
            
            document.getElementById('processTime').textContent = elapsed + 's';
            document.getElementById('confidence').textContent = 
                (predictions[0][1] * 100).toFixed(1) + '%';
        }
    </script>
</body>
</html>
'''

@app.route('/')
def home():
    """Serve the demo interface."""
    return render_template_string(DEMO_HTML)

@app.route('/detect', methods=['POST'])
def detect_instrument():
    """
    Detect instrument in uploaded image.
    Accepts base64 encoded image or file upload.
    """
    try:
        start_time = time.time()
        
        # Get image from request
        data = request.get_json()
        
        if data and 'image' in data:
            # Base64 encoded image
            image_data = data['image']
            # Remove data URL prefix if present
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            # Decode base64
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            # Save temporarily
            temp_path = '/tmp/temp_instrument.jpg'
            image.save(temp_path)
            
        else:
            # File upload
            file = request.files.get('image')
            if not file:
                return jsonify({'error': 'No image provided'}), 400
            
            temp_path = '/tmp/temp_instrument.jpg'
            file.save(temp_path)
        
        # Detect instruments
        predictions = detector.detect(temp_path, top_k=5)
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        # Calculate processing time
        process_time = time.time() - start_time
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'process_time': f'{process_time:.3f}s',
            'model': 'CLIP ViT-B/32'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'model': 'loaded'})

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🎸 INSTRUMENT DETECTION API")
    print("="*50)
    print("\n📱 Web Interface: http://localhost:5000")
    print("🔌 API Endpoint: http://localhost:5000/detect")
    print("\nPress Ctrl+C to stop\n")
    
    # Run server
    app.run(host='0.0.0.0', port=5000, debug=False)