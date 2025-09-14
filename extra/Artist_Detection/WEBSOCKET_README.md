# 🎤 Artist Detection WebSocket Server - Hackathon Edition

A bulletproof, real-time WebSocket server for artist detection, designed specifically for **Electron frontends** and hackathon use cases.

## 🚀 Quick Start (2 minutes setup)

### 1. Start the Server
```powershell
# In PowerShell (run as administrator if needed)
.\start_socket_server.ps1

# Or with custom settings
.\start_socket_server.ps1 -Host "0.0.0.0" -Port 8765 -CameraMode "external"
```

### 2. Test the Connection
Open `test_socket_client.html` in your browser to verify everything works.

### 3. Connect Your Electron App
```javascript
// In your Electron renderer process
const socket = new WebSocket('ws://127.0.0.1:8765');

socket.onopen = () => {
    console.log('🎤 Connected to Artist Detection Server!');
};

socket.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('Received:', data);
};
```

## 🎯 Features for Hackathons

✅ **Bulletproof Error Handling** - Won't crash during demos  
✅ **Real-time Detection** - Instant results via WebSockets  
✅ **Multiple Client Support** - Connect multiple Electron windows  
✅ **Built-in Statistics** - Track performance metrics  
✅ **Easy Setup** - One-click startup script  
✅ **Test Client Included** - Verify everything works  
✅ **Detailed Logging** - Debug issues quickly  

## 📡 WebSocket API Reference

### Connection
```javascript
const socket = new WebSocket('ws://127.0.0.1:8765');
```

### Message Format
All messages are JSON with a `type` field:
```javascript
{
    "type": "message_type",
    "data": "...",
    "timestamp": 1634567890
}
```

### 🔍 Artist Detection
**Send:**
```javascript
socket.send(JSON.stringify({
    type: 'detect',
    image: 'data:image/jpeg;base64,/9j/4AAQSkZJRgABA...',  // Base64 image
    broadcast: false  // Optional: broadcast results to other clients
}));
```

**Receive:**
```javascript
{
    type: 'detection_result',
    success: true,
    detections: [[x1, y1, x2, y2], [x1, y1, x2, y2]],  // Bounding boxes
    count: 2,
    process_time: 150,  // milliseconds
    timestamp: 1634567890,
    camera_mode: 'external'
}
```

### 🏓 Health Check
**Send:**
```javascript
socket.send(JSON.stringify({type: 'ping'}));
```

**Receive:**
```javascript
{
    type: 'pong',
    timestamp: 1634567890,
    server_time: 1634567890
}
```

### 📊 Server Statistics
**Send:**
```javascript
socket.send(JSON.stringify({type: 'stats'}));
```

**Receive:**
```javascript
{
    type: 'stats_result',
    stats: {
        total_detections: 42,
        successful_detections: 40,
        failed_detections: 2,
        average_process_time: 120,
        uptime_seconds: 3600,
        connected_clients: 2,
        camera_mode: 'external'
    },
    timestamp: 1634567890
}
```

### 📷 Camera Info
**Send:**
```javascript
socket.send(JSON.stringify({type: 'camera_info'}));
```

**Receive:**
```javascript
{
    type: 'camera_info_result',
    camera_mode: 'external',
    description: 'External (facing artists)',
    timestamp: 1634567890
}
```

## 🖥️ Electron Integration Examples

### Basic Connection Manager
```javascript
// main-process.js or renderer-process.js
class ArtistDetectionClient {
    constructor() {
        this.socket = null;
        this.isConnected = false;
    }
    
    connect() {
        this.socket = new WebSocket('ws://127.0.0.1:8765');
        
        this.socket.onopen = () => {
            this.isConnected = true;
            console.log('🎤 Connected to Artist Detection Server');
        };
        
        this.socket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleMessage(data);
        };
        
        this.socket.onclose = () => {
            this.isConnected = false;
            console.log('🔌 Disconnected from server');
            // Auto-reconnect logic here if needed
        };
    }
    
    detectArtists(imageBase64) {
        if (!this.isConnected) return;
        
        this.socket.send(JSON.stringify({
            type: 'detect',
            image: imageBase64
        }));
    }
    
    handleMessage(data) {
        switch(data.type) {
            case 'detection_result':
                if (data.success) {
                    console.log(`Found ${data.count} artists in ${data.process_time}ms`);
                    // Update your UI here
                    this.updateUI(data.detections);
                }
                break;
            case 'welcome':
                console.log(data.message);
                break;
        }
    }
    
    updateUI(detections) {
        // Update your Electron UI with detection results
        detections.forEach((box, i) => {
            const [x1, y1, x2, y2] = box;
            console.log(`Person ${i+1}: ${x1},${y1} to ${x2},${y2}`);
        });
    }
}

// Usage
const client = new ArtistDetectionClient();
client.connect();
```

### Canvas Integration for Real-time
```javascript
// For video/camera feeds in Electron
function processVideoFrame(canvas) {
    // Get image data from canvas
    const imageData = canvas.toDataURL('image/jpeg', 0.8);
    
    // Send for detection
    client.detectArtists(imageData);
}

// Set up periodic detection (every 500ms)
setInterval(() => {
    if (videoCanvas && client.isConnected) {
        processVideoFrame(videoCanvas);
    }
}, 500);
```

### File Upload Integration
```javascript
// For file-based detection in Electron
function handleFileUpload(filePath) {
    const fs = require('fs');
    const path = require('path');
    
    // Read image file
    const imageBuffer = fs.readFileSync(filePath);
    const base64 = imageBuffer.toString('base64');
    const mimeType = path.extname(filePath) === '.png' ? 'image/png' : 'image/jpeg';
    const dataUrl = `data:${mimeType};base64,${base64}`;
    
    // Send for detection
    client.detectArtists(dataUrl);
}
```

## 🛠️ Configuration Options

### Environment Variables
```bash
SOCKET_HOST=127.0.0.1        # Server host (use 0.0.0.0 for external access)
SOCKET_PORT=8765             # Server port
CAMERA_MODE=external         # 'external' or 'internal'
```

### PowerShell Parameters
```powershell
.\start_socket_server.ps1 -Host "0.0.0.0" -Port 9000 -CameraMode "internal"
```

## 🐛 Troubleshooting

### Common Issues

**❌ "Failed to connect"**
- Check if server is running: `netstat -an | findstr :8765`
- Verify firewall settings
- Try `127.0.0.1` instead of `localhost`

**❌ "Image decode error"**
- Ensure image is valid base64 with data URL prefix
- Check image format (JPEG/PNG supported)
- Reduce image size if too large

**❌ "Module not found: artist_detection"**
- Run from the correct directory (Artist_Detection folder)
- Check if `artist_detection` folder exists

**❌ "YOLO model not found"**
- The server will automatically download YOLOv8n if needed
- Ensure internet connection for first run

### Debug Mode
Enable detailed logging:
```python
# In socket_server.py, change line 40:
logging.basicConfig(level=logging.DEBUG)
```

## 📊 Performance Tips for Hackathons

### Image Optimization
- Resize images to 640x640 before sending
- Use JPEG with 80% quality for speed
- Process every 2-3 frames for video, not every frame

### Connection Management
- Implement reconnection logic
- Use ping/pong to detect disconnections
- Handle multiple simultaneous requests gracefully

### Error Recovery
```javascript
// Robust error handling for demos
socket.onerror = (error) => {
    console.error('WebSocket error:', error);
    // Show user-friendly message
    showError('Connection issue - retrying...');
    
    // Auto-retry after 3 seconds
    setTimeout(() => {
        client.connect();
    }, 3000);
};
```

## 🎯 Hackathon Best Practices

### 1. Always Test First
```bash
# Open test client before connecting Electron
start test_socket_client.html
```

### 2. Have a Backup Plan
- Keep both Flask and WebSocket servers ready
- Test with static images if camera fails
- Prepare demo data/images

### 3. Monitor Performance
- Use the stats endpoint to track performance
- Monitor memory usage during long runs
- Restart server if needed between major demos

### 4. Demo-Ready Features
- Real-time detection counts
- Visual feedback in UI
- Error recovery messages
- Performance metrics display

## 📁 Project Structure
```
Artist_Detection/
├── socket_server.py              # Main WebSocket server
├── socket_requirements.txt       # Dependencies for socket server
├── start_socket_server.ps1       # Easy startup script
├── test_socket_client.html       # Test client
├── WEBSOCKET_README.md           # This documentation
├── artist_detection/             # Detection modules
│   └── detection/
│       └── artist_detector.py    # YOLO detection logic
└── venv_socket/                  # Virtual environment (created automatically)
```

## 🎤 Quick Commands Reference

```powershell
# Start server (basic)
.\start_socket_server.ps1

# Start server with custom settings
.\start_socket_server.ps1 -Host "0.0.0.0" -Port 9000

# Test connection
start test_socket_client.html

# Check server status
netstat -an | findstr :8765

# View server logs in real-time
Get-Content server.log -Wait -Tail 50
```

## 🏆 Success Checklist for Demos

- [ ] Server starts without errors
- [ ] Test client connects successfully
- [ ] Image detection returns results
- [ ] Electron app connects to WebSocket
- [ ] Real-time updates work
- [ ] Error handling works gracefully
- [ ] Performance is acceptable (< 500ms)
- [ ] Multiple clients can connect
- [ ] Server survives connection drops
- [ ] Demo images/videos prepared

---

**🎉 Ready for your hackathon! Good luck!** 

Need help? The test client includes comprehensive logging and error messages to help you debug any issues quickly.

*Built with ❤️ for hackathon success by focusing on simplicity and reliability.*