#!/usr/bin/env python3
"""
🎤 Artist Detection WebSocket Server - Hackathon Edition
Real-time artist detection via WebSockets for Electron frontends
Bulletproof, simple, and fast!
"""

import asyncio
import websockets
import json
import base64
import io
import cv2
import numpy as np
from PIL import Image
import time
import os
import sys
import logging
from pathlib import Path
import traceback
from typing import Dict, Any, Optional

# Add artist_detection to path
sys.path.append('artist_detection')
try:
    from artist_detection.detection.artist_detector import detect_artists
except ImportError as e:
    print(f"❌ Error importing artist detector: {e}")
    print("🔧 Make sure you're running from the correct directory")
    sys.exit(1)

# Configuration
SOCKET_HOST = os.environ.get('SOCKET_HOST', '127.0.0.1')
SOCKET_PORT = int(os.environ.get('SOCKET_PORT', '8765'))
CAMERA_MODE = os.environ.get('CAMERA_MODE', 'external')

# Setup colorful logging for hackathon debugging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Track connected clients for broadcasting
connected_clients = set()

class ArtistDetectionServer:
    """Bulletproof WebSocket server for real-time artist detection"""
    
    def __init__(self):
        self.stats = {
            'total_detections': 0,
            'successful_detections': 0,
            'failed_detections': 0,
            'average_process_time': 0,
            'uptime_start': time.time()
        }
    
    async def register_client(self, websocket):
        """Register a new client connection"""
        connected_clients.add(websocket)
        client_ip = websocket.remote_address[0] if websocket.remote_address else "unknown"
        logger.info(f"🔗 New client connected: {client_ip} | Total clients: {len(connected_clients)}")
        
        # Send welcome message with server info
        welcome_msg = {
            'type': 'welcome',
            'message': '🎤 Connected to Artist Detection WebSocket Server',
            'camera_mode': CAMERA_MODE,
            'server_stats': self.get_server_stats(),
            'timestamp': time.time()
        }
        await self.safe_send(websocket, welcome_msg)
    
    async def unregister_client(self, websocket):
        """Unregister a client connection"""
        connected_clients.discard(websocket)
        logger.info(f"📤 Client disconnected | Remaining clients: {len(connected_clients)}")
    
    async def safe_send(self, websocket, message):
        """Safely send a message to a client with error handling"""
        try:
            await websocket.send(json.dumps(message))
        except websockets.exceptions.ConnectionClosed:
            logger.warning("⚠️ Tried to send to closed connection")
            await self.unregister_client(websocket)
        except Exception as e:
            logger.error(f"❌ Error sending message: {e}")
    
    async def broadcast_to_all(self, message, exclude_websocket=None):
        """Broadcast a message to all connected clients"""
        if not connected_clients:
            return
        
        # Create a copy to avoid modification during iteration
        clients_copy = connected_clients.copy()
        
        for client in clients_copy:
            if client != exclude_websocket and not client.closed:
                await self.safe_send(client, message)
    
    def decode_image(self, image_data: str) -> Optional[np.ndarray]:
        """Decode base64 image data to OpenCV format with error handling"""
        try:
            # Handle data URL format
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            # Decode base64
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to OpenCV format (BGR)
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            return frame
            
        except Exception as e:
            logger.error(f"❌ Image decode error: {e}")
            return None
    
    async def process_detection(self, image_data: str) -> Dict[str, Any]:
        """Process artist detection with comprehensive error handling"""
        start_time = time.time()
        
        try:
            # Decode image
            frame = self.decode_image(image_data)
            if frame is None:
                self.stats['failed_detections'] += 1
                return {
                    'success': False,
                    'error': 'Failed to decode image data',
                    'timestamp': time.time()
                }
            
            # Perform detection
            detections = detect_artists(frame)
            process_time = int((time.time() - start_time) * 1000)
            
            # Update stats
            self.stats['total_detections'] += 1
            self.stats['successful_detections'] += 1
            
            # Calculate rolling average
            if self.stats['average_process_time'] == 0:
                self.stats['average_process_time'] = process_time
            else:
                self.stats['average_process_time'] = int(
                    (self.stats['average_process_time'] + process_time) / 2
                )
            
            logger.info(f"✅ Detection complete: {len(detections)} people found in {process_time}ms")
            
            return {
                'success': True,
                'detections': detections,
                'count': len(detections),
                'process_time': process_time,
                'timestamp': time.time(),
                'camera_mode': CAMERA_MODE
            }
            
        except Exception as e:
            self.stats['failed_detections'] += 1
            error_msg = f"Detection error: {str(e)}"
            logger.error(f"❌ {error_msg}")
            logger.debug(traceback.format_exc())
            
            return {
                'success': False,
                'error': error_msg,
                'timestamp': time.time()
            }
    
    def get_server_stats(self) -> Dict[str, Any]:
        """Get current server statistics"""
        uptime = int(time.time() - self.stats['uptime_start'])
        return {
            **self.stats,
            'uptime_seconds': uptime,
            'connected_clients': len(connected_clients),
            'camera_mode': CAMERA_MODE
        }
    
    async def handle_message(self, websocket, message_text: str):
        """Handle incoming WebSocket messages with routing"""
        try:
            message = json.loads(message_text)
            message_type = message.get('type', 'unknown')
            
            logger.info(f"📨 Received message type: {message_type}")
            
            # Route message based on type
            if message_type == 'detect':
                await self.handle_detect_message(websocket, message)
            elif message_type == 'ping':
                await self.handle_ping_message(websocket, message)
            elif message_type == 'stats':
                await self.handle_stats_message(websocket, message)
            elif message_type == 'camera_info':
                await self.handle_camera_info_message(websocket, message)
            else:
                await self.safe_send(websocket, {
                    'type': 'error',
                    'error': f'Unknown message type: {message_type}',
                    'timestamp': time.time()
                })
                
        except json.JSONDecodeError:
            logger.error("❌ Invalid JSON received")
            await self.safe_send(websocket, {
                'type': 'error',
                'error': 'Invalid JSON format',
                'timestamp': time.time()
            })
        except Exception as e:
            logger.error(f"❌ Message handling error: {e}")
            await self.safe_send(websocket, {
                'type': 'error',
                'error': f'Message processing error: {str(e)}',
                'timestamp': time.time()
            })
    
    async def handle_detect_message(self, websocket, message):
        """Handle artist detection requests"""
        if 'image' not in message:
            await self.safe_send(websocket, {
                'type': 'detection_result',
                'success': False,
                'error': 'No image data provided',
                'timestamp': time.time()
            })
            return
        
        # Process detection
        result = await self.process_detection(message['image'])
        result['type'] = 'detection_result'
        
        # Send result back to client
        await self.safe_send(websocket, result)
        
        # Optional: Broadcast to other clients (useful for multi-client scenarios)
        if result['success'] and message.get('broadcast', False):
            broadcast_msg = {
                'type': 'detection_broadcast',
                'count': result['count'],
                'timestamp': result['timestamp'],
                'camera_mode': CAMERA_MODE
            }
            await self.broadcast_to_all(broadcast_msg, exclude_websocket=websocket)
    
    async def handle_ping_message(self, websocket, message):
        """Handle ping/pong for connection health"""
        await self.safe_send(websocket, {
            'type': 'pong',
            'timestamp': time.time(),
            'server_time': time.time()
        })
    
    async def handle_stats_message(self, websocket, message):
        """Handle server statistics requests"""
        await self.safe_send(websocket, {
            'type': 'stats_result',
            'stats': self.get_server_stats(),
            'timestamp': time.time()
        })
    
    async def handle_camera_info_message(self, websocket, message):
        """Handle camera info requests"""
        await self.safe_send(websocket, {
            'type': 'camera_info_result',
            'camera_mode': CAMERA_MODE,
            'description': 'External (facing artists)' if CAMERA_MODE == 'external' else 'Internal (facing sound engineer)',
            'timestamp': time.time()
        })
    
    async def handle_client(self, websocket, path):
        """Main client handler with comprehensive error handling"""
        await self.register_client(websocket)
        
        try:
            async for message in websocket:
                await self.handle_message(websocket, message)
                
        except websockets.exceptions.ConnectionClosed:
            logger.info("🔌 Client connection closed normally")
        except Exception as e:
            logger.error(f"❌ Client handler error: {e}")
        finally:
            await self.unregister_client(websocket)

# Global server instance
server_instance = ArtistDetectionServer()

async def health_check():
    """Periodic health check and stats logging"""
    while True:
        await asyncio.sleep(30)  # Every 30 seconds
        stats = server_instance.get_server_stats()
        logger.info(f"📊 Health Check | Clients: {stats['connected_clients']} | "
                   f"Total Detections: {stats['total_detections']} | "
                   f"Success Rate: {stats['successful_detections']}/{stats['total_detections']} | "
                   f"Avg Time: {stats['average_process_time']}ms")

async def main():
    """Main server startup with bulletproof error handling"""
    try:
        camera_desc = 'External (Artists)' if CAMERA_MODE == 'external' else 'Internal (Sound Engineer)'
        
        print("🎤" + "="*50)
        print("🎤 ARTIST DETECTION WEBSOCKET SERVER")
        print("🎤" + "="*50)
        print(f"📷 Camera Mode: {camera_desc}")
        print(f"🌐 Server Address: ws://{SOCKET_HOST}:{SOCKET_PORT}")
        print(f"🔗 WebSocket URL: ws://{SOCKET_HOST}:{SOCKET_PORT}")
        print("🎤" + "="*50)
        print("🚀 Starting server...")
        
        # Start health check task
        health_task = asyncio.create_task(health_check())
        
        # Start WebSocket server
        async with websockets.serve(
            server_instance.handle_client, 
            SOCKET_HOST, 
            SOCKET_PORT,
            ping_interval=20,
            ping_timeout=10,
            close_timeout=5
        ):
            print("✅ WebSocket server started successfully!")
            print("🎯 Ready for Electron frontend connections!")
            print("💡 Tip: Use the test client (test_socket_client.html) to verify")
            print("⏹️  Press Ctrl+C to stop the server")
            
            # Keep server running
            await asyncio.Future()  # Run forever
            
    except KeyboardInterrupt:
        print("\n🛑 Server shutdown requested by user")
    except Exception as e:
        print(f"❌ Server startup error: {e}")
        traceback.print_exc()
    finally:
        print("👋 Artist Detection WebSocket Server stopped")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Critical error: {e}")
        sys.exit(1)