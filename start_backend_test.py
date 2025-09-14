#!/usr/bin/env python3
"""
Start Backend for Socket.IO Testing

This script starts the SonicMind-AI backend server for testing Socket.IO integration.
It imports and runs the main server with minimal setup.
"""

import sys
import os

# Add backend directory to Python path
backend_dir = os.path.join(os.path.dirname(__file__), 'backend')
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

# Import and start the main server
if __name__ == "__main__":
    print("🚀" + "="*60)
    print("    SonicMind-AI Backend - Socket.IO Test Server")
    print("="*62)
    print()
    print("Starting backend for Socket.IO integration testing...")
    print("Frontend can connect to: http://localhost:8000")
    print("Press Ctrl+C to stop the server")
    print("="*62)
    
    try:
        # Import main module and start server
        from main import main
        main()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        sys.exit(1)