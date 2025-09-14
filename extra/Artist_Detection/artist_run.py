#!/usr/bin/env python3
"""
SonicMind AI - Artist Detection API Runner
Simple script to run the API server with camera configuration
"""

import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(
        description="🎯 Run SonicMind AI Artist Detection API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python artist_run.py                    # Start with external camera (default)
  python artist_run.py --camera external  # External camera (facing artists)
  python artist_run.py --camera internal  # Internal camera (facing sound engineer)
  python artist_run.py --port 5002        # Custom port
        """
    )
    
    parser.add_argument('--camera', choices=['external', 'internal'], 
                       default='external',
                       help='Camera mode: external (default, faces artists) or internal (faces sound engineer)')
    parser.add_argument('--port', type=int, default=5001,
                       help='Port to run API on (default: 5001)')
    parser.add_argument('--debug', action='store_true',
                       help='Run in debug mode')
    
    args = parser.parse_args()
    
    print("🎯 SonicMind AI - Artist Detection API")
    print("=" * 40)
    print(f"📷 Camera Mode: {args.camera}")
    print(f"🌐 Port: {args.port}")
    print("=" * 40)
    
    if args.camera == "external":
        print("📹 External camera - facing artists/performers")
    else:
        print("📹 Internal camera - facing sound engineer")
    
    # Set environment variables for the API to pick up
    os.environ['CAMERA_MODE'] = args.camera
    os.environ['API_PORT'] = str(args.port)
    if args.debug:
        os.environ['FLASK_DEBUG'] = '1'
    
    # Import and run the API
    try:
        print("\n🚀 Starting API server...")
        
        # Check if we have the artist detection API
        if os.path.exists("artist_detection_api.py"):
            import artist_detection_api
            # The API should read the environment variables
            artist_detection_api.app.run(host='0.0.0.0', port=args.port, debug=args.debug)
        else:
            print("❌ artist_detection_api.py not found!")
            print("   Make sure you're in the correct directory.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n👋 API server stopped!")
    except Exception as e:
        print(f"❌ Error starting API: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()