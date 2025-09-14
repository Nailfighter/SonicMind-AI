#!/usr/bin/env python3
"""
Unified Test Interface for Hackathon Project
Test both Artist Detection and Instrument Detection with GUI or Terminal
"""

import argparse
import requests
import base64
import json
import sys
import os
from pathlib import Path

def encode_image(image_path):
    """Encode image to base64"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e:
        print(f"❌ Error reading image: {e}")
        return None

def test_instrument_detection(image_path, api_url="http://localhost:5000"):
    """Test instrument detection API"""
    print(f"🎸 Testing Instrument Detection...")
    print(f"📷 Image: {image_path}")
    print(f"🌐 API: {api_url}")
    
    image_data = encode_image(image_path)
    if not image_data:
        return False
    
    try:
        response = requests.post(f"{api_url}/detect", 
                               json={"image": f"data:image/jpeg;base64,{image_data}"},
                               timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success! Found instruments:")
            for prediction in result.get('predictions', []):
                # prediction is a tuple: (instrument_name, confidence)
                if isinstance(prediction, list) and len(prediction) >= 2:
                    instrument = prediction[0]
                    confidence = prediction[1]
                    print(f"   🎵 {instrument}: {confidence:.2%}")
            print(f"⏱️  Processing time: {result.get('process_time', 'N/A')}")
            return True
        else:
            print(f"❌ API Error {response.status_code}: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection Error: {e}")
        return False

def test_artist_detection(image_path, api_url="http://localhost:5001"):
    """Test artist detection API"""
    print(f"👤 Testing Artist Detection...")
    print(f"📷 Image: {image_path}")
    print(f"🌐 API: {api_url}")
    
    image_data = encode_image(image_path)
    if not image_data:
        return False
    
    try:
        response = requests.post(f"{api_url}/detect", 
                               json={"image": f"data:image/jpeg;base64,{image_data}"},
                               timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success! Found {result.get('count', 0)} person(s):")
            for i, detection in enumerate(result.get('detections', [])):
                x1, y1, x2, y2 = detection
                w, h = x2 - x1, y2 - y1
                print(f"   🧑 Person {i+1}: x={x1}, y={y1}, width={w}, height={h}")
            print(f"⏱️  Processing time: {result.get('process_time', 'N/A')}ms")
            return True
        else:
            print(f"❌ API Error {response.status_code}: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection Error: {e}")
        return False

def check_api_health(api_url, service_name):
    """Check if API is running"""
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        if response.status_code == 200:
            print(f"✅ {service_name} API is healthy")
            return True
        else:
            print(f"⚠️  {service_name} API returned {response.status_code}")
            return False
    except:
        print(f"❌ {service_name} API is not responding")
        return False

def interactive_gui_mode():
    """Interactive GUI mode for testing"""
    print("🖥️  Starting Interactive GUI Mode...")
    print("\nAvailable Commands:")
    print("  1. Test instrument detection")
    print("  2. Test artist detection") 
    print("  3. Test both detections")
    print("  4. Check API health")
    print("  5. Exit")
    
    while True:
        try:
            choice = input("\n👉 Choose an option (1-5): ").strip()
            
            if choice == "1":
                image_path = input("📁 Enter image path: ").strip()
                if os.path.exists(image_path):
                    test_instrument_detection(image_path)
                else:
                    print("❌ Image file not found!")
                    
            elif choice == "2":
                image_path = input("📁 Enter image path: ").strip()
                if os.path.exists(image_path):
                    test_artist_detection(image_path)
                else:
                    print("❌ Image file not found!")
                    
            elif choice == "3":
                image_path = input("📁 Enter image path: ").strip()
                if os.path.exists(image_path):
                    print("🔄 Testing both detections...\n")
                    test_instrument_detection(image_path)
                    print()
                    test_artist_detection(image_path)
                else:
                    print("❌ Image file not found!")
                    
            elif choice == "4":
                print("🏥 Checking API Health...")
                check_api_health("http://localhost:5000", "Instrument Detection")
                check_api_health("http://localhost:5001", "Artist Detection")
                
            elif choice == "5":
                print("👋 Goodbye!")
                break
                
            else:
                print("❌ Invalid choice. Please enter 1-5.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except EOFError:
            break

def terminal_mode(args):
    """Terminal mode with command line arguments"""
    success = True
    
    if not os.path.exists(args.image):
        print(f"❌ Image file not found: {args.image}")
        return False
    
    print("🚀 Hackathon Project - AI Detection Testing")
    print("=" * 50)
    
    if args.detection in ['both', 'instrument']:
        success &= test_instrument_detection(args.image, args.instrument_url)
        if args.detection == 'both':
            print()
    
    if args.detection in ['both', 'artist']:
        success &= test_artist_detection(args.image, args.artist_url)
    
    return success

def find_sample_images():
    """Find sample images for testing"""
    sample_dirs = ['training_artifacts/instrument_dataset/test', 'samples', '.']
    samples = []
    
    for directory in sample_dirs:
        if os.path.exists(directory):
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                        samples.append(os.path.join(root, file))
                        if len(samples) >= 5:  # Limit to 5 samples
                            return samples
    return samples

def main():
    parser = argparse.ArgumentParser(
        description="🎯 Test Artist & Instrument Detection APIs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python artist_test.py                          # Interactive GUI mode
  python artist_test.py image.jpg                # Test both detections
  python artist_test.py image.jpg --instrument   # Test only instrument detection  
  python artist_test.py image.jpg --artist       # Test only artist detection
  python artist_test.py --health                 # Check API health
        """
    )
    
    parser.add_argument('image', nargs='?', help='Path to image file')
    parser.add_argument('--detection', choices=['both', 'instrument', 'artist'], 
                       default='both', help='Which detection to test')
    parser.add_argument('--instrument-url', default='http://localhost:5000',
                       help='Instrument detection API URL')
    parser.add_argument('--artist-url', default='http://localhost:5001',
                       help='Artist detection API URL')
    parser.add_argument('--health', action='store_true',
                       help='Check API health only')
    parser.add_argument('--samples', action='store_true',
                       help='Show available sample images')
    
    args = parser.parse_args()
    
    if args.health:
        print("🏥 Checking API Health...")
        check_api_health(args.instrument_url, "Instrument Detection")
        check_api_health(args.artist_url, "Artist Detection")
        return
    
    if args.samples:
        print("🖼️  Finding sample images...")
        samples = find_sample_images()
        if samples:
            print("Found sample images:")
            for i, sample in enumerate(samples, 1):
                print(f"  {i}. {sample}")
        else:
            print("No sample images found.")
        return
    
    if not args.image:
        # No image provided - start GUI mode
        interactive_gui_mode()
    else:
        # Terminal mode with provided image
        success = terminal_mode(args)
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()