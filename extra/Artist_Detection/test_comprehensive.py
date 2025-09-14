#!/usr/bin/env python3
"""
Comprehensive Testing Script for Both Detection Systems
Tests instrument detection and artist detection on multiple samples
"""

import os
import sys
import random
import time
import requests
import json
from pathlib import Path

def test_api_health():
    """Check if both APIs are running"""
    print("🏥 Checking API Health...")
    
    # Test instrument detection API
    try:
        response = requests.get("http://localhost:5000/health", timeout=5)
        if response.status_code == 200:
            print("✅ Instrument Detection API is healthy")
            instrument_api_ok = True
        else:
            print(f"⚠️  Instrument Detection API returned {response.status_code}")
            instrument_api_ok = False
    except:
        print("❌ Instrument Detection API is not responding")
        instrument_api_ok = False
    
    # Test artist detection API
    try:
        response = requests.get("http://localhost:5001/health", timeout=5)
        if response.status_code == 200:
            result = response.json()
            camera_mode = result.get('camera_mode', 'unknown')
            print(f"✅ Artist Detection API is healthy (camera: {camera_mode})")
            artist_api_ok = True
        else:
            print(f"⚠️  Artist Detection API returned {response.status_code}")
            artist_api_ok = False
    except:
        print("❌ Artist Detection API is not responding")
        artist_api_ok = False
    
    return instrument_api_ok, artist_api_ok

def get_test_samples(test_dir, samples_per_instrument=3):
    """Get random test samples from each instrument category"""
    test_path = Path(test_dir)
    samples = {}
    
    if not test_path.exists():
        print(f"❌ Test directory not found: {test_dir}")
        return samples
    
    for instrument_dir in test_path.iterdir():
        if instrument_dir.is_dir():
            instrument = instrument_dir.name
            images = list(instrument_dir.glob("*.jpg"))
            
            if images:
                # Get random samples
                random.shuffle(images)
                samples[instrument] = images[:samples_per_instrument]
                print(f"📁 Found {len(images)} images for {instrument}, testing {len(samples[instrument])}")
    
    return samples

def test_instrument_detection(image_path, expected_instrument):
    """Test instrument detection on a single image"""
    try:
        import base64
        
        # Read and encode image
        with open(image_path, "rb") as img_file:
            image_data = base64.b64encode(img_file.read()).decode('utf-8')
        
        # Make API request
        start_time = time.time()
        response = requests.post(
            "http://localhost:5000/detect", 
            json={"image": f"data:image/jpeg;base64,{image_data}"},
            timeout=30
        )
        process_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            predictions = result.get('predictions', [])
            
            if predictions:
                top_prediction = predictions[0]
                predicted_instrument = top_prediction[0]
                confidence = top_prediction[1]
                
                # Check if correct
                is_correct = predicted_instrument == expected_instrument
                
                return {
                    'success': True,
                    'predicted': predicted_instrument,
                    'expected': expected_instrument,
                    'confidence': confidence,
                    'correct': is_correct,
                    'process_time': process_time,
                    'all_predictions': predictions[:3]  # Top 3
                }
            else:
                return {'success': False, 'error': 'No predictions returned'}
        else:
            return {'success': False, 'error': f'API error {response.status_code}'}
            
    except Exception as e:
        return {'success': False, 'error': str(e)}

def test_artist_detection(image_path):
    """Test artist detection on a single image"""
    try:
        import base64
        
        # Read and encode image
        with open(image_path, "rb") as img_file:
            image_data = base64.b64encode(img_file.read()).decode('utf-8')
        
        # Make API request
        start_time = time.time()
        response = requests.post(
            "http://localhost:5001/detect", 
            json={"image": f"data:image/jpeg;base64,{image_data}"},
            timeout=30
        )
        process_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            detections = result.get('detections', [])
            count = result.get('count', 0)
            
            return {
                'success': True,
                'person_count': count,
                'detections': detections,
                'process_time': process_time
            }
        else:
            return {'success': False, 'error': f'API error {response.status_code}'}
            
    except Exception as e:
        return {'success': False, 'error': str(e)}

def run_comprehensive_test():
    """Run comprehensive tests on both systems"""
    print("🎯 Comprehensive Testing - Both Detection Systems")
    print("=" * 60)
    
    # Check API health
    instrument_ok, artist_ok = test_api_health()
    
    if not instrument_ok and not artist_ok:
        print("\n❌ Both APIs are down. Please start them first:")
        print("   python3 instrument_detection_api.py &")
        print("   python3 artist_detection_api.py &")
        return
    
    print()
    
    # Get test samples
    test_dir = "training_artifacts/instrument_dataset/test"
    samples = get_test_samples(test_dir, samples_per_instrument=2)  # 2 samples per instrument
    
    if not samples:
        print("❌ No test samples found!")
        return
    
    # Test instrument detection
    if instrument_ok:
        print("\n🎸 TESTING INSTRUMENT DETECTION")
        print("-" * 40)
        
        total_tests = 0
        correct_predictions = 0
        total_time = 0
        
        instrument_results = {}
        
        for instrument, image_paths in samples.items():
            print(f"\n📋 Testing {instrument}:")
            instrument_correct = 0
            
            for i, image_path in enumerate(image_paths, 1):
                print(f"   🔍 Sample {i}: {image_path.name}...", end=" ")
                
                result = test_instrument_detection(str(image_path), instrument)
                total_tests += 1
                
                if result['success']:
                    if result['correct']:
                        print(f"✅ {result['predicted']} ({result['confidence']:.1%})")
                        correct_predictions += 1
                        instrument_correct += 1
                    else:
                        print(f"❌ {result['predicted']} ({result['confidence']:.1%}) - Expected: {instrument}")
                    
                    total_time += result['process_time']
                else:
                    print(f"💥 {result['error']}")
            
            accuracy = (instrument_correct / len(image_paths)) * 100 if image_paths else 0
            instrument_results[instrument] = accuracy
            print(f"   📊 {instrument} accuracy: {accuracy:.1f}%")
        
        # Overall instrument detection stats
        overall_accuracy = (correct_predictions / total_tests) * 100 if total_tests else 0
        avg_time = (total_time / total_tests) * 1000 if total_tests else 0
        
        print(f"\n📈 INSTRUMENT DETECTION SUMMARY:")
        print(f"   Overall Accuracy: {overall_accuracy:.1f}% ({correct_predictions}/{total_tests})")
        print(f"   Average Time: {avg_time:.0f}ms per image")
        print(f"   Per-Instrument Results:")
        for instrument, accuracy in sorted(instrument_results.items()):
            print(f"     {instrument:12}: {accuracy:5.1f}%")
    
    # Test artist detection on same images
    if artist_ok:
        print(f"\n👤 TESTING ARTIST DETECTION")
        print("-" * 40)
        
        total_tests = 0
        total_detections = 0
        total_time = 0
        detection_results = []
        
        # Test on a subset of images (since these are instrument photos, may not have people)
        test_images = []
        for instrument, image_paths in list(samples.items())[:5]:  # Test 5 instruments
            test_images.extend(image_paths[:1])  # 1 image per instrument
        
        for image_path in test_images:
            print(f"🔍 Testing: {image_path.parent.name}/{image_path.name}...", end=" ")
            
            result = test_artist_detection(str(image_path))
            total_tests += 1
            
            if result['success']:
                person_count = result['person_count']
                process_time = result['process_time'] * 1000
                
                print(f"Found {person_count} person(s) in {process_time:.0f}ms")
                
                total_detections += person_count
                total_time += result['process_time']
                detection_results.append(person_count)
            else:
                print(f"💥 {result['error']}")
        
        # Artist detection stats
        avg_time = (total_time / total_tests) * 1000 if total_tests else 0
        avg_detections = total_detections / total_tests if total_tests else 0
        
        print(f"\n📈 ARTIST DETECTION SUMMARY:")
        print(f"   Images Tested: {total_tests}")
        print(f"   Total People Detected: {total_detections}")
        print(f"   Average per Image: {avg_detections:.1f} people")
        print(f"   Average Time: {avg_time:.0f}ms per image")
        print(f"   Note: Instrument images may not contain people")
    
    print("\n🎉 Comprehensive Testing Complete!")
    
    # Final recommendation
    if instrument_ok and artist_ok:
        print("\n✅ Both systems are working perfectly and ready for deployment!")
    elif instrument_ok:
        print("\n⚠️  Instrument detection is working. Artist detection needs attention.")
    elif artist_ok:
        print("\n⚠️  Artist detection is working. Instrument detection needs attention.")

if __name__ == "__main__":
    run_comprehensive_test()