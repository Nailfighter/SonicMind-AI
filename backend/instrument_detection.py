#!/usr/bin/env python3
"""
ðŸŽ¸ SonicMind AI - Instrument Detection System
Real-time instrument recognition using CLIP vision model
"""

import cv2
import numpy as np
import threading
import time
from typing import Callable, Tuple, Optional
from PIL import Image

# CLIP imports
try:
    import clip
    import torch
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False


class InstrumentDetector:
    """Real-time instrument detection using CLIP"""

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.preprocess = None
        self.available = False

        # Camera and threading
        self.camera = None
        self.detection_thread = None
        self.running = False

        # Callback for detection results
        self.callback = None

        # Detection parameters
        self.detection_interval = 1.0  # seconds between detections
        self.confidence_threshold = 0.3

        # Instrument categories with multiple text prompts for better accuracy
        self.instruments = {
            'acoustic_guitar': [
                "a photo of an acoustic guitar",
                "someone playing acoustic guitar",
                "a person holding an acoustic guitar",
                "acoustic guitar being played",
                "a wooden acoustic guitar"
            ],
            'electric_guitar': [
                "a photo of an electric guitar",
                "someone playing electric guitar",
                "a person holding an electric guitar",
                "electric guitar being played",
                "a solid body electric guitar"
            ],
            'bass_guitar': [
                "a photo of a bass guitar",
                "someone playing bass guitar",
                "a person holding a bass guitar",
                "bass guitar being played",
                "a 4-string bass guitar"
            ],
            'piano': [
                "a photo of a piano",
                "someone playing piano",
                "a person at the piano",
                "piano keys being played",
                "a grand piano or upright piano"
            ],
            'keyboard': [
                "a photo of a keyboard",
                "someone playing keyboard",
                "a person at a keyboard",
                "electronic keyboard being played",
                "a synthesizer keyboard"
            ],
            'drums': [
                "a photo of drums",
                "someone playing drums",
                "a person playing drum kit",
                "drums being played",
                "a drum set or drum kit"
            ],
            'violin': [
                "a photo of a violin",
                "someone playing violin",
                "a person holding a violin",
                "violin being played with bow",
                "a classical violin"
            ],
            'cello': [
                "a photo of a cello",
                "someone playing cello",
                "a person playing cello",
                "cello being played with bow",
                "a large string instrument cello"
            ],
            'saxophone': [
                "a photo of a saxophone",
                "someone playing saxophone",
                "a person holding a saxophone",
                "saxophone being played",
                "a brass saxophone instrument"
            ],
            'trumpet': [
                "a photo of a trumpet",
                "someone playing trumpet",
                "a person holding a trumpet",
                "trumpet being played",
                "a brass trumpet instrument"
            ],
            'flute': [
                "a photo of a flute",
                "someone playing flute",
                "a person holding a flute",
                "flute being played",
                "a silver flute instrument"
            ],
            'harmonica': [
                "a photo of a harmonica",
                "someone playing harmonica",
                "a person holding a harmonica",
                "harmonica being played",
                "a small harmonica mouth organ"
            ]
        }

        # Initialize CLIP model
        self._initialize_clip()

        # Pre-compute text features for efficiency
        if self.available:
            self._precompute_text_features()

    def _initialize_clip(self):
        """Initialize CLIP model"""
        if not CLIP_AVAILABLE:
            print("âš ï¸ CLIP not available - instrument detection disabled")
            print("   Install with: pip install git+https://github.com/openai/CLIP.git")
            return

        try:
            print(f"ðŸ” Loading CLIP model on {self.device}...")
            self.model, self.preprocess = clip.load(
                "ViT-B/32", device=self.device)
            self.available = True
            print("âœ… Instrument detector ready")

        except Exception as e:
            print(f"âŒ CLIP model loading failed: {e}")
            print("   Falling back to mock detection")

    def _precompute_text_features(self):
        """Pre-compute text embeddings for all instruments"""
        print("ðŸ§  Pre-computing text features...")
        self.text_features = {}

        with torch.no_grad():
            for instrument, prompts in self.instruments.items():
                # Tokenize all prompts for this instrument
                text_tokens = clip.tokenize(prompts).to(self.device)

                # Encode and normalize
                features = self.model.encode_text(text_tokens)
                features = features / features.norm(dim=-1, keepdim=True)

                # Average the features from all prompts
                avg_features = features.mean(dim=0, keepdim=True)
                avg_features = avg_features / \
                    avg_features.norm(dim=-1, keepdim=True)

                self.text_features[instrument] = avg_features

        print(
            f"âœ… Pre-computed features for {len(self.text_features)} instruments")

    def start_detection(self, camera_index: int = 0) -> bool:
        """Start camera-based instrument detection"""
        if self.running:
            return False

        try:
            # Initialize camera
            self.camera = cv2.VideoCapture(camera_index)
            if not self.camera.isOpened():
                print(f"âŒ Failed to open camera {camera_index}")
                return False

            # Set camera properties for better performance
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, 15)

            # Start detection thread
            self.running = True
            self.detection_thread = threading.Thread(
                target=self._detection_loop, daemon=True)
            self.detection_thread.start()

            print(f"âœ… Instrument detection started on camera {camera_index}")
            return True

        except Exception as e:
            print(f"âŒ Camera initialization failed: {e}")
            return False

    def stop_detection(self):
        """Stop instrument detection"""
        self.running = False

        if self.camera:
            try:
                self.camera.release()
            except Exception:
                pass
            finally:
                self.camera = None

        print("ðŸ›‘ Instrument detection stopped")

    def set_callback(self, callback: Callable[[str, float], None]):
        """Set callback function for detection results"""
        self.callback = callback

    def detect_from_frame(self, frame: np.ndarray) -> Tuple[str, float]:
        """Detect instrument in a single frame"""
        if not self.available:
            return self._mock_detection()

        try:
            # Convert BGR to RGB
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                rgb_frame = frame

            # Convert to PIL Image and preprocess
            pil_image = Image.fromarray(rgb_frame)
            image_input = self.preprocess(
                pil_image).unsqueeze(0).to(self.device)

            # Encode image
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                image_features = image_features / \
                    image_features.norm(dim=-1, keepdim=True)

            # Calculate similarities with all instruments
            best_instrument = "none"
            best_confidence = 0.0

            for instrument, text_features in self.text_features.items():
                similarity = (image_features @
                              text_features.T).squeeze().item()

                # Convert similarity to confidence (sigmoid-like transformation)
                confidence = 1.0 / (1.0 + np.exp(-10 * (similarity - 0.2)))

                if confidence > best_confidence:
                    best_confidence = confidence
                    best_instrument = instrument

            # Only return detection if confidence is above threshold
            if best_confidence >= self.confidence_threshold:
                return best_instrument, best_confidence
            else:
                return "none", best_confidence

        except Exception as e:
            print(f"âš ï¸ Detection error: {e}")
            return "none", 0.0

    def _mock_detection(self) -> Tuple[str, float]:
        """Mock detection for testing when CLIP is unavailable"""
        import random

        # Simulate detection with some probability
        if random.random() < 0.3:  # 30% chance of detection
            instrument = random.choice(list(self.instruments.keys()))
            confidence = random.uniform(0.4, 0.8)
            return instrument, confidence
        else:
            return "none", 0.0

    def _detection_loop(self):
        """Main detection loop running in separate thread"""
        print("ðŸ” Detection loop started")

        last_detection_time = 0
        frame_skip_count = 0

        while self.running:
            try:
                if not self.camera or not self.camera.isOpened():
                    time.sleep(0.1)
                    continue

                # Read frame from camera
                ret, frame = self.camera.read()
                if not ret:
                    print("âš ï¸ Failed to read camera frame")
                    time.sleep(0.1)
                    continue

                # Skip frames to reduce processing load
                frame_skip_count += 1
                # Process every 15th frame (~1 FPS at 15 FPS camera)
                if frame_skip_count < 15:
                    continue
                frame_skip_count = 0

                # Check if enough time has passed since last detection
                current_time = time.time()
                if current_time - last_detection_time < self.detection_interval:
                    continue

                last_detection_time = current_time

                # Detect instrument in frame
                instrument, confidence = self.detect_from_frame(frame)

                # Call callback if instrument detected with sufficient confidence
                if instrument != "none" and confidence >= self.confidence_threshold:
                    if self.callback:
                        self.callback(instrument, confidence)

                    print(f"ðŸŽµ Detected: {instrument} ({confidence:.1%})")

            except Exception as e:
                print(f"âš ï¸ Detection loop error: {e}")
                time.sleep(1.0)

        print("ðŸ›‘ Detection loop stopped")

    def get_supported_instruments(self) -> list:
        """Get list of supported instruments"""
        return list(self.instruments.keys())

    def set_confidence_threshold(self, threshold: float):
        """Set minimum confidence threshold for detections"""
        self.confidence_threshold = max(0.0, min(1.0, threshold))
        print(
            f"ðŸŽ¯ Confidence threshold set to {self.confidence_threshold:.1%}")

    def set_detection_interval(self, interval: float):
        """Set interval between detections in seconds"""
        self.detection_interval = max(0.1, interval)
        print(
            f"â±ï¸ Detection interval set to {self.detection_interval:.1f}s")

    def cleanup(self):
        """Cleanup resources"""
        self.stop_detection()

# ==================== TESTING FUNCTIONS ==================== #


def test_instrument_detection():
    """Test instrument detection with webcam"""
    print("ðŸ§ª Testing Instrument Detection")
    print("Press 'q' to quit, 's' to save frame")

    detector = InstrumentDetector()

    def detection_callback(instrument, confidence):
        print(
            f"ðŸŽµ Callback: {instrument} detected with {confidence:.1%} confidence")

    detector.set_callback(detection_callback)

    if detector.start_detection(0):
        print("Detection started. Monitoring for instruments...")

        try:
            # Let it run for a while
            time.sleep(30)
        except KeyboardInterrupt:
            pass

        detector.stop_detection()
    else:
        print("Failed to start detection")

    detector.cleanup()


def test_single_image(image_path: str):
    """Test detection on a single image file"""
    detector = InstrumentDetector()

    try:
        # Load image
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"âŒ Could not load image: {image_path}")
            return

        # Detect instrument
        instrument, confidence = detector.detect_from_frame(frame)

        print(f"ðŸ“¸ Image: {image_path}")
        print(f"ðŸŽµ Result: {instrument} ({confidence:.1%})")

        # Show supported instruments
        print("\nðŸ“‹ Supported instruments:")
        for inst in detector.get_supported_instruments():
            print(f"   â€¢ {inst}")

    except Exception as e:
        print(f"âŒ Test failed: {e}")

    detector.cleanup()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Test with image file
        test_single_image(sys.argv[1])
    else:
        # Test with webcam
        test_instrument_detection()
