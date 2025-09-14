#!/usr/bin/env python3
"""
ðŸ  SonicMind AI - Material Detection & Room Acoustics Analysis
Analyzes room materials to determine acoustic properties and generate EQ presets
"""

import cv2
import numpy as np
import threading
import time
from typing import Callable, Dict, Optional, List
from PIL import Image

# CLIP imports
try:
    import clip
    import torch
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False


class MaterialDetector:
    """Room material analysis using CLIP vision model"""

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.preprocess = None
        self.available = False

        # Camera and threading
        self.camera = None
        self.analysis_thread = None
        self.running = False
        self.analysis_complete = False

        # Callback for analysis results
        self.callback = None

        # Analysis parameters
        self.analysis_duration = 5.0  # seconds to analyze before concluding
        self.confidence_threshold = 0.4

        # Material categories with acoustic properties
        self.materials = {
            'wood': {
                'prompts': [
                    "a photo of wood texture and grain",
                    "wooden surface with natural grain",
                    "close-up of wood material",
                    "hardwood or softwood surface",
                    "wooden furniture or paneling"
                ],
                'acoustic_properties': {
                    'absorption_coefficient': 0.15,  # Low-medium absorption
                    'reflection_factor': 0.85,
                    'reverb_characteristics': 'warm',
                    'frequency_response': 'balanced'
                }
            },
            'concrete': {
                'prompts': [
                    "a photo of concrete surface",
                    "rough concrete wall or floor",
                    "gray concrete texture",
                    "cement or concrete material",
                    "industrial concrete surface"
                ],
                'acoustic_properties': {
                    'absorption_coefficient': 0.02,  # Very low absorption
                    'reflection_factor': 0.98,
                    'reverb_characteristics': 'bright_harsh',
                    'frequency_response': 'bass_heavy'
                }
            },
            'carpet': {
                'prompts': [
                    "a photo of carpet texture",
                    "soft carpet or rug surface",
                    "fabric carpet flooring",
                    "woven carpet material",
                    "plush carpet texture"
                ],
                'acoustic_properties': {
                    'absorption_coefficient': 0.65,  # High absorption
                    'reflection_factor': 0.35,
                    'reverb_characteristics': 'muffled',
                    'frequency_response': 'high_cut'
                }
            },
            'glass': {
                'prompts': [
                    "a photo of glass surface",
                    "clear glass window or panel",
                    "reflective glass material",
                    "transparent glass surface",
                    "smooth glass texture"
                ],
                'acoustic_properties': {
                    'absorption_coefficient': 0.03,  # Very low absorption
                    'reflection_factor': 0.97,
                    'reverb_characteristics': 'bright_ringing',
                    'frequency_response': 'high_boost'
                }
            },
            'fabric': {
                'prompts': [
                    "a photo of fabric texture",
                    "soft fabric or textile material",
                    "woven fabric surface",
                    "cloth or textile texture",
                    "upholstery fabric material"
                ],
                'acoustic_properties': {
                    'absorption_coefficient': 0.45,  # Medium-high absorption
                    'reflection_factor': 0.55,
                    'reverb_characteristics': 'soft',
                    'frequency_response': 'mid_absorb'
                }
            },
            'metal': {
                'prompts': [
                    "a photo of metal surface",
                    "brushed or polished metal",
                    "metallic surface texture",
                    "steel or aluminum material",
                    "industrial metal surface"
                ],
                'acoustic_properties': {
                    'absorption_coefficient': 0.05,  # Very low absorption
                    'reflection_factor': 0.95,
                    'reverb_characteristics': 'metallic_ringing',
                    'frequency_response': 'mid_high_boost'
                }
            },
            'drywall': {
                'prompts': [
                    "a photo of painted wall",
                    "smooth drywall surface",
                    "white or colored wall paint",
                    "interior wall surface",
                    "painted plaster wall"
                ],
                'acoustic_properties': {
                    'absorption_coefficient': 0.08,  # Low absorption
                    'reflection_factor': 0.92,
                    'reverb_characteristics': 'neutral',
                    'frequency_response': 'relatively_flat'
                }
            },
            'brick': {
                'prompts': [
                    "a photo of brick wall",
                    "red brick surface texture",
                    "brick masonry wall",
                    "textured brick material",
                    "brick wall construction"
                ],
                'acoustic_properties': {
                    'absorption_coefficient': 0.12,  # Low-medium absorption
                    'reflection_factor': 0.88,
                    'reverb_characteristics': 'slightly_warm',
                    'frequency_response': 'mid_emphasis'
                }
            }
        }

        # EQ presets based on room acoustics
        self.room_presets = {
            'live_room': {  # Hard surfaces, lots of reflection
                'description': 'Live room with hard reflective surfaces',
                'eq_adjustments': {
                    'low': -1.0,      # Reduce bass buildup
                    'low_mid': -0.5,  # Slight low-mid cut
                    'mid': 0.0,       # Keep mids natural
                    'high_mid': -1.0,  # Reduce harsh reflections
                    'high': -1.5      # Tame excessive brightness
                }
            },
            'dead_room': {  # Soft surfaces, lots of absorption
                'description': 'Dead room with absorptive surfaces',
                'eq_adjustments': {
                    'low': 1.0,       # Add bass presence
                    'low_mid': 0.5,   # Boost low-mids
                    'mid': 1.0,       # Boost mids for presence
                    'high_mid': 1.5,  # Add high-mid clarity
                    'high': 2.0       # Add air and sparkle
                }
            },
            'balanced_room': {  # Mixed surfaces
                'description': 'Balanced room with mixed acoustic properties',
                'eq_adjustments': {
                    'low': 0.0,       # Neutral bass
                    'low_mid': 0.2,   # Slight warmth
                    'mid': 0.0,       # Natural mids
                    'high_mid': 0.3,  # Slight presence boost
                    'high': 0.0       # Natural highs
                }
            },
            'bass_heavy': {  # Small room or corner placement
                'description': 'Room with bass buildup issues',
                'eq_adjustments': {
                    'low': -2.0,      # Significant bass cut
                    'low_mid': -1.0,  # Low-mid reduction
                    'mid': 0.5,       # Compensate with mid boost
                    'high_mid': 0.8,  # Add clarity
                    'high': 0.5       # Slight high boost
                }
            },
            'bright_harsh': {  # Lots of hard surfaces
                'description': 'Bright room with harsh reflections',
                'eq_adjustments': {
                    'low': 0.5,       # Add warmth
                    'low_mid': 0.8,   # Boost low-mids
                    'mid': 0.0,       # Keep mids neutral
                    'high_mid': -1.5,  # Reduce harshness
                    'high': -2.0      # Significant high cut
                }
            }
        }

        # Initialize CLIP model
        self._initialize_clip()

        # Pre-compute text features
        if self.available:
            self._precompute_text_features()

    def _initialize_clip(self):
        """Initialize CLIP model for material analysis"""
        if not CLIP_AVAILABLE:
            print("âš ï¸ CLIP not available - material detection disabled")
            print("   Install with: pip install git+https://github.com/openai/CLIP.git")
            return

        try:
            print(
                f"ðŸ” Loading CLIP model for material analysis on {self.device}...")
            self.model, self.preprocess = clip.load(
                "ViT-B/32", device=self.device)
            self.available = True
            print("âœ… Material detector ready")

        except Exception as e:
            print(f"âŒ CLIP model loading failed: {e}")
            print("   Falling back to mock analysis")

    def _precompute_text_features(self):
        """Pre-compute text embeddings for all materials"""
        print("ðŸ§  Pre-computing material text features...")
        self.text_features = {}

        with torch.no_grad():
            for material, data in self.materials.items():
                prompts = data['prompts']

                # Tokenize all prompts for this material
                text_tokens = clip.tokenize(prompts).to(self.device)

                # Encode and normalize
                features = self.model.encode_text(text_tokens)
                features = features / features.norm(dim=-1, keepdim=True)

                # Average the features from all prompts
                avg_features = features.mean(dim=0, keepdim=True)
                avg_features = avg_features / \
                    avg_features.norm(dim=-1, keepdim=True)

                self.text_features[material] = avg_features

        print(
            f"âœ… Pre-computed features for {len(self.text_features)} materials")

    def start_detection(self, camera_index: int = 0) -> bool:
        """Start camera-based room analysis"""
        if self.running:
            return False

        try:
            # Initialize camera
            self.camera = cv2.VideoCapture(camera_index)
            if not self.camera.isOpened():
                print(f"âŒ Failed to open camera {camera_index}")
                return False

            # Set camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, 10)

            # Start analysis thread
            self.running = True
            self.analysis_complete = False
            self.analysis_thread = threading.Thread(
                target=self._analysis_loop, daemon=True)
            self.analysis_thread.start()

            print(f"âœ… Material analysis started on camera {camera_index}")
            return True

        except Exception as e:
            print(f"âŒ Camera initialization failed: {e}")
            return False

    def stop_detection(self):
        """Stop material analysis"""
        self.running = False

        if self.camera:
            try:
                self.camera.release()
            except Exception:
                pass
            finally:
                self.camera = None

        print("ðŸ›‘ Material analysis stopped")

    def set_callback(self, callback: Callable[[Dict], None]):
        """Set callback function for analysis results"""
        self.callback = callback

    def analyze_frame(self, frame: np.ndarray) -> Dict[str, float]:
        """Analyze materials in a single frame"""
        if not self.available:
            return self._mock_material_analysis()

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

            # Calculate similarities with all materials
            material_scores = {}

            for material, text_features in self.text_features.items():
                similarity = (image_features @
                              text_features.T).squeeze().item()

                # Convert similarity to confidence
                confidence = 1.0 / (1.0 + np.exp(-8 * (similarity - 0.1)))
                material_scores[material] = confidence

            return material_scores

        except Exception as e:
            print(f"âš ï¸ Material analysis error: {e}")
            return self._mock_material_analysis()

    def _mock_material_analysis(self) -> Dict[str, float]:
        """Mock material analysis for testing"""
        import random

        # Generate random but somewhat realistic material scores
        materials = list(self.materials.keys())
        scores = {}

        # Pick 2-3 dominant materials
        dominant = random.sample(materials, random.randint(2, 3))

        for material in materials:
            if material in dominant:
                scores[material] = random.uniform(0.4, 0.8)
            else:
                scores[material] = random.uniform(0.1, 0.3)

        return scores

    def _analyze_room_acoustics(self, material_samples: List[Dict[str, float]]) -> Dict:
        """Analyze room acoustics based on multiple material samples"""
        if not material_samples:
            return self._default_room_analysis()

        # Average material scores across all samples
        avg_scores = {}
        for material in self.materials.keys():
            scores = [sample.get(material, 0.0) for sample in material_samples]
            avg_scores[material] = np.mean(scores)

        # Find dominant materials (above threshold)
        dominant_materials = {
            material: score for material, score in avg_scores.items()
            if score >= self.confidence_threshold
        }

        if not dominant_materials:
            # Fallback to highest scoring material
            best_material = max(avg_scores.items(), key=lambda x: x[1])
            dominant_materials = {best_material[0]: best_material[1]}

        # Calculate overall acoustic properties
        total_absorption = 0.0
        total_reflection = 0.0
        acoustic_characteristics = []

        for material, confidence in dominant_materials.items():
            props = self.materials[material]['acoustic_properties']
            weight = confidence

            total_absorption += props['absorption_coefficient'] * weight
            total_reflection += props['reflection_factor'] * weight
            acoustic_characteristics.append(props['reverb_characteristics'])

        # Normalize by total confidence
        total_confidence = sum(dominant_materials.values())
        avg_absorption = total_absorption / total_confidence
        avg_reflection = total_reflection / total_confidence

        # Determine room type and EQ preset
        room_type, eq_preset = self._determine_room_preset(
            avg_absorption, acoustic_characteristics)

        # Get the most confident material as primary
        primary_material = max(dominant_materials.items(), key=lambda x: x[1])

        return {
            'dominant_material': primary_material[0],
            'material_confidence': primary_material[1],
            'all_materials': avg_scores,
            'dominant_materials': dominant_materials,
            'acoustic_properties': {
                'average_absorption': avg_absorption,
                'average_reflection': avg_reflection,
                'room_type': room_type,
                'reverb_characteristics': acoustic_characteristics,
                'preset_name': room_type
            },
            'eq_preset': eq_preset,
            'timestamp': time.time()
        }

    def _determine_room_preset(self, avg_absorption: float, characteristics: List[str]) -> tuple:
        """Determine appropriate room preset based on acoustic properties"""

        # Categorize based on absorption coefficient
        if avg_absorption >= 0.4:  # High absorption
            room_type = 'dead_room'
        elif avg_absorption <= 0.1:  # Low absorption
            if 'bright_harsh' in characteristics or 'metallic_ringing' in characteristics:
                room_type = 'bright_harsh'
            elif 'bass_heavy' in characteristics:
                room_type = 'bass_heavy'
            else:
                room_type = 'live_room'
        else:  # Medium absorption
            room_type = 'balanced_room'

        eq_preset = self.room_presets.get(
            room_type, self.room_presets['balanced_room'])

        return room_type, eq_preset

    def _default_room_analysis(self) -> Dict:
        """Default room analysis when detection fails"""
        return {
            'dominant_material': 'drywall',
            'material_confidence': 0.5,
            'all_materials': {material: 0.2 for material in self.materials.keys()},
            'dominant_materials': {'drywall': 0.5},
            'acoustic_properties': {
                'average_absorption': 0.08,
                'average_reflection': 0.92,
                'room_type': 'balanced_room',
                'reverb_characteristics': ['neutral'],
                'preset_name': 'balanced_room'
            },
            'eq_preset': self.room_presets['balanced_room'],
            'timestamp': time.time()
        }

    def _analysis_loop(self):
        """Main analysis loop - analyzes room for set duration"""
        print("ðŸ” Room analysis loop started")

        material_samples = []
        analysis_start_time = time.time()
        frame_count = 0

        while self.running and not self.analysis_complete:
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

                frame_count += 1

                # Analyze every 10th frame to reduce load
                if frame_count % 10 == 0:
                    material_scores = self.analyze_frame(frame)
                    material_samples.append(material_scores)

                # Check if analysis duration has elapsed
                elapsed_time = time.time() - analysis_start_time
                if elapsed_time >= self.analysis_duration:
                    print(
                        f"ðŸ  Room analysis complete after {elapsed_time:.1f}s")
                    break

                time.sleep(0.1)  # Small delay between frames

            except Exception as e:
                print(f"âš ï¸ Analysis loop error: {e}")
                time.sleep(1.0)

        # Generate final room analysis
        if material_samples:
            room_analysis = self._analyze_room_acoustics(material_samples)

            # Call callback with results
            if self.callback:
                self.callback(room_analysis)

            self.analysis_complete = True

            print(f"ðŸ  Room analysis results:")
            print(
                f"   Primary material: {room_analysis['dominant_material']} ({room_analysis['material_confidence']:.1%})")
            print(
                f"   Room type: {room_analysis['acoustic_properties']['room_type']}")
            print(
                f"   Average absorption: {room_analysis['acoustic_properties']['average_absorption']:.2f}")

        else:
            print("âš ï¸ No material samples collected")
            if self.callback:
                self.callback(self._default_room_analysis())

        print("ðŸ›‘ Analysis loop stopped")

    def get_supported_materials(self) -> List[str]:
        """Get list of supported materials"""
        return list(self.materials.keys())

    def get_room_presets(self) -> Dict:
        """Get available room presets"""
        return self.room_presets.copy()

    def set_analysis_duration(self, duration: float):
        """Set analysis duration in seconds"""
        self.analysis_duration = max(1.0, duration)
        print(f"â±ï¸ Analysis duration set to {self.analysis_duration:.1f}s")

    def set_confidence_threshold(self, threshold: float):
        """Set minimum confidence threshold"""
        self.confidence_threshold = max(0.0, min(1.0, threshold))
        print(
            f"ðŸŽ¯ Confidence threshold set to {self.confidence_threshold:.1%}")

    def cleanup(self):
        """Cleanup resources"""
        self.stop_detection()

# ==================== TESTING FUNCTIONS ==================== #


def test_material_detection():
    """Test material detection with webcam"""
    print("ðŸ§ª Testing Material Detection")
    print("Camera will analyze room for 10 seconds...")

    detector = MaterialDetector()

    def analysis_callback(analysis):
        print("\nðŸ  Room Analysis Results:")
        print(
            f"   Primary Material: {analysis['dominant_material']} ({analysis['material_confidence']:.1%})")
        print(f"   Room Type: {analysis['acoustic_properties']['room_type']}")
        print(
            f"   Absorption Coefficient: {analysis['acoustic_properties']['average_absorption']:.3f}")
        print(f"   EQ Preset: {analysis['eq_preset']['description']}")

        print("\nðŸŽ›ï¸ Suggested EQ Adjustments:")
        for band, adjustment in analysis['eq_preset']['eq_adjustments'].items():
            print(f"   {band}: {adjustment:+.1f} dB")

        print("\nðŸ“Š All Material Scores:")
        for material, score in sorted(analysis['all_materials'].items(), key=lambda x: x[1], reverse=True):
            print(f"   {material}: {score:.1%}")

    detector.set_callback(analysis_callback)
    detector.set_analysis_duration(10.0)  # 10 second analysis

    if detector.start_detection(0):
        print("Analysis started. Point camera around the room...")

        # Wait for analysis to complete
        while detector.running and not detector.analysis_complete:
            time.sleep(1)

        time.sleep(2)  # Brief pause to see results
        detector.stop_detection()
    else:
        print("Failed to start analysis")

    detector.cleanup()


def test_single_image(image_path: str):
    """Test material analysis on a single image"""
    detector = MaterialDetector()

    try:
        # Load image
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"âŒ Could not load image: {image_path}")
            return

        # Analyze materials
        material_scores = detector.analyze_frame(frame)

        print(f"ðŸ“¸ Image: {image_path}")
        print("ðŸ  Material Analysis:")

        # Sort by confidence
        sorted_materials = sorted(
            material_scores.items(), key=lambda x: x[1], reverse=True)

        for material, score in sorted_materials:
            print(f"   {material}: {score:.1%}")

        # Show supported materials
        print(
            f"\nðŸ“‹ Supported materials: {', '.join(detector.get_supported_materials())}")

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
        test_material_detection()
