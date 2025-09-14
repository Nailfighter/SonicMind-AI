#!/usr/bin/env python3
"""
Zero-Shot CLIP Instrument Detector - Hackathon Speed Run
No training needed - uses CLIP's existing knowledge!
"""

import torch
import clip
from PIL import Image
import numpy as np
from typing import Dict, List, Tuple
import time

class ZeroShotInstrumentDetector:
    """Fast instrument detection using pre-trained CLIP - no training needed!"""
    
    def __init__(self, device=None):
        """Initialize with pre-trained CLIP model."""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading CLIP model on {self.device}...")
        
        # Load the model - ViT-B/32 is fast and good
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        
        # Define instruments
        self.instruments = [
            'acoustic_guitar', 'electric_guitar', 'bass_guitar',
            'piano', 'keyboard', 'synthesizer',
            'drums', 'drum_kit', 'djembe', 'tabla',
            'violin', 'cello', 'viola',
            'saxophone', 'trumpet', 'trombone', 'flute',
            'harmonica', 'accordion', 'banjo', 'sitar', 'ukulele'
        ]
        
        # Create multiple prompts per instrument for better accuracy
        self.prompts = self._create_prompts()
        
        # Pre-compute text features (do this once, use forever!)
        print("Pre-computing text features...")
        self.text_features = self._precompute_text_features()
        print(f"Ready! Cached {len(self.text_features)} text encodings")
        
    def _create_prompts(self) -> Dict[str, List[str]]:
        """Create multiple prompts per instrument for better accuracy."""
        prompts = {}
        
        for instrument in self.instruments:
            # Clean instrument name
            clean_name = instrument.replace('_', ' ')
            
            # Multiple prompts increase accuracy without training!
            prompts[instrument] = [
                f"a photo of a {clean_name}",
                f"a {clean_name} musical instrument",
                f"someone playing {clean_name}",
                f"a person playing {clean_name}",
                f"a musician with a {clean_name}",
                f"{clean_name} being played",
                f"a {clean_name} in a concert",
                f"close-up of a {clean_name}",
            ]
        
        return prompts
    
    def _precompute_text_features(self) -> Dict[str, torch.Tensor]:
        """Pre-compute all text features - massive speedup!"""
        text_features = {}
        
        with torch.no_grad():
            for instrument, prompt_list in self.prompts.items():
                # Tokenize all prompts for this instrument
                text_tokens = clip.tokenize(prompt_list).to(self.device)
                
                # Encode and normalize
                features = self.model.encode_text(text_tokens)
                features = features / features.norm(dim=-1, keepdim=True)
                
                # Store the mean of all prompt features
                text_features[instrument] = features.mean(dim=0, keepdim=True)
                text_features[instrument] /= text_features[instrument].norm()
        
        return text_features
    
    def detect(self, image_path: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Detect instruments in image - FAST!
        
        Args:
            image_path: Path to image file
            top_k: Return top K predictions
            
        Returns:
            List of (instrument, confidence) tuples
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # Encode image
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Compare with all instrument text features
        similarities = {}
        for instrument, text_feat in self.text_features.items():
            similarity = (image_features @ text_feat.T).squeeze().item()
            similarities[instrument] = similarity
        
        # Sort by similarity
        sorted_instruments = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        
        # Convert to probabilities using softmax
        scores = torch.tensor([s for _, s in sorted_instruments])
        probs = torch.softmax(scores * 100, dim=0).numpy()  # Scale up for sharper distribution
        
        # Return top K with probabilities
        results = []
        for i in range(min(top_k, len(sorted_instruments))):
            instrument = sorted_instruments[i][0]
            confidence = float(probs[i])
            results.append((instrument, confidence))
        
        return results
    
    def detect_batch(self, image_paths: List[str]) -> List[List[Tuple[str, float]]]:
        """Process multiple images efficiently."""
        results = []
        for path in image_paths:
            results.append(self.detect(path))
        return results

def quick_test():
    """Quick test to make sure it works!"""
    print("\n" + "="*50)
    print("ZERO-SHOT INSTRUMENT DETECTOR TEST")
    print("="*50)
    
    # Initialize detector
    detector = ZeroShotInstrumentDetector()
    
    # Test with a dummy image (you can replace with actual path)
    import os
    
    # Find a test image
    test_images = []
    if os.path.exists("instrument_dataset/test"):
        for instrument_dir in os.listdir("instrument_dataset/test"):
            instrument_path = os.path.join("instrument_dataset/test", instrument_dir)
            if os.path.isdir(instrument_path):
                images = [f for f in os.listdir(instrument_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
                if images:
                    test_images.append(os.path.join(instrument_path, images[0]))
                    if len(test_images) >= 5:  # Test 5 images
                        break
    
    if test_images:
        print(f"\nTesting with {len(test_images)} images...")
        for img_path in test_images:
            start = time.time()
            results = detector.detect(img_path, top_k=3)
            elapsed = time.time() - start
            
            true_instrument = os.path.basename(os.path.dirname(img_path))
            print(f"\nImage: {true_instrument}")
            print(f"Predictions (in {elapsed:.3f}s):")
            for instrument, confidence in results:
                print(f"  {instrument}: {confidence:.2%}")
                if instrument == true_instrument:
                    print(f"  ✓ CORRECT!")
    else:
        print("No test images found. Add images to test!")
    
    print("\n✅ Detector ready for deployment!")
    return detector

if __name__ == "__main__":
    detector = quick_test()
    print("\nTo use in your code:")
    print("  detector = ZeroShotInstrumentDetector()")
    print("  results = detector.detect('path/to/image.jpg')")
    print("  print(results)  # [(instrument, confidence), ...]")