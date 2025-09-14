#!/usr/bin/env python3
"""
SonicMind-AI Audio Analysis Formatter
=====================================
Converts AI model output into structured JSON format for audio analysis
Designed for hackathon speed - focuses on practical audio engineering insights
"""

import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
import os
import sys

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from gui import Analyzer, ModelWrapper, DynamicEQModel
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch or gui modules not available. Running in demo mode.")

class AudioAnalysisFormatter:
    """Formats AI model output into structured audio analysis JSON"""
    
    def __init__(self, sample_rate=44100):
        self.sr = sample_rate
        self.analyzer = Analyzer(sample_rate) if 'Analyzer' in globals() else None
        
        # Audio frequency band definitions (Hz)
        self.freq_bands = {
            "sub_bass": (20, 60),
            "bass": (60, 250), 
            "low_mids": (250, 500),
            "mids": (500, 2000),
            "high_mids": (2000, 4000),
            "presence": (4000, 6000),
            "brilliance": (6000, 20000)
        }
        
        # Audio problem detection thresholds
        self.problem_thresholds = {
            "muddy": {"band": "bass", "threshold": 0.35},
            "thin": {"band": "bass", "threshold": 0.15},
            "boxy": {"band": "low_mids", "threshold": 0.25},
            "harsh": {"band": "high_mids", "threshold": 0.25},
            "sibilant": {"band": "presence", "threshold": 0.20},
            "dull": {"band": "brilliance", "threshold": 0.10}
        }

    def analyze_audio_clip(self, audio_data: np.ndarray, model_output: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Analyze audio clip and return structured JSON
        
        Args:
            audio_data: Raw audio samples (numpy array)
            model_output: Optional AI model predictions (15 values: 5 bands × 3 params)
        
        Returns:
            Structured dictionary with audio analysis
        """
        
        # Generate analysis timestamp
        timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
        
        # Extract basic features
        if self.analyzer:
            features = self.analyzer.features(audio_data)
        else:
            # Fallback feature extraction
            features = self._extract_features_fallback(audio_data)
        
        # Analyze frequency balance
        freq_analysis = self._analyze_frequency_balance(features, audio_data)
        
        # Detect audio problems
        audio_problems = self._detect_audio_problems(freq_analysis)
        
        # Process AI model output if available
        ai_suggestions = self._process_model_output(model_output) if model_output is not None else None
        
        # Calculate confidence metrics
        confidence_metrics = self._calculate_confidence(features, audio_problems)
        
        # Build the structured output
        analysis = {
            "analysis_id": f"audio_analysis_{timestamp}",
            "timestamp": timestamp,
            "confidence_overall": confidence_metrics["overall"],
            
            "audio_characteristics": {
                "duration_s": len(audio_data) / self.sr,
                "sample_rate": self.sr,
                "rms_db": self._linear_to_db(features["rms"]),
                "peak_db": self._linear_to_db(features["peak"]),
                "crest_factor_db": self._linear_to_db(features["peak"] / (features["rms"] + 1e-8)),
                "dynamic_range": self._classify_dynamic_range(features)
            },
            
            "frequency_analysis": {
                "balance": freq_analysis,
                "tonal_content": {
                    "tonality_index": features.get("tonal", 0.0),
                    "classification": self._classify_tonality(features.get("tonal", 0.0))
                }
            },
            
            "detected_problems": audio_problems,
            
            "eq_recommendations": {
                "urgency": self._calculate_eq_urgency(audio_problems),
                "suggested_corrections": self._generate_eq_suggestions(audio_problems),
                "ai_model_active": model_output is not None,
                "ai_predictions": ai_suggestions
            },
            
            "mix_quality_assessment": {
                "overall_grade": self._grade_mix_quality(audio_problems, features),
                "problem_count": len([p for p in audio_problems if p["severity"] > 0.5]),
                "frequency_balance_score": confidence_metrics["frequency_balance"],
                "dynamic_health_score": confidence_metrics["dynamics"]
            },
            
            "real_time_metrics": {
                "processing_suitable": len(audio_data) >= self.sr,  # At least 1 second
                "stability_index": confidence_metrics["stability"],
                "recommended_action": self._recommend_action(audio_problems, features)
            }
        }
        
        return analysis

    def _extract_features_fallback(self, audio_data: np.ndarray) -> Dict[str, float]:
        """Fallback feature extraction when Analyzer class not available"""
        if len(audio_data) == 0:
            return {"rms": 0, "peak": 0, "bass": 0, "mids": 0, "highs": 0, "total": 1, "tonal": 0.0}
        
        # Basic time domain features
        rms = float(np.sqrt(np.mean(audio_data ** 2)))
        peak = float(np.max(np.abs(audio_data)))
        
        # Simple frequency analysis
        fft = np.fft.rfft(audio_data)
        mag = np.abs(fft)
        freqs = np.fft.rfftfreq(len(audio_data), 1/self.sr)
        
        # Band energies
        bass = float(mag[(freqs >= 20) & (freqs < 250)].sum())
        mids = float(mag[(freqs >= 250) & (freqs < 4000)].sum())
        highs = float(mag[(freqs >= 4000) & (freqs < 20000)].sum())
        total = bass + mids + highs + 1e-8
        
        # Tonality measure
        tonal = float(mag.max() / (mag.sum() + 1e-8))
        
        return {
            "rms": rms, "peak": peak, "bass": bass, "mids": mids, 
            "highs": highs, "total": total, "tonal": tonal
        }

    def _analyze_frequency_balance(self, features: Dict, audio_data: np.ndarray) -> Dict[str, Any]:
        """Analyze frequency balance across spectrum"""
        total = features["total"]
        
        balance = {
            "bass_ratio": features["bass"] / total,
            "mids_ratio": features["mids"] / total, 
            "highs_ratio": features["highs"] / total,
            "balance_classification": ""
        }
        
        # Classify overall balance
        if balance["bass_ratio"] > 0.4:
            balance["balance_classification"] = "bass_heavy"
        elif balance["highs_ratio"] > 0.35:
            balance["balance_classification"] = "bright"
        elif balance["mids_ratio"] > 0.6:
            balance["balance_classification"] = "mid_focused"
        else:
            balance["balance_classification"] = "balanced"
            
        return balance

    def _detect_audio_problems(self, freq_analysis: Dict) -> List[Dict[str, Any]]:
        """Detect common audio problems based on frequency analysis"""
        problems = []
        
        bass_ratio = freq_analysis["bass_ratio"]
        mids_ratio = freq_analysis["mids_ratio"] 
        highs_ratio = freq_analysis["highs_ratio"]
        
        # Check for common problems
        if bass_ratio > 0.35:
            problems.append({
                "problem": "muddy",
                "description": "Excessive bass energy making mix unclear",
                "severity": min((bass_ratio - 0.25) * 2, 1.0),
                "frequency_range": "60-250 Hz",
                "suggested_action": "reduce_bass"
            })
            
        if bass_ratio < 0.15:
            problems.append({
                "problem": "thin",
                "description": "Insufficient bass foundation",
                "severity": min((0.25 - bass_ratio) * 2, 1.0),
                "frequency_range": "60-250 Hz", 
                "suggested_action": "boost_bass"
            })
            
        if mids_ratio > 0.65:
            problems.append({
                "problem": "boxy",
                "description": "Overpowering midrange frequencies",
                "severity": min((mids_ratio - 0.55) * 3, 1.0),
                "frequency_range": "250-2000 Hz",
                "suggested_action": "reduce_mids"
            })
            
        if highs_ratio > 0.35:
            problems.append({
                "problem": "harsh",
                "description": "Excessive high frequency content",
                "severity": min((highs_ratio - 0.25) * 2.5, 1.0),
                "frequency_range": "4000+ Hz",
                "suggested_action": "reduce_highs"
            })
            
        if highs_ratio < 0.12:
            problems.append({
                "problem": "dull", 
                "description": "Lack of high frequency clarity",
                "severity": min((0.20 - highs_ratio) * 3, 1.0),
                "frequency_range": "6000+ Hz",
                "suggested_action": "boost_highs"
            })
            
        return problems

    def _process_model_output(self, model_output: np.ndarray) -> Dict[str, Any]:
        """Process AI model predictions into readable format"""
        if len(model_output) != 15:
            return {"error": "Invalid model output length"}
        
        # Reshape to 5 bands × 3 parameters
        bands = model_output.reshape(5, 3)
        
        band_names = ["80Hz", "300Hz", "1kHz", "4kHz", "10kHz"]
        predictions = []
        
        for i, (freq, q, gain) in enumerate(bands):
            predictions.append({
                "band": i + 1,
                "name": band_names[i],
                "frequency_hz": float(freq),
                "q_factor": float(q),
                "gain_db": float(gain),
                "action": "boost" if gain > 0.1 else "cut" if gain < -0.1 else "neutral"
            })
        
        return {
            "predictions": predictions,
            "max_correction_db": float(np.max(np.abs(bands[:, 2]))),
            "total_corrections": int(np.sum(np.abs(bands[:, 2]) > 0.1))
        }

    def _calculate_confidence(self, features: Dict, problems: List) -> Dict[str, float]:
        """Calculate confidence metrics for analysis"""
        
        # Overall confidence based on signal level and problems detected
        signal_confidence = min(features["rms"] * 10, 1.0)  # Assume good signal > 0.1 RMS
        problem_confidence = max(0.3, 1.0 - len(problems) * 0.15)
        
        overall = (signal_confidence + problem_confidence) / 2
        
        return {
            "overall": round(overall, 2),
            "signal_level": round(signal_confidence, 2),
            "frequency_balance": round(problem_confidence, 2),
            "dynamics": round(min(features["peak"] / (features["rms"] + 1e-8) / 20, 1.0), 2),
            "stability": round(0.7 + np.random.random() * 0.2, 2)  # Placeholder
        }

    def _classify_dynamic_range(self, features: Dict) -> str:
        """Classify dynamic range of audio"""
        crest_factor = features["peak"] / (features["rms"] + 1e-8)
        
        if crest_factor > 10:
            return "high_dynamic_range"
        elif crest_factor > 4:
            return "moderate_dynamic_range" 
        else:
            return "compressed"

    def _classify_tonality(self, tonal_value: float) -> str:
        """Classify tonality of audio content"""
        if tonal_value > 0.6:
            return "highly_tonal"
        elif tonal_value > 0.3:
            return "moderately_tonal"
        else:
            return "broadband"

    def _calculate_eq_urgency(self, problems: List) -> str:
        """Calculate urgency level for EQ corrections"""
        if not problems:
            return "low"
        
        max_severity = max(p["severity"] for p in problems)
        
        if max_severity > 0.8:
            return "high"
        elif max_severity > 0.5:
            return "medium"
        else:
            return "low"

    def _generate_eq_suggestions(self, problems: List) -> List[Dict[str, Any]]:
        """Generate EQ correction suggestions"""
        suggestions = []
        
        for problem in problems:
            if problem["severity"] > 0.3:
                suggestions.append({
                    "problem": problem["problem"],
                    "action": problem["suggested_action"],
                    "frequency_range": problem["frequency_range"],
                    "estimated_correction_db": round(problem["severity"] * -6, 1),
                    "priority": "high" if problem["severity"] > 0.7 else "medium"
                })
        
        return suggestions

    def _grade_mix_quality(self, problems: List, features: Dict) -> str:
        """Grade overall mix quality"""
        problem_penalty = sum(p["severity"] for p in problems)
        signal_quality = min(features["rms"] * 5, 1.0)
        
        score = signal_quality - problem_penalty * 0.3
        
        if score > 0.8:
            return "excellent"
        elif score > 0.6:
            return "good"
        elif score > 0.4:
            return "fair"
        else:
            return "needs_work"

    def _recommend_action(self, problems: List, features: Dict) -> str:
        """Recommend next action"""
        if not problems:
            return "monitor_only"
        
        urgent_problems = [p for p in problems if p["severity"] > 0.7]
        
        if urgent_problems:
            return "immediate_eq_correction"
        elif len(problems) > 2:
            return "comprehensive_eq_review"
        else:
            return "minor_eq_adjustments"

    def _linear_to_db(self, linear_value: float) -> float:
        """Convert linear amplitude to dB"""
        return round(20 * np.log10(max(abs(linear_value), 1e-8)), 2)

def demo_analysis():
    """Demo function showing how to use the analyzer"""
    print("# SonicMind-AI")
    print("Real Time Live Sound Engineering Assistant\n")
    
    # Create sample audio data (sine wave for demo)
    sample_rate = 44100
    duration = 3.0  # 3 seconds
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Mix of frequencies to simulate realistic audio
    audio = (0.3 * np.sin(2 * np.pi * 100 * t) +    # Bass
             0.5 * np.sin(2 * np.pi * 440 * t) +     # Mid (A4)
             0.2 * np.sin(2 * np.pi * 2000 * t) +    # High mid
             0.1 * np.random.normal(0, 0.02, len(t))) # Noise
    
    # Simulate AI model output (15 values)
    model_output = np.array([
        80,   1.0, -2.1,  # Band 1: 80Hz, Q=1.0, Gain=-2.1dB
        300,  1.2,  0.8,  # Band 2: 300Hz, Q=1.2, Gain=+0.8dB  
        1000, 1.5, -0.3,  # Band 3: 1kHz, Q=1.5, Gain=-0.3dB
        4000, 1.1,  1.2,  # Band 4: 4kHz, Q=1.1, Gain=+1.2dB
        10000, 0.8, -0.5  # Band 5: 10kHz, Q=0.8, Gain=-0.5dB
    ])
    
    # Analyze audio
    formatter = AudioAnalysisFormatter(sample_rate)
    analysis = formatter.analyze_audio_clip(audio, model_output)
    
    # Output formatted JSON
    print(json.dumps(analysis, indent=2))

if __name__ == "__main__":
    demo_analysis()