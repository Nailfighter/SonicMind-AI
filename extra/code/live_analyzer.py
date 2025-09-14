#!/usr/bin/env python3
"""
Live Audio Analyzer Integration
==============================
Connects the SonicMind-AI model with the JSON formatter for real-time analysis
Perfect for hackathon demos - fast and practical
"""

import json
import numpy as np
from datetime import datetime
from audio_analyzer import AudioAnalysisFormatter
import time
import os

try:
    from gui import LiveAutoEQApp, Analyzer, ModelWrapper, AudioEngine
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("Audio modules not available - running in demo mode")

class LiveAudioAnalyzer:
    """Real-time audio analysis with JSON output"""
    
    def __init__(self, sample_rate=44100, model_path=None):
        self.sr = sample_rate
        self.formatter = AudioAnalysisFormatter(sample_rate)
        self.analyzer = Analyzer(sample_rate) if AUDIO_AVAILABLE else None
        
        # Try to load AI model
        self.model = None
        if model_path and os.path.exists(model_path):
            try:
                self.model = ModelWrapper(sample_rate, model_path)
                print(f"✅ AI Model loaded: {self.model.status}")
            except Exception as e:
                print(f"❌ Failed to load AI model: {e}")
        else:
            print("🔄 No AI model found - using rule-based analysis only")
    
    def analyze_live_audio(self, duration_seconds=5, output_file=None):
        """
        Capture and analyze live audio
        
        Args:
            duration_seconds: How long to record audio
            output_file: Optional file to save JSON output
        """
        
        if not AUDIO_AVAILABLE:
            print("⚠️ Audio not available - generating demo analysis")
            return self._demo_analysis()
        
        print(f"🎤 Recording {duration_seconds} seconds of audio...")
        
        try:
            # Record audio
            audio_data = sd.rec(
                int(self.sr * duration_seconds), 
                samplerate=self.sr, 
                channels=1, 
                dtype='float32'
            )
            sd.wait()  # Wait for recording to complete
            
            # Convert to 1D array
            audio_data = audio_data.flatten()
            
            print("🔍 Analyzing audio...")
            
            # Get AI predictions if model available
            model_output = None
            if self.model and self.model.available:
                model_output = self.model.predict_params(audio_data)
                print(f"🤖 AI model made {len(model_output)} predictions")
            
            # Format analysis
            analysis = self.formatter.analyze_audio_clip(audio_data, model_output)
            
            # Save to file if requested
            if output_file:
                with open(output_file, 'w') as f:
                    json.dump(analysis, f, indent=2)
                print(f"💾 Analysis saved to {output_file}")
            
            return analysis
            
        except Exception as e:
            print(f"❌ Error during live analysis: {e}")
            return self._demo_analysis()
    
    def analyze_audio_file(self, file_path, output_file=None):
        """
        Analyze audio from file
        
        Args:
            file_path: Path to audio file
            output_file: Optional file to save JSON output
        """
        try:
            import librosa
            
            print(f"📁 Loading audio file: {file_path}")
            audio_data, _ = librosa.load(file_path, sr=self.sr, mono=True, duration=10)
            
            print("🔍 Analyzing audio...")
            
            # Get AI predictions if model available
            model_output = None
            if self.model and self.model.available:
                model_output = self.model.predict_params(audio_data)
                print(f"🤖 AI model made predictions")
            
            # Format analysis
            analysis = self.formatter.analyze_audio_clip(audio_data, model_output)
            
            # Save to file if requested
            if output_file:
                with open(output_file, 'w') as f:
                    json.dump(analysis, f, indent=2)
                print(f"💾 Analysis saved to {output_file}")
            
            return analysis
            
        except ImportError:
            print("❌ librosa not installed - cannot load audio files")
            return self._demo_analysis()
        except Exception as e:
            print(f"❌ Error analyzing file: {e}")
            return self._demo_analysis()
    
    def _demo_analysis(self):
        """Generate demo analysis for when audio is not available"""
        
        # Create demo audio data
        duration = 3.0
        t = np.linspace(0, duration, int(self.sr * duration))
        
        # Simulate problematic audio (bass heavy mix)
        audio = (0.6 * np.sin(2 * np.pi * 80 * t) +      # Heavy bass
                 0.3 * np.sin(2 * np.pi * 440 * t) +      # Mid
                 0.1 * np.sin(2 * np.pi * 4000 * t) +     # Reduced highs
                 0.05 * np.random.normal(0, 0.01, len(t)))  # Noise
        
        # Simulate AI model correction suggestions
        model_output = np.array([
            80,   1.2, -3.2,   # Reduce bass
            300,  1.0,  0.5,   # Slight mid boost
            1000, 1.4, -0.8,   # Reduce low mids
            4000, 1.1,  2.1,   # Boost presence
            10000, 0.9, 1.5    # Boost air
        ])
        
        return self.formatter.analyze_audio_clip(audio, model_output)
    
    def start_continuous_monitoring(self, interval_seconds=10, output_dir="./analysis_logs"):
        """
        Continuously monitor and analyze audio
        
        Args:
            interval_seconds: Time between analyses
            output_dir: Directory to save analysis logs
        """
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print(f"🔄 Starting continuous monitoring (every {interval_seconds}s)")
        print(f"📂 Saving logs to: {output_dir}")
        
        analysis_count = 0
        
        try:
            while True:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = os.path.join(output_dir, f"analysis_{timestamp}.json")
                
                print(f"\n--- Analysis #{analysis_count + 1} ---")
                analysis = self.analyze_live_audio(
                    duration_seconds=5, 
                    output_file=output_file
                )
                
                # Print key findings
                self._print_summary(analysis)
                
                analysis_count += 1
                
                print(f"⏳ Waiting {interval_seconds} seconds...")
                time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            print(f"\n🛑 Monitoring stopped. Analyzed {analysis_count} clips.")
    
    def _print_summary(self, analysis):
        """Print a quick summary of the analysis"""
        
        print("📊 Quick Summary:")
        print(f"   Overall confidence: {analysis['confidence_overall']}")
        print(f"   Mix quality: {analysis['mix_quality_assessment']['overall_grade']}")
        print(f"   Problems detected: {analysis['mix_quality_assessment']['problem_count']}")
        print(f"   Recommended action: {analysis['real_time_metrics']['recommended_action']}")
        
        if analysis['eq_recommendations']['ai_model_active']:
            ai_corrections = analysis['eq_recommendations']['ai_predictions']['total_corrections']
            print(f"   🤖 AI suggestions: {ai_corrections} EQ corrections")
        else:
            print("   🤖 AI: Not active")

def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='SonicMind-AI Live Audio Analyzer')
    parser.add_argument('--mode', choices=['live', 'file', 'monitor', 'demo'], 
                       default='demo', help='Analysis mode')
    parser.add_argument('--duration', type=int, default=5, 
                       help='Recording duration in seconds')
    parser.add_argument('--file', type=str, 
                       help='Audio file to analyze (for file mode)')
    parser.add_argument('--output', type=str, 
                       help='Output JSON file path')
    parser.add_argument('--model', type=str, 
                       help='Path to AI model file')
    parser.add_argument('--interval', type=int, default=10,
                       help='Monitoring interval in seconds')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = LiveAudioAnalyzer(model_path=args.model)
    
    print("# SonicMind-AI")
    print("Real Time Live Sound Engineering Assistant")
    print("=" * 50)
    
    # Run based on mode
    if args.mode == 'live':
        analysis = analyzer.analyze_live_audio(args.duration, args.output)
        print(json.dumps(analysis, indent=2))
        
    elif args.mode == 'file':
        if not args.file:
            print("❌ Please specify --file for file mode")
            return
        analysis = analyzer.analyze_audio_file(args.file, args.output)
        print(json.dumps(analysis, indent=2))
        
    elif args.mode == 'monitor':
        analyzer.start_continuous_monitoring(args.interval)
        
    else:  # demo mode
        print("\n🎯 Running demo analysis...")
        analysis = analyzer._demo_analysis()
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(analysis, f, indent=2)
            print(f"💾 Demo analysis saved to {args.output}")
        
        print(json.dumps(analysis, indent=2))

if __name__ == "__main__":
    main()