#!/usr/bin/env python3
"""
SonicMind-AI Test GUI
===================
Simple GUI for testing audio analysis with 5-second capture timer
Perfect for hackathon demos!
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import json
import threading
import time
from datetime import datetime
import numpy as np

try:
    from live_analyzer import LiveAudioAnalyzer
    ANALYZER_AVAILABLE = True
except ImportError:
    ANALYZER_AVAILABLE = False
    print("Live analyzer not available - demo mode only")

class SonicMindTestGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("SonicMind-AI Test Interface")
        self.root.geometry("1000x700")
        self.root.configure(bg='#2b2b2b')
        
        # Analyzer
        self.analyzer = LiveAudioAnalyzer() if ANALYZER_AVAILABLE else None
        self.is_recording = False
        self.countdown_active = False
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface"""
        
        # Main title
        title_frame = tk.Frame(self.root, bg='#2b2b2b')
        title_frame.pack(pady=20)
        
        title_label = tk.Label(
            title_frame, 
            text="🎵 SonicMind-AI", 
            font=("Arial", 24, "bold"),
            fg='#00ff88',
            bg='#2b2b2b'
        )
        title_label.pack()
        
        subtitle_label = tk.Label(
            title_frame,
            text="Real Time Live Sound Engineering Assistant",
            font=("Arial", 12),
            fg='#cccccc',
            bg='#2b2b2b'
        )
        subtitle_label.pack()
        
        # Status area
        status_frame = tk.Frame(self.root, bg='#2b2b2b')
        status_frame.pack(pady=10)
        
        self.status_label = tk.Label(
            status_frame,
            text="🔄 Ready to analyze audio",
            font=("Arial", 14),
            fg='#ffffff',
            bg='#2b2b2b'
        )
        self.status_label.pack()
        
        # Countdown display
        self.countdown_label = tk.Label(
            status_frame,
            text="",
            font=("Arial", 48, "bold"),
            fg='#ff6b6b',
            bg='#2b2b2b'
        )
        self.countdown_label.pack(pady=10)
        
        # Control buttons
        button_frame = tk.Frame(self.root, bg='#2b2b2b')
        button_frame.pack(pady=20)
        
        self.record_button = tk.Button(
            button_frame,
            text="🎤 Start 5-Second Analysis",
            font=("Arial", 16, "bold"),
            fg='white',
            bg='#4CAF50',
            activebackground='#45a049',
            padx=20,
            pady=10,
            command=self.start_analysis,
            cursor='hand2'
        )
        self.record_button.pack(side='left', padx=10)
        
        self.demo_button = tk.Button(
            button_frame,
            text="🎯 Demo Analysis",
            font=("Arial", 16),
            fg='white',
            bg='#2196F3',
            activebackground='#1976D2',
            padx=20,
            pady=10,
            command=self.run_demo,
            cursor='hand2'
        )
        self.demo_button.pack(side='left', padx=10)
        
        self.clear_button = tk.Button(
            button_frame,
            text="🗑️ Clear Results",
            font=("Arial", 16),
            fg='white',
            bg='#FF9800',
            activebackground='#F57C00',
            padx=20,
            pady=10,
            command=self.clear_results,
            cursor='hand2'
        )
        self.clear_button.pack(side='left', padx=10)
        
        # Results area
        results_frame = tk.Frame(self.root, bg='#2b2b2b')
        results_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Split into two columns
        left_frame = tk.Frame(results_frame, bg='#2b2b2b')
        left_frame.pack(side='left', fill='both', expand=True, padx=(0, 10))
        
        right_frame = tk.Frame(results_frame, bg='#2b2b2b')
        right_frame.pack(side='right', fill='both', expand=True, padx=(10, 0))
        
        # Summary panel (left)
        summary_label = tk.Label(
            left_frame,
            text="📊 Analysis Summary",
            font=("Arial", 14, "bold"),
            fg='#00ff88',
            bg='#2b2b2b'
        )
        summary_label.pack(anchor='w')
        
        self.summary_text = scrolledtext.ScrolledText(
            left_frame,
            height=15,
            bg='#1e1e1e',
            fg='#ffffff',
            font=("Consolas", 10),
            wrap=tk.WORD
        )
        self.summary_text.pack(fill='both', expand=True, pady=(5, 0))
        
        # JSON output panel (right)
        json_label = tk.Label(
            right_frame,
            text="📋 Full JSON Analysis",
            font=("Arial", 14, "bold"),
            fg='#00ff88',
            bg='#2b2b2b'
        )
        json_label.pack(anchor='w')
        
        self.json_text = scrolledtext.ScrolledText(
            right_frame,
            height=15,
            bg='#1e1e1e',
            fg='#cccccc',
            font=("Consolas", 9),
            wrap=tk.WORD
        )
        self.json_text.pack(fill='both', expand=True, pady=(5, 0))
        
        # Footer
        footer_label = tk.Label(
            self.root,
            text="💡 Tip: Click 'Start 5-Second Analysis' to capture live audio, or 'Demo Analysis' for instant results!",
            font=("Arial", 10),
            fg='#888888',
            bg='#2b2b2b'
        )
        footer_label.pack(pady=10)
        
    def start_analysis(self):
        """Start the 5-second analysis with countdown"""
        if self.is_recording or self.countdown_active:
            return
            
        self.is_recording = True
        self.record_button.config(state='disabled', bg='#666666', text="🔄 Recording...")
        self.demo_button.config(state='disabled')
        
        # Start countdown in separate thread
        threading.Thread(target=self.countdown_and_record, daemon=True).start()
        
    def countdown_and_record(self):
        """Run countdown and then record audio"""
        try:
            self.countdown_active = True
            
            # 3-second countdown
            for i in range(3, 0, -1):
                self.root.after(0, lambda i=i: self.update_countdown(f"🔴 {i}"))
                self.root.after(0, lambda: self.update_status("🎤 Get ready to speak..."))
                time.sleep(1)
            
            # Start recording
            self.root.after(0, lambda: self.update_countdown("🎤 RECORDING"))
            self.root.after(0, lambda: self.update_status("🎤 Recording 5 seconds of audio..."))
            
            # Analyze audio (5 seconds)
            if self.analyzer and ANALYZER_AVAILABLE:
                analysis = self.analyzer.analyze_live_audio(duration_seconds=5)
            else:
                # Demo mode
                time.sleep(5)  # Simulate recording time
                analysis = self.create_demo_analysis()
            
            # Display results
            self.root.after(0, lambda: self.display_results(analysis))
            
        except Exception as e:
            self.root.after(0, lambda: self.show_error(f"Analysis failed: {e}"))
        finally:
            self.countdown_active = False
            self.is_recording = False
            self.root.after(0, self.reset_buttons)
            
    def run_demo(self):
        """Run demo analysis immediately"""
        if self.is_recording:
            return
            
        self.demo_button.config(state='disabled', bg='#666666', text="🔄 Analyzing...")
        self.record_button.config(state='disabled')
        
        # Run demo in thread
        threading.Thread(target=self.demo_analysis_thread, daemon=True).start()
        
    def demo_analysis_thread(self):
        """Demo analysis in separate thread"""
        try:
            self.root.after(0, lambda: self.update_status("🎯 Running demo analysis..."))
            
            if self.analyzer and ANALYZER_AVAILABLE:
                analysis = self.analyzer._demo_analysis()
            else:
                analysis = self.create_demo_analysis()
                
            self.root.after(0, lambda: self.display_results(analysis))
            
        except Exception as e:
            self.root.after(0, lambda: self.show_error(f"Demo failed: {e}"))
        finally:
            self.root.after(0, self.reset_buttons)
            
    def create_demo_analysis(self):
        """Create demo analysis when analyzer not available"""
        timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
        
        return {
            "analysis_id": f"demo_analysis_{timestamp}",
            "timestamp": timestamp,
            "confidence_overall": 0.87,
            "audio_characteristics": {
                "duration_s": 5.0,
                "sample_rate": 44100,
                "rms_db": -12.3,
                "peak_db": -3.2,
                "dynamic_range": "moderate_dynamic_range"
            },
            "frequency_analysis": {
                "balance": {
                    "bass_ratio": 0.35,
                    "mids_ratio": 0.45,
                    "highs_ratio": 0.20,
                    "balance_classification": "balanced"
                },
                "tonal_content": {
                    "tonality_index": 0.52,
                    "classification": "moderately_tonal"
                }
            },
            "detected_problems": [
                {
                    "problem": "dull",
                    "description": "Lack of high frequency clarity",
                    "severity": 0.6,
                    "frequency_range": "6000+ Hz",
                    "suggested_action": "boost_highs"
                }
            ],
            "eq_recommendations": {
                "urgency": "medium",
                "ai_model_active": True,
                "ai_predictions": {
                    "predictions": [
                        {"band": 1, "name": "80Hz", "gain_db": -1.2, "action": "cut"},
                        {"band": 2, "name": "300Hz", "gain_db": 0.5, "action": "boost"},
                        {"band": 3, "name": "1kHz", "gain_db": 0.8, "action": "boost"},
                        {"band": 4, "name": "4kHz", "gain_db": 1.5, "action": "boost"},
                        {"band": 5, "name": "10kHz", "gain_db": 2.3, "action": "boost"}
                    ],
                    "max_correction_db": 2.3,
                    "total_corrections": 4
                }
            },
            "mix_quality_assessment": {
                "overall_grade": "good",
                "problem_count": 1,
                "frequency_balance_score": 0.78
            },
            "real_time_metrics": {
                "recommended_action": "minor_eq_adjustments"
            }
        }
    
    def display_results(self, analysis):
        """Display analysis results in both panels"""
        try:
            # Update status
            self.update_status("✅ Analysis complete!")
            self.update_countdown("")
            
            # Clear previous results
            self.summary_text.delete(1.0, tk.END)
            self.json_text.delete(1.0, tk.END)
            
            # Display summary
            summary = self.create_summary(analysis)
            self.summary_text.insert(tk.END, summary)
            
            # Display JSON
            json_output = json.dumps(analysis, indent=2)
            self.json_text.insert(tk.END, json_output)
            
        except Exception as e:
            self.show_error(f"Display error: {e}")
    
    def create_summary(self, analysis):
        """Create human-readable summary"""
        summary = f"""🎵 SONICMIND-AI ANALYSIS RESULTS
{'='*50}

📊 OVERALL ASSESSMENT:
   • Confidence: {analysis.get('confidence_overall', 'N/A')}
   • Mix Quality: {analysis.get('mix_quality_assessment', {}).get('overall_grade', 'N/A').title()}
   • Problems Found: {analysis.get('mix_quality_assessment', {}).get('problem_count', 'N/A')}

🎛️ FREQUENCY BALANCE:
   • Bass: {analysis.get('frequency_analysis', {}).get('balance', {}).get('bass_ratio', 0)*100:.1f}%
   • Mids: {analysis.get('frequency_analysis', {}).get('balance', {}).get('mids_ratio', 0)*100:.1f}%
   • Highs: {analysis.get('frequency_analysis', {}).get('balance', {}).get('highs_ratio', 0)*100:.1f}%
   • Classification: {analysis.get('frequency_analysis', {}).get('balance', {}).get('balance_classification', 'N/A').replace('_', ' ').title()}

⚠️ DETECTED PROBLEMS:
"""
        
        problems = analysis.get('detected_problems', [])
        if problems:
            for i, problem in enumerate(problems, 1):
                summary += f"   {i}. {problem.get('problem', 'Unknown').title()}: {problem.get('description', 'N/A')}\n"
                summary += f"      Severity: {problem.get('severity', 0):.2f} | Range: {problem.get('frequency_range', 'N/A')}\n"
        else:
            summary += "   ✅ No significant problems detected!\n"
            
        summary += f"""
🤖 AI RECOMMENDATIONS:
   • Model Active: {'Yes' if analysis.get('eq_recommendations', {}).get('ai_model_active') else 'No'}
   • Urgency Level: {analysis.get('eq_recommendations', {}).get('urgency', 'N/A').title()}
"""

        ai_predictions = analysis.get('eq_recommendations', {}).get('ai_predictions', {})
        if ai_predictions and 'predictions' in ai_predictions:
            summary += "   • EQ Suggestions:\n"
            for pred in ai_predictions['predictions']:
                action = pred.get('action', 'neutral')
                if action != 'neutral':
                    summary += f"     - {pred.get('name', 'N/A')}: {pred.get('gain_db', 0):+.1f}dB ({action})\n"
        
        summary += f"""
🎯 RECOMMENDED ACTION:
   {analysis.get('real_time_metrics', {}).get('recommended_action', 'N/A').replace('_', ' ').title()}

📈 TECHNICAL DETAILS:
   • Duration: {analysis.get('audio_characteristics', {}).get('duration_s', 'N/A')}s
   • RMS Level: {analysis.get('audio_characteristics', {}).get('rms_db', 'N/A')}dB
   • Peak Level: {analysis.get('audio_characteristics', {}).get('peak_db', 'N/A')}dB
   • Dynamic Range: {analysis.get('audio_characteristics', {}).get('dynamic_range', 'N/A').replace('_', ' ').title()}

✨ Analysis completed at {analysis.get('timestamp', 'N/A')}
"""
        
        return summary
    
    def update_status(self, text):
        """Update status label"""
        self.status_label.config(text=text)
        
    def update_countdown(self, text):
        """Update countdown label"""
        self.countdown_label.config(text=text)
        
    def reset_buttons(self):
        """Reset button states"""
        self.record_button.config(
            state='normal', 
            bg='#4CAF50', 
            text="🎤 Start 5-Second Analysis"
        )
        self.demo_button.config(
            state='normal',
            bg='#2196F3',
            text="🎯 Demo Analysis"
        )
        self.update_countdown("")
        
    def clear_results(self):
        """Clear all results"""
        self.summary_text.delete(1.0, tk.END)
        self.json_text.delete(1.0, tk.END)
        self.update_status("🔄 Ready to analyze audio")
        self.update_countdown("")
        
    def show_error(self, message):
        """Show error message"""
        self.update_status(f"❌ {message}")
        messagebox.showerror("Error", message)
        
    def run(self):
        """Start the GUI"""
        self.root.mainloop()

def main():
    """Main function"""
    print("🎵 Starting SonicMind-AI Test GUI...")
    app = SonicMindTestGUI()
    app.run()

if __name__ == "__main__":
    main()