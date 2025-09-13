#!/usr/bin/env python3
"""
Live Effects Pedal GUI Application (No Presets)
===============================================
Tkinter GUI for real-time audio testing of SonicMind Live Effects Pedal.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import sys
import os

# Import LivePedal from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from LivePedal import LiveEffectsPedal


class PedalGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🎸 SonicMind Live Effects Pedal")
        self.geometry("900x650")
        self.configure(bg="black")

        # Initialize pedal
        self.pedal = LiveEffectsPedal(block_size=128)
        self.sliders = {}

        # Layout
        self.create_widgets()

        # Start audio in a separate thread
        threading.Thread(target=self.start_audio, daemon=True).start()

        # Update status in background
        self.running = True
        threading.Thread(target=self.update_status_loop, daemon=True).start()

    def start_audio(self):
        """Start the pedal's audio stream safely"""
        try:
            self.pedal.start()
            time.sleep(0.1)  # Give audio thread time to initialize
            if not self.pedal.is_running:
                messagebox.showerror("Error", "❌ Failed to start audio processing")
                self.destroy()
        except Exception as e:
            messagebox.showerror("Error", f"❌ Audio initialization failed:\n{e}")
            self.destroy()

    def create_widgets(self):
        # Title
        title = tk.Label(
            self, text="🎸 SonicMind Live Pedalboard",
            fg="white", bg="black", font=("Arial", 20, "bold")
        )
        title.pack(pady=10)

        # Status area
        self.status_label = tk.Label(
            self, text="🔄 Initializing...",
            fg="white", bg="black", font=("Arial", 12)
        )
        self.status_label.pack(pady=10)

        # Effect management controls
        control_frame = tk.Frame(self, bg="black")
        control_frame.pack(pady=5)

        tk.Label(control_frame, text="Add Effect:", fg="white", bg="black").pack(side=tk.LEFT, padx=5)
        self.effect_choice = ttk.Combobox(control_frame, values=[
            "compressor", "distortion", "eq", "delay", "reverb"
        ])
        self.effect_choice.pack(side=tk.LEFT, padx=5)
        self.effect_choice.current(0)

        add_btn = ttk.Button(control_frame, text="➕ Add", command=self.add_effect)
        add_btn.pack(side=tk.LEFT, padx=5)

        clear_btn = ttk.Button(control_frame, text="🗑️ Clear Chain", command=self.clear_chain)
        clear_btn.pack(side=tk.LEFT, padx=5)

        # Effects frame
        self.effects_frame = tk.Frame(self, bg="black")
        self.effects_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # Quit button
        quit_btn = ttk.Button(self, text="❌ Quit", command=self.quit_app)
        quit_btn.pack(pady=10)

    def clear_effects_ui(self):
        for widget in self.effects_frame.winfo_children():
            widget.destroy()
        self.sliders.clear()

    def build_effects_ui(self):
        """Dynamically build UI for all effects in chain"""
        self.clear_effects_ui()
        chain = self.pedal.get_chain_info()

        if not chain:
            lbl = tk.Label(
                self.effects_frame, text="(No effects loaded)",
                fg="gray", bg="black", font=("Arial", 12)
            )
            lbl.pack()
            return

        for i, effect in enumerate(chain):
            frame = tk.LabelFrame(
                self.effects_frame,
                text=f"Effect {i}: {effect['type']}",
                fg="white", bg="black",
                font=("Arial", 12, "bold"), labelanchor="n"
            )
            frame.pack(fill=tk.X, padx=10, pady=5)

            # Toggle button
            toggle_var = tk.BooleanVar(value=effect['enabled'])
            toggle_btn = ttk.Checkbutton(
                frame, text="Enabled", variable=toggle_var,
                command=lambda idx=i, var=toggle_var: self.toggle_effect(idx, var.get())
            )
            toggle_btn.pack(anchor="w", padx=5, pady=2)

            # Sliders for parameters
            for param, value in effect['parameters'].items():
                slider_frame = tk.Frame(frame, bg="black")
                slider_frame.pack(fill=tk.X, padx=5, pady=2)

                lbl = tk.Label(
                    slider_frame, text=f"{param}:",
                    fg="white", bg="black", width=12, anchor="w"
                )
                lbl.pack(side=tk.LEFT)

                var = tk.DoubleVar(value=value)
                slider = ttk.Scale(
                    slider_frame, from_=0.0, to=2.0, variable=var,
                    command=lambda val, idx=i, pname=param: self.set_param(idx, pname, float(val))
                )
                slider.pack(side=tk.LEFT, fill=tk.X, expand=True)

                self.sliders[(i, param)] = var

    def set_param(self, effect_id, param_name, value):
        try:
            self.pedal.set_effect_parameter(effect_id, param_name, value)
        except Exception as e:
            print(f"Param error: {e}")

    def toggle_effect(self, effect_id, enabled):
        chain = self.pedal.get_chain_info()
        if 0 <= effect_id < len(chain):
            if enabled != chain[effect_id]['enabled']:
                self.pedal.toggle_effect(effect_id)

    def add_effect(self):
        choice = self.effect_choice.get().strip().lower()
        if not choice:
            return

        try:
            if choice == "compressor":
                self.pedal.add_effect("compressor", threshold=0.5, ratio=2.0, makeup_gain=1.0)
            elif choice == "distortion":
                self.pedal.add_effect("distortion", drive=0.5, tone=0.7, level=0.8)
            elif choice == "eq":
                self.pedal.add_effect("eq", bass=1.0, mid=1.0, treble=1.0)
            elif choice == "delay":
                self.pedal.add_effect("delay", time=0.25, feedback=0.4, mix=0.3)
            elif choice == "reverb":
                self.pedal.add_effect("reverb", room_size=0.6, decay=0.5, mix=0.3)
            else:
                messagebox.showwarning("Invalid Effect", f"Effect '{choice}' is not supported")
                return
        except Exception as e:
            messagebox.showerror("Error", f"Failed to add effect: {e}")
            return

        self.build_effects_ui()

    def clear_chain(self):
        self.pedal.clear_chain()
        self.build_effects_ui()

    def update_status_loop(self):
        """Background loop to update status"""
        while self.running:
            try:
                status = self.pedal.get_status()
                latency_ms = (status['block_size'] / status['sample_rate']) * 1000
                self.status_label.config(
                    text=f"Status: {'🟢 Running' if status['running'] else '🔴 Stopped'} | "
                         f"CPU: {status['cpu_load']:.1f}% | "
                         f"Latency: ~{latency_ms:.1f} ms | "
                         f"Effects: {status['effects_count']} | "
                         f"Dropouts: {status['dropout_count']}"
                )
            except Exception as e:
                self.status_label.config(text=f"❌ Error: {e}")
            time.sleep(0.2)

    def quit_app(self):
        self.running = False
        self.pedal.stop()
        self.destroy()


def main():
    app = PedalGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
