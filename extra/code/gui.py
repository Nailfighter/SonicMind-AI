#!/usr/bin/env python3
"""
Live Auto‑EQ GUI (Tkinter) — Rules + Neural Model (v4)
======================================================
Updates in this version:
• Distinct AI status messages: Torch missing vs model file missing vs loaded OK.
• Default model path set to: src/fma_eq_model_npy.pth (can still pass CLI arg).
• Warm‑up 5s, then 1s auto‑updates using a sliding 5s analysis window.
• Settling for steady tones: deadband + stability hold + softened moves on tonal content.
• No extra audio delay unless Delay/Reverb toggled; realtime path uses zero‑latency EQ/Comp.
• Lower latency defaults (blocksize=128). Weights fixed at 70:30 (rules:model).
"""

import os
import sys
import time
import math
import threading
from collections import deque
from dataclasses import dataclass

import numpy as np
import sounddevice as sd

# Optional deps
SCIPY_AVAILABLE = False
try:
    from scipy import signal
    SCIPY_AVAILABLE = True
except Exception:
    pass

TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except Exception:
    pass

try:
    import tkinter as tk
    from tkinter import ttk
    from tkinter import messagebox
except Exception as e:
    print("Tkinter not available:", e)
    raise

# ----------------------------- Utility ------------------------------------ #


def db_to_lin(db):
    return 10 ** (db / 20.0)


def lin_to_db(x, eps=1e-12):
    return 20 * np.log10(np.maximum(np.abs(x), eps))

# --------------------------- Audio Primitives ------------------------------ #


@dataclass
class EQBand:
    freq: float  # Hz
    q: float     # Q factor
    gain_db: float  # dB


class ParametricEQ:
    """5‑band peaking EQ using biquads (SciPy); FFT fallback if SciPy not present."""

    def __init__(self, sr: int, bands: list[EQBand]):
        self.sr = sr
        self.bands = bands
        self._biquads = []  # (b,a) per band
        self._design()

    def update(self, bands: list[EQBand]):
        self.bands = bands
        self._design()

    def _design(self):
        self._biquads.clear()
        for b in self.bands:
            if SCIPY_AVAILABLE:
                # RBJ peaking EQ design
                A = 10 ** (b.gain_db / 40)
                w0 = 2 * math.pi * b.freq / self.sr
                alpha = math.sin(w0) / (2 * max(b.q, 1e-4))
                cosw = math.cos(w0)
                b0 = 1 + alpha * A
                b1 = -2 * cosw
                b2 = 1 - alpha * A
                a0 = 1 + alpha / A
                a1 = -2 * cosw
                a2 = 1 - alpha / A
                bcoef = np.array([b0, b1, b2], dtype=np.float64) / a0
                acoef = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64)
                self._biquads.append((bcoef, acoef))
            else:
                self._biquads.append(None)

    def process(self, x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return x
        if SCIPY_AVAILABLE:
            y = x
            for coefs in self._biquads:
                if coefs is None:
                    continue
                b, a = coefs
                if not np.isfinite(b).all() or not np.isfinite(a).all():
                    continue
                y = signal.lfilter(b, a, y)
            return y.astype(np.float32, copy=False)
        else:
            # FFT fallback (approximate); not used in realtime path if SciPy present
            Y = np.fft.rfft(x)
            freqs = np.fft.rfftfreq(x.size, 1 / self.sr)
            mag = np.ones_like(Y, dtype=np.float64)
            for band in self.bands:
                bw = max(band.freq / max(band.q, 1e-3), 10.0)
                g = 10 ** (band.gain_db / 20.0)
                mag *= 1 + (g - 1) * np.exp(-0.5 *
                                            ((freqs - band.freq) / (bw / 2.355)) ** 2)
            Y *= mag
            y = np.fft.irfft(Y, n=x.size)
            return y.astype(np.float32, copy=False)


class Dynamics:
    def __init__(self, sr: int):
        self.sr = sr
        # compressor params — zero lookahead (no added latency)
        self.threshold = -16.0  # dBFS
        self.ratio = 3.0
        self.attack = 0.01
        self.release = 0.2
        self.makeup = 3.0  # dB
        self.env = 0.0

    def compressor(self, x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return x
        absx = np.abs(x)
        atk = math.exp(-1.0 / (self.attack * self.sr))
        rel = math.exp(-1.0 / (self.release * self.sr))
        y = np.empty_like(x)
        env = self.env
        thr_lin = db_to_lin(self.threshold)
        makeup_lin = db_to_lin(self.makeup)
        for i, s in enumerate(absx):
            if s > env:
                env = atk * env + (1 - atk) * s
            else:
                env = rel * env + (1 - rel) * s
            if env <= thr_lin:
                gain = 1.0
            else:
                over_db = lin_to_db(env) - self.threshold
                comp_db = over_db - over_db / self.ratio
                gain = db_to_lin(-comp_db)
            y[i] = x[i] * gain * makeup_lin
        self.env = env
        return y


class FX:
    def __init__(self, sr: int):
        self.sr = sr
        # Delay defaults (audible when enabled)
        self.delay_s = 0.25
        self.delay_fb = 0.35
        self.delay_mix = 0.35
        self._delay_z = np.zeros(int(sr * 2), dtype=np.float32)
        self._delay_idx = 0

        self.reverb_mix = 0.18
        # Simple Schroeder/Moorer — combs + allpass (streaming, no lookahead)
        self._comb_delays = [int(sr * t) for t in (0.029, 0.037, 0.041)]
        self._comb_bufs = [np.zeros(d, dtype=np.float32)
                           for d in self._comb_delays]
        self._comb_idx = [0, 0, 0]
        self._comb_fb = [0.805, 0.827, 0.783]
        self._ap_delay = int(sr * 0.012)
        self._ap_buf = np.zeros(self._ap_delay, dtype=np.float32)
        self._ap_idx = 0
        self._ap_g = 0.7

        self.drive = 0.0  # 0..1

    def delay(self, x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return x
        out = np.copy(x)
        dlen = int(max(1, self.sr * self.delay_s))
        for i in range(x.size):
            rpos = (self._delay_idx - dlen) % self._delay_z.size
            d = self._delay_z[rpos]
            y = x[i] * (1 - self.delay_mix) + d * self.delay_mix
            out[i] = y
            self._delay_z[self._delay_idx] = x[i] + d * self.delay_fb
            self._delay_idx = (self._delay_idx + 1) % self._delay_z.size
        return out

    def reverb(self, x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return x
        y = np.zeros_like(x)
        for ci, (buf, fb) in enumerate(zip(self._comb_bufs, self._comb_fb)):
            idx = self._comb_idx[ci]
            d = buf.size
            for i, s in enumerate(x):
                v = buf[idx]
                y[i] += v
                buf[idx] = s + v * fb
                idx += 1
                if idx >= d:
                    idx = 0
            self._comb_idx[ci] = idx
        y /= len(self._comb_bufs)
        ap = self._ap_buf
        idx = self._ap_idx
        d = ap.size
        g = self._ap_g
        out = np.empty_like(x)
        for i, s in enumerate(y):
            v = ap[idx]
            ap[idx] = s + v * g
            out[i] = -g * ap[idx] + v
            idx += 1
            if idx >= d:
                idx = 0
        self._ap_idx = idx
        return x * (1 - self.reverb_mix) + out * self.reverb_mix

    def saturate(self, x: np.ndarray) -> np.ndarray:
        if self.drive <= 0.001:
            return x
        g = 1.0 + 6.0 * float(self.drive)
        y = np.tanh(x * g)
        return 0.8 * y + 0.2 * x

# ----------------------- Features & Rule Engine ---------------------------- #


class Analyzer:
    def __init__(self, sr: int):
        self.sr = sr

    def features(self, x: np.ndarray) -> dict:
        if x.size == 0:
            return dict(rms=0, peak=0, bass=0, mids=0, highs=0, total=1, tonal=0.0)
        X = np.fft.rfft(x)
        mag = np.abs(X)
        freqs = np.fft.rfftfreq(x.size, 1 / self.sr)

        def band(a, b):
            m = mag[(freqs >= a) & (freqs < b)].sum()
            return float(m)
        bass = band(20, 250)
        mids = band(250, 4000)
        highs = band(4000, 20000)
        total = bass + mids + highs + 1e-8
        rms = float(np.sqrt(np.mean(x ** 2)))
        peak = float(np.max(np.abs(x)) + 1e-9)
        # dominance of strongest bin
        tonal = float(mag.max() / (mag.sum() + 1e-8))
        return dict(rms=rms, peak=peak, bass=bass, mids=mids, highs=highs, total=total, tonal=tonal)


class RuleEQ:
    """Small corrections toward a pleasant tonal target (dominant influence)."""

    def __init__(self):
        self.target = dict(bass=0.27, mids=0.53, highs=0.20)
        self.max_step_db_first = 0.5  # first correction
        self.max_step_db = 0.25       # subsequent 1s corrections
        self.deadband = 0.02          # proportion error deadband

    def suggest(self, feats: dict, bands: list[EQBand], first_step=False) -> list[float]:
        if feats['total'] <= 1e-8:
            return [0.0] * len(bands)
        cur = dict(
            bass=feats['bass'] / feats['total'],
            mids=feats['mids'] / feats['total'],
            highs=feats['highs'] / feats['total'],
        )
        region_weights = [
            ('bass', 0.6),   # band 0
            ('bass', 0.6),   # band 1
            ('mids', 0.7),   # band 2
            ('mids', 0.7),   # band 3
            ('highs', 1.0),  # band 4
        ]
        cap = self.max_step_db_first if first_step else self.max_step_db
        deltas = []
        for (region, weight) in region_weights:
            err = self.target[region] - cur[region]
            if abs(err) < self.deadband:
                raw_db = 0.0
            else:
                raw_db = 6.0 * err * weight
            raw_db = max(-cap, min(cap, raw_db))
            deltas.append(raw_db)
        if feats.get('tonal', 0.0) > 0.6:
            deltas = [d * 0.4 for d in deltas]
        return deltas

# --------------------------- Neural Model ---------------------------------- #


class DynamicEQModel(nn.Module if TORCH_AVAILABLE else object):
    """Architecture aligned to your checkpoint (with BN16 and 16→8→15 head).
    Layers (with indices matching the checkpoint):
      0 Linear(12,512)  1 ReLU  2 BN512  3 Dropout(0.3)
      4 Linear(512,256) 5 ReLU  6 BN256  7 Dropout(0.2)
      8 Linear(256,128) 9 ReLU 10 BN128 11 Dropout(0.1)
     12 Linear(128,64) 13 ReLU 14 BN64  15 Dropout(0.1)
     16 Linear(64,32)  17 ReLU 18 BN32  19 Dropout(0.05)
     20 Linear(32,16)  21 ReLU 22 BN16  23 Dropout(0.05)
     24 Linear(16,8)   25 ReLU 26 Linear(8,15)
    """

    def __init__(self):
        if not TORCH_AVAILABLE:
            raise RuntimeError("Torch not available")
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(12, 512), nn.ReLU(), nn.BatchNorm1d(
                512), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(), nn.BatchNorm1d(
                256), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(), nn.BatchNorm1d(
                128), nn.Dropout(0.1),
            nn.Linear(128, 64),  nn.ReLU(), nn.BatchNorm1d(
                64),  nn.Dropout(0.1),
            nn.Linear(64, 32),   nn.ReLU(), nn.BatchNorm1d(
                32),  nn.Dropout(0.05),
            nn.Linear(32, 16),   nn.ReLU(), nn.BatchNorm1d(
                16),  nn.Dropout(0.05),
            nn.Linear(16, 8),    nn.ReLU(),
            nn.Linear(8, 15)
        )

    def forward(self, x):
        return self.network(x)


class ModelWrapper:
    def __init__(self, sr: int, path: str | None):
        self.sr = sr
        self.available = False
        self.device = None
        self.model = None
        self.status = ""

        if not TORCH_AVAILABLE:
            self.status = "❌ PyTorch not installed/detected — AI disabled"
            print(self.status)
            return

        # Default to known path if none provided
        if not path:
            path = os.path.join("src", "fma_eq_model_npy.pth")

        if not os.path.exists(path):
            self.status = f"❌ Model file not found at: {path} — AI disabled"
            print(self.status)
            return

        try:
            self.device = torch.device('mps' if torch.backends.mps.is_available() else (
                'cuda' if torch.cuda.is_available() else 'cpu'))
            ckpt = torch.load(path, map_location=self.device)
            state = ckpt['model_state_dict'] if isinstance(
                ckpt, dict) and 'model_state_dict' in ckpt else ckpt
            self.model = DynamicEQModel()
            # load strictly now that the architecture matches the checkpoint
            self.model.load_state_dict(state, strict=True)
            self.model.to(self.device)
            self.model.eval()
            self.available = True
            self.status = f"✅ Loaded EQ model on {self.device}: {path}"
            print(self.status)
        except Exception as e:
            self.status = f"❌ Failed to load model: {e} — AI disabled"
            print(self.status)

    def __init__(self, sr: int, path: str | None):
        self.sr = sr
        self.available = False
        self.device = None
        self.model = None
        self.status = ""

        if not TORCH_AVAILABLE:
            self.status = "❌ PyTorch not installed/detected — AI disabled"
            print(self.status)
            return

        # Default to known path if none provided
        if not path:
            path = os.path.join("src", "fma_eq_model_npy.pth")

        if not os.path.exists(path):
            self.status = f"❌ Model file not found at: {path} — AI disabled"
            print(self.status)
            return

        try:
            self.device = torch.device('mps' if torch.backends.mps.is_available() else (
                'cuda' if torch.cuda.is_available() else 'cpu'))
            ckpt = torch.load(path, map_location=self.device)
            state = ckpt['model_state_dict'] if isinstance(
                ckpt, dict) and 'model_state_dict' in ckpt else ckpt
            self.model = DynamicEQModel()
            self.model.load_state_dict(state)
            self.model.to(self.device)
            self.model.eval()
            self.available = True
            self.status = f"✅ Loaded EQ model on {self.device}: {path}"
            print(self.status)
        except Exception as e:
            self.status = f"❌ Failed to load model: {e} — AI disabled"
            print(self.status)

    def _features12(self, x: np.ndarray) -> np.ndarray:
        X = np.fft.rfft(x)
        mag = np.abs(X)
        freqs = np.fft.rfftfreq(x.size, 1 / self.sr)

        def band(a, b):
            return float(mag[(freqs >= a) & (freqs <= b)].sum())
        bass = band(20, 250)
        mids = band(250, 4000)
        highs = band(4000, 20000)
        total = bass + mids + highs + 1e-8
        rms = float(np.sqrt(np.mean(x ** 2)))
        peak = float(np.max(np.abs(x)) + 1e-9)
        crest = peak / (rms + 1e-8)
        sc = float((freqs * mag).sum() / (mag.sum() + 1e-8)) / 1000.0
        feat = np.array([
            bass / total, mids / total, highs / total,
            rms, peak, crest, sc,
            float(np.mean(x)), float(np.std(x)),
            float(np.mean(np.diff(x))) if x.size > 1 else 0.0,
            float(np.std(np.diff(x))) if x.size > 1 else 0.0,
            x.size / self.sr
        ], dtype=np.float32)
        return feat

    def predict_params(self, x: np.ndarray) -> np.ndarray | None:
        if not self.available or self.model is None:
            return None
        with torch.no_grad():
            f = self._features12(x)
            t = torch.from_numpy(f).unsqueeze(0).to(self.device)
            y = self.model(t).cpu().numpy()[0]
            return y

# ------------------------- Audio Engine (Rt) ------------------------------- #


class Ring:
    def __init__(self, n: int):
        self.buf = np.zeros(n, dtype=np.float32)
        self.n = n
        self.w = 0
        self.lock = threading.Lock()

    def push(self, x: np.ndarray):
        with self.lock:
            m = x.size
            if m >= self.n:
                self.buf[:] = x[-self.n:]
                self.w = 0
                return
            end = min(self.n - self.w, m)
            self.buf[self.w:self.w + end] = x[:end]
            if end < m:
                r = m - end
                self.buf[0:r] = x[end:]
                self.w = r
            else:
                self.w = (self.w + m) % self.n

    def read_all(self) -> np.ndarray:
        with self.lock:
            idx = self.w
            return np.concatenate((self.buf[idx:], self.buf[:idx])).copy()


class AudioEngine:
    def __init__(self, sr=44100, block=128):  # lower default blocksize for latency
        self.sr = sr
        self.block = block
        self.bands = [
            EQBand(80,   1.0,  0.0),
            EQBand(300,  1.0,  0.0),
            EQBand(1000, 1.2,  0.0),
            EQBand(4000, 1.2,  0.0),
            EQBand(10000, 1.0,  0.0),
        ]
        self.eq = ParametricEQ(sr, self.bands)
        self.dyn = Dynamics(sr)
        self.fx = FX(sr)
        self.enable_comp = True
        self.enable_delay = False
        self.enable_reverb = False
        self.enable_drive = False

        self.stream = None
        self.running = False
        self.gain = 0.8  # output trim
        self.in_ring = Ring(int(sr * 5.1))  # 5 sec buffer for analysis only
        self.lock = threading.Lock()

        self.input_device = None
        self.output_device = None

    def set_devices(self, in_dev, out_dev):
        self.input_device = in_dev
        self.output_device = out_dev

    def set_bands(self, bands: list[EQBand]):
        with self.lock:
            self.bands = bands
            self.eq.update(self.bands)

    def apply_deltas_db(self, deltas: list[float]):
        with self.lock:
            for i, d in enumerate(deltas):
                self.bands[i].gain_db = float(
                    np.clip(self.bands[i].gain_db + d, -12.0, 12.0))
            self.eq.update(self.bands)

    def set_toggles(self, comp=None, delay=None, reverb=None, drive=None):
        if comp is not None:
            self.enable_comp = bool(comp)
        if delay is not None:
            self.enable_delay = bool(delay)
        if reverb is not None:
            self.enable_reverb = bool(reverb)
        if drive is not None:
            self.enable_drive = bool(drive)

    def callback(self, indata, outdata, frames, time_info, status):
        if status:
            pass
        x = indata[:, 0].astype(np.float32, copy=False)
        self.in_ring.push(x)  # analysis buffer (does NOT add playback latency)
        with self.lock:
            y = x
            # Zero‑latency path: EQ → Drive → Comp → (Delay/Reverb if enabled)
            y = self.eq.process(y)
            if self.enable_drive:
                y = self.fx.saturate(y)
            if self.enable_comp:
                y = self.dyn.compressor(y)
            if self.enable_delay:
                y = self.fx.delay(y)
            if self.enable_reverb:
                y = self.fx.reverb(y)
            y = np.clip(y * self.gain, -1.0, 1.0)
        outdata[:, 0] = y

    def start(self):
        if self.running:
            return
        self.running = True
        self.stream = sd.Stream(
            samplerate=self.sr,
            blocksize=self.block,
            channels=1,
            dtype='float32',
            callback=self.callback,
            device=(self.input_device, self.output_device)
        )
        self.stream.start()

    def stop(self):
        self.running = False
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            finally:
                self.stream = None

# ---------------------------- GUI & Control -------------------------------- #


class LiveAutoEQApp:
    def __init__(self, model_path: str | None = None):
        self.sr = 44100
        # Default model path if not provided
        if not model_path:
            model_path = os.path.join("src", "fma_eq_model_npy.pth")
        self.engine = AudioEngine(self.sr, 128)
        self.an = Analyzer(self.sr)
        self.rules = RuleEQ()
        self.model = ModelWrapper(self.sr, model_path)

        # Fixed blend weights: Rules 0.70, Model 0.30
        self.RULE_W = 0.70
        self.MODEL_W = 0.30
        self.ema = 0.6  # smoothing of applied delta
        self._last_delta = np.zeros(5, dtype=np.float32)

        # Stability detection
        self.delta_hist = deque(maxlen=5)

        # Tk UI
        self.root = tk.Tk()
        self.root.title("Live Auto‑EQ — Rules 70% + AI 30% (1s cadence)")
        self._build_ui()

        # Auto loop
        self._auto_thread = threading.Thread(
            target=self._auto_loop, daemon=True)
        self._auto_run = False

    # ---------------------- UI ---------------------- #
    def _build_ui(self):
        frm = ttk.Frame(self.root, padding=10)
        frm.grid(sticky='nsew')
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Device row (separate input/output)
        dev_row = ttk.Frame(frm)
        dev_row.grid(row=0, column=0, sticky='ew', pady=(0, 8))
        ttk.Label(dev_row, text="Input device:").pack(side='left')
        self.in_dev_var = tk.StringVar()
        self.in_dev_cmb = ttk.Combobox(
            dev_row, textvariable=self.in_dev_var, width=45, state='readonly')
        self.in_dev_cmb.pack(side='left', padx=6)
        ttk.Label(dev_row, text="Output device:").pack(side='left')
        self.out_dev_var = tk.StringVar()
        self.out_dev_cmb = ttk.Combobox(
            dev_row, textvariable=self.out_dev_var, width=45, state='readonly')
        self.out_dev_cmb.pack(side='left', padx=6)
        self._populate_devices()

        # EQ controls
        self.band_vars = []  # tuples of (freq_var, q_var, gain_var)
        grid = ttk.Frame(frm)
        grid.grid(row=1, column=0, sticky='nsew')
        ttk.Label(grid, text="Band").grid(row=0, column=0)
        ttk.Label(grid, text="Freq (Hz)").grid(row=0, column=1)
        ttk.Label(grid, text="Q").grid(row=0, column=2)
        ttk.Label(grid, text="Gain (dB)").grid(row=0, column=3)
        for i, b in enumerate(self.engine.bands):
            ttk.Label(grid, text=f"{i+1}").grid(row=i+1, column=0, sticky='w')
            fvar = tk.DoubleVar(value=b.freq)
            qvar = tk.DoubleVar(value=b.q)
            gvar = tk.DoubleVar(value=b.gain_db)
            self.band_vars.append((fvar, qvar, gvar))
            f_entry = ttk.Entry(grid, textvariable=fvar, width=8)
            q_entry = ttk.Entry(grid, textvariable=qvar, width=6)
            g_scale = ttk.Scale(grid, from_=-12, to=12, orient='horizontal', variable=gvar,
                                command=lambda _=None: self._on_band_change())
            f_entry.grid(row=i+1, column=1)
            q_entry.grid(row=i+1, column=2)
            g_scale.grid(row=i+1, column=3, sticky='ew')
        grid.columnconfigure(3, weight=1)

        # Toggles & gains
        opts = ttk.Frame(frm)
        opts.grid(row=2, column=0, sticky='ew', pady=(8, 8))
        self.comp_var = tk.BooleanVar(value=True)
        self.delay_var = tk.BooleanVar(value=False)
        self.rev_var = tk.BooleanVar(value=False)
        self.drive_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(opts, text="Compressor", variable=self.comp_var,
                        command=self._apply_toggles).pack(side='left')
        ttk.Checkbutton(opts, text="Delay", variable=self.delay_var,
                        command=self._apply_toggles).pack(side='left')
        ttk.Checkbutton(opts, text="Reverb", variable=self.rev_var,
                        command=self._apply_toggles).pack(side='left')
        ttk.Checkbutton(opts, text="Drive", variable=self.drive_var,
                        command=self._apply_toggles).pack(side='left')
        ttk.Label(opts, text=" Output Trim").pack(side='left', padx=(12, 4))
        self.trim_var = tk.DoubleVar(value=0.8)
        ttk.Scale(opts, from_=0.2, to=1.2, orient='horizontal', variable=self.trim_var,
                  command=lambda _=None: self._apply_trim()).pack(side='left', fill='x', expand=True)

        # Smoothing only (weights fixed 70:30)
        auto = ttk.Frame(frm)
        auto.grid(row=3, column=0, sticky='ew')
        ttk.Label(
            auto, text="Auto‑EQ: warm‑up 5s, then 1s cadence — small corrections (Rules 70% + Model 30%)").pack(anchor='w')
        self.ema_var = tk.DoubleVar(value=0.6)
        row2 = ttk.Frame(auto)
        row2.pack(fill='x', pady=(4, 0))
        ttk.Label(row2, text="Smoothing (EMA)").pack(side='left')
        ttk.Scale(row2, from_=0, to=0.95, orient='horizontal', variable=self.ema_var).pack(
            side='left', fill='x', expand=True, padx=6)

        # Buttons
        btns = ttk.Frame(frm)
        btns.grid(row=4, column=0, sticky='ew', pady=(10, 0))
        ttk.Button(btns, text="Start",
                   command=self.start_audio).pack(side='left')
        ttk.Button(btns, text="Stop", command=self.stop_audio).pack(
            side='left', padx=6)
        self.auto_btn = ttk.Button(
            btns, text="Enable Auto‑EQ", command=self.toggle_auto)
        self.auto_btn.pack(side='left', padx=6)
        ttk.Button(btns, text="Reset Gains", command=self.reset_gains).pack(
            side='left', padx=6)
        ttk.Button(btns, text="Test Tone",
                   command=self.play_test_tone).pack(side='right')

        # Status lines
        self.status_var = tk.StringVar(value="Idle")
        self.model_var = tk.StringVar(value=self.model.status)
        ttk.Label(frm, textvariable=self.status_var, anchor='w').grid(
            row=5, column=0, sticky='ew', pady=(8, 0))
        ttk.Label(frm, textvariable=self.model_var, anchor='w', foreground='gray').grid(
            row=6, column=0, sticky='ew', pady=(2, 0))

    # -------- devices -------- #
    def _populate_devices(self):
        devs = sd.query_devices()
        self._in_map = []
        self._out_map = []
        in_names = []
        out_names = []
        for i, d in enumerate(devs):
            name = f"[{i}] {d['name']} — {d['hostapi']}"
            if d.get('max_input_channels', 0) > 0:
                self._in_map.append(i)
                in_names.append(name)
            if d.get('max_output_channels', 0) > 0:
                self._out_map.append(i)
                out_names.append(name)
        self.in_dev_cmb['values'] = in_names
        self.out_dev_cmb['values'] = out_names
        try:
            d_in, d_out = sd.default.device
        except Exception:
            d_in, d_out = None, None
        if d_in in self._in_map:
            self.in_dev_cmb.current(self._in_map.index(d_in))
        elif self._in_map:
            self.in_dev_cmb.current(0)
        if d_out in self._out_map:
            self.out_dev_cmb.current(self._out_map.index(d_out))
        elif self._out_map:
            self.out_dev_cmb.current(0)

    def _selected_devices(self):
        in_idx = self._in_map[self.in_dev_cmb.current()
                              ] if self._in_map else None
        out_idx = self._out_map[self.out_dev_cmb.current()
                                ] if self._out_map else None
        return in_idx, out_idx

    # -------- misc -------- #
    def play_test_tone(self):
        try:
            _, out_idx = self._selected_devices()
            tone = np.sin(2 * np.pi * 440 * np.arange(int(self.sr * 0.5)
                                                      ) / self.sr).astype(np.float32) * 0.3
            sd.play(tone, samplerate=self.sr, device=out_idx)
            sd.wait()
        except Exception as e:
            messagebox.showerror("Playback Error", str(e))

    def _apply_trim(self):
        self.engine.gain = float(self.trim_var.get())

    def _apply_toggles(self):
        self.engine.set_toggles(self.comp_var.get(), self.delay_var.get(
        ), self.rev_var.get(), self.drive_var.get())

    def _on_band_change(self):
        bands = []
        for (fvar, qvar, gvar) in self.band_vars:
            f = float(np.clip(fvar.get(), 20, 20000))
            q = float(np.clip(qvar.get(), 0.1, 12.0))
            g = float(np.clip(gvar.get(), -12.0, 12.0))
            bands.append(EQBand(f, q, g))
        self.engine.set_bands(bands)

    def reset_gains(self):
        for (_, _, gvar) in self.band_vars:
            gvar.set(0.0)
        self._on_band_change()

    def start_audio(self):
        try:
            in_idx, out_idx = self._selected_devices()
            self.engine.set_devices(in_idx, out_idx)
            self.engine.start()
            self.model_var.set(self.model.status)
            self.status_var.set(
                f"Audio running @ 44.1 kHz — In[{in_idx}] → Out[{out_idx}] — speak into the mic…")
        except Exception as e:
            messagebox.showerror("Audio Error", str(e))

    def stop_audio(self):
        self.engine.stop()
        self.status_var.set("Stopped")

    def toggle_auto(self):
        if not self._auto_run:
            self._auto_run = True
            if not self._auto_thread.is_alive():
                self._auto_thread = threading.Thread(
                    target=self._auto_loop, daemon=True)
                self._auto_thread.start()
            self.auto_btn.config(text="Disable Auto‑EQ")
            self.status_var.set("Auto‑EQ enabled (5s warm‑up → 1s cadence)")
        else:
            self._auto_run = False
            self.auto_btn.config(text="Enable Auto‑EQ")
            self.status_var.set("Auto‑EQ disabled")

    def _update_ui_from_engine(self):
        for i, b in enumerate(self.engine.bands):
            self.band_vars[i][0].set(b.freq)
            self.band_vars[i][1].set(b.q)
            self.band_vars[i][2].set(b.gain_db)

    # ---------- Auto loop: warm‑up 5s, then 1s cadence with settling ---------- #
    def _auto_loop(self):
        first = True
        while True:
            if not self._auto_run:
                time.sleep(0.1)
                continue

            x = self.engine.in_ring.read_all()
            peak = np.max(np.abs(x)) + 1e-9
            x = x / peak * 0.7

            feats = self.an.features(x)
            rule_delta = np.array(self.rules.suggest(
                feats, self.engine.bands, first_step=first), dtype=np.float32)

            model_delta = np.zeros(5, dtype=np.float32)
            if self.model and self.model.available:
                y = self.model.predict_params(x)
                if y is not None and y.size == 15:
                    gains = y.reshape(5, 3)[:, 2]
                    cur_g = np.array(
                        [b.gain_db for b in self.engine.bands], dtype=np.float32)
                    model_delta = np.clip(
                        (gains - cur_g) * 0.2, -0.5 if first else -0.25, 0.5 if first else 0.25)

            ema = float(self.ema_var.get())
            blend = 0.70 * rule_delta + 0.30 * model_delta

            # Stability detection: if recent blends are tiny, hold
            self.delta_hist.append(blend.copy())
            hold = False
            if len(self.delta_hist) == self.delta_hist.maxlen:
                m = np.mean([np.linalg.norm(d, ord=2)
                            for d in self.delta_hist])
                if m < (0.04 if first else 0.02):
                    hold = True

            if hold:
                final = np.zeros(5, dtype=np.float32)
            else:
                smoothed = ema * self._last_delta + (1 - ema) * blend
                self._last_delta = smoothed
                cap = 0.5 if first else 0.25
                final = np.clip(smoothed, -cap, +cap)

            self.engine.apply_deltas_db(final.tolist())
            self._update_ui_from_engine()
            txt = ("Auto step dB: " + ", ".join(f"{d:+.2f}" for d in final) +
                   ("  [HOLD]" if hold else "") +
                   f"  tonal={feats.get('tonal', 0):.2f}  AI={'on' if self.model.available else 'off'}")
            self.status_var.set(txt)

            time.sleep(5.0 if first else 1.0)
            first = False

    # ---------------------- Run ---------------------- #
    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.mainloop()

    def on_close(self):
        self._auto_run = False
        self.engine.stop()
        self.root.destroy()

# ------------------------------ Main --------------------------------------- #


if __name__ == "__main__":
    # Allow override via CLI; otherwise default path is src/fma_eq_model_npy.pth
    model_path = "fma_eq_model_npy.pth"
    app = LiveAutoEQApp(model_path)
    app.run()
