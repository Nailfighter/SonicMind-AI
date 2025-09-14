# -*- mode: python ; coding: utf-8 -*-

import os
import sys
from pathlib import Path

# Get the current working directory (should be backend directory when running PyInstaller)
backend_dir = Path.cwd()
project_root = backend_dir.parent

# Add the backend directory to the path
sys.path.insert(0, str(backend_dir))

# Define data files to include
datas = [
    # Include the model file
    (str(backend_dir / 'fma_eq_model_npy.pth'), '.'),
    # Include any other data files if needed
]

# Define hidden imports (modules that PyInstaller might miss)
hiddenimports = [
    'socketio',
    'eventlet',
    'eventlet.wsgi',
    'flask',
    'numpy',
    'scipy',
    'sounddevice',
    'PIL',
    'cv2',
    'torch',
    'torchvision',
    'open_clip',
    'auto_eq_system',
    'instrument_detection',
    'material_detection',
    'threading',
    'time',
    'json',
    'socketio.server',
    'socketio.event',
    'flask.app',
    'eventlet.listen',
    'eventlet.wsgi.server'
]

# Define binaries (external executables if any)
binaries = []

# Analysis configuration
a = Analysis(
    ['main.py'],
    pathex=[str(backend_dir)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

# Remove duplicate files
pyz = PYZ(a.pure, a.zipped_data, cipher=None)

# Create the executable
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='SonicMind-Backend',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # Keep console for debugging
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Add icon path if you have one
    version=None,  # Add version info if needed
)
