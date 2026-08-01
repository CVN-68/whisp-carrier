# -*- mode: python ; coding: utf-8 -*-
# whisp-carier PyInstaller spec file

import sys
from pathlib import Path
import faster_whisper
import ctranslate2
import tokenizers

block_cipher = None

# Collect data files from key packages
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

datas = []
datas += collect_data_files('faster_whisper')
datas += collect_data_files('ctranslate2')
datas += collect_data_files('tokenizers')
datas += collect_data_files('huggingface_hub')
datas += collect_data_files('silero_vad')
datas += collect_data_files('numpy')

# Bundle ffmpeg.exe from Faster-Whisper-XXL
ffmpeg_src = Path(r'C:\Users\Owner1\Downloads\Faster-Whisper-XXL_r245.4_windows\Faster-Whisper-XXL\ffmpeg.exe')
if ffmpeg_src.exists():
    datas += [(str(ffmpeg_src), '.')]

# Hidden imports that PyInstaller misses
hiddenimports = [
    'faster_whisper',
    'faster_whisper.transcribe',
    'faster_whisper.vad',
    'faster_whisper.audio',
    'faster_whisper.tokenizer',
    'faster_whisper.utils',
    'ctranslate2',
    'tokenizers',
    'huggingface_hub',
    'silero_vad',
    'tqdm',
    'numpy',
    'numpy._core',
    'numpy._core._exceptions',
    'numpy._core.multiarray',
    'numpy._core._multiarray_umath',
    'numpy._core._methods',
    'numpy.lib',
    'numpy.lib.stride_tricks',
    'soundfile',
    'av',
    'audio_filter',
    'vad',
    'winsound',
]

a = Analysis(
    ['whisp_carier.py'],
    pathex=[r'C:\Users\Owner1\whisp-carier'],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib', 'tkinter', 'PyQt5', 'PyQt6',
        'PySide2', 'PySide6', 'wx', 'IPython',
        'jupyter', 'notebook', 'pyannote.audio',
        'speechbrain', 'pytorch_lightning',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='whisp-carier',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='whisp-carier',
)
