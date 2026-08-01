# whisp-carier

RTX 5090 (Blackwell / sm_120) native faster-whisper CLI.  
Drop-in replacement for Faster-Whisper-XXL, built on open-source components.

## Features

- **RTX 5090 native** — torch 2.8.0+cu128, no compatibility fallback
- **Amatsukaze compatible** — same CLI interface as faster-whisper-xxl.exe
- **Built-in VAD** — silero v6 (via faster-whisper 1.2+)
- **Hallucination loop prevention** — automatic detection and suppression
- **Vocal extraction** — MelBand-Roformer (SOTA quality) and MDX Kim_Vocal_2
- **Audio filters** — loudnorm, bandpass, RNNoise, FFT denoise, noise gate, etc.
- **Multiple output formats** — SRT, VTT, JSON, TXT, TSV, LRC

## Requirements

- Windows 10/11 (x64)
- Python 3.11
- NVIDIA RTX GPU with CUDA 12.8+ driver
- CUDA Toolkit 12.8
- ffmpeg in PATH

## Installation

```powershell
pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## Usage

```powershell
# Basic
python whisp_carier.py "video.mp4" -m large-v3 -o source -pp

# High quality
python whisp_carier.py "video.mp4" -m large-v3 --beam_size 10 --best_of 10 -o source -pp

# With vocal extraction (heavy, for noisy audio)
python whisp_carier.py "video.mp4" -m large-v3 --ff_vocal_extract mb-roformer -o source -pp
```

## Amatsukaze Integration

1. Set whisper path to: `C:\Users\<you>\whisp-carier\whisp-carier.bat`
2. Recommended extra options: `--beam_size 10 --best_of 10`

## Architecture

```
whisp_carier.py   — Main CLI, transcription logic, output writers
audio_filter.py   — ffmpeg filters + vocal extraction (MDX/Roformer)
vad.py            — Custom VAD backends (pyannote, auditok, webrtc)
whisp-carier.bat  — Launcher for Amatsukaze compatibility
```

## Status

**Active — Evaluation phase.** Feedback welcome.  
Looking for integration testing with Amatsukaze.

## Acknowledgements / Based On

This project builds on the following open-source projects:

| Project | Role | Link |
|---------|------|------|
| OpenAI Whisper | Original speech recognition model | https://github.com/openai/whisper |
| faster-whisper | CTranslate2-based Whisper inference engine | https://github.com/SYSTRAN/faster-whisper |
| PyTorch | GPU computation (CUDA 12.8 / sm_120 support) | https://pytorch.org/ |
| silero-vad | Voice Activity Detection model | https://github.com/snakers4/silero-vad |
| audio-separator | Vocal extraction (MDX / Mel-Band-Roformer) | https://github.com/karaokenerds/python-audio-separator |
| stable-ts | Timestamp realignment (experimental) | https://github.com/jianfch/stable-ts |
| CTranslate2 | Efficient transformer inference | https://github.com/OpenNMT/CTranslate2 |
| ffmpeg | Audio preprocessing & filtering | https://ffmpeg.org/ |

Inspired by [Faster-Whisper-XXL](https://github.com/Purfview/whisper-standalone-win) (Purfview) — a proprietary Whisper CLI with RTX 5090 support.  
whisp-carier reimplements equivalent functionality using only open-source components.

## License

MIT
