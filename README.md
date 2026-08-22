# whisp-carrier

RTX 5090 (Blackwell / sm_120) native faster-whisper CLI.  
Drop-in replacement for Faster-Whisper-XXL, built on open-source components.

## Features

- **RTX 5090 native** — torch 2.8.0+cu128, no compatibility fallback
- **Amatsukaze compatible** — same CLI interface as faster-whisper-xxl.exe
- **Model aliases** — `-m anime-whisper` for Japanese anime dialogue; any Hugging Face
  Whisper fine-tune is converted to CTranslate2 on first use
- **Built-in VAD** — silero v6 (via faster-whisper 1.2+)
- **Hallucination loop prevention** — segments that are one phrase or character
  repeated are dropped, which measured 24.3% -> 22.0% CER against ARIB captions
  on nine TV recordings (`--loop_filter false` to keep them)
- **Vocal extraction** — MelBand-Roformer (SOTA quality) and MDX Kim_Vocal_2
- **Audio filters** — loudnorm, bandpass, RNNoise, FFT denoise, noise gate, etc.
- **Subtitle formatting** — sentence splitting, line width and count limits, Japanese kinsoku, re-timing from word timestamps
- **YAML profiles** — switch settings from a config file without touching the caller's options
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
python whisp_carrier.py "video.mp4" -m large-v3 -o source -pp

# High quality
python whisp_carrier.py "video.mp4" -m large-v3 --beam_size 10 --best_of 10 -o source -pp

# With vocal extraction (heavy, for noisy audio)
python whisp_carrier.py "video.mp4" -m large-v3 --ff_vocal_extract mb-roformer -o source -pp

# Standard subtitle formatting (42 chars, 2 lines) / Japanese (16 chars, 2 lines)
python whisp_carrier.py "video.mp4" -l en --standard -o source
python whisp_carrier.py "video.mp4" -l ja --standard_asia -o source
```

## Models

`--model` takes a built-in Whisper size, an alias, a local directory or a Hugging
Face repo id. Run `--list_models` for the alias table.

```powershell
# Japanese anime / visual novel dialogue
python whisp_carrier.py "video.mp4" -m anime-whisper --standard_asia -o source -pp

# Any Whisper fine-tune published in transformers format
python whisp_carrier.py "video.mp4" -m efwkjn/whisper-ja-anime-v0.3 -l ja -o source
```

| Alias | Source | License | Notes |
|-------|--------|---------|-------|
| `anime-whisper` | [litagin/anime-whisper](https://huggingface.co/litagin/anime-whisper) | MIT | kotoba-whisper-v2.0 fine-tuned on 5,300h of anime-style acted dialogue. Reported CER 13.0 against 16.5 for whisper-large-v3 on out-of-training visual novel audio. |
| `kotoba-v2` | [kotoba-tech/kotoba-whisper-v2.0-faster](https://huggingface.co/kotoba-tech/kotoba-whisper-v2.0-faster) | Apache-2.0 | General Japanese, distilled from large-v3. Already CTranslate2, so nothing is converted. |

An alias also carries the option defaults that model wants. `anime-whisper`
selects `--language ja` and `--no_repeat_ngram_size 5`, and warns if an initial
prompt is passed, which is known to send it into hallucination loops. Anything
set on the command line or in the YAML file wins, and every decision is echoed
on `[MODEL]` lines:

```
[MODEL] alias 'anime-whisper' -> litagin/anime-whisper (MIT)
[MODEL] using converted model: ...\_models\ct2-litagin-anime-whisper-float16
[MODEL]   language = 'ja'  (default for anime-whisper)
[MODEL]   no_repeat_ngram_size = 5  (default for anime-whisper)
```

### Conversion

faster-whisper only loads CTranslate2 models, so a transformers-format source is
converted once into `_models/ct2-<name>-<quantization>/` next to the script (or
under `--model_dir`). The conversion needs `transformers`, downloads the weights
through the normal Hugging Face cache, and is skipped on every later run. Use
`--reconvert` to redo it, for example after switching `--compute_type`.

Two details are handled during conversion because getting them wrong fails
quietly rather than loudly:

- **tokenizer.json** is generated when the source does not ship one. Without it
  faster-whisper silently falls back to the whisper-tiny tokenizer and produces
  wrong text with no error.
- **alignment heads** are validated against the real decoder depth. Distilled
  models inherit their teacher's head list, so `--word_timestamps` would index a
  decoder layer that does not exist and kill the process with no traceback.

### Conversion needs the script version

| | exe build | script version |
|---|---|---|
| Normal transcription (`large-v3` etc.) | yes | yes |
| CTranslate2 models | yes | yes |
| Loading an already converted model | yes | yes |
| Converting a transformers model (`-m anime-whisper`) | no | yes |

The exe does not bundle `transformers`. Conversion is a one-off step, and
shipping it would add a few hundred MB and an untested frozen code path to the
distribution. **To use anime-whisper, convert once with the script version and
hand the resulting directory to the exe.**

```powershell
# once, with the script version
python whisp_carrier.py test_speech.wav -m anime-whisper -o . -f srt

# afterwards the exe can use it
whisp-carrier.exe "video.mp4" -m _models\ct2-litagin-anime-whisper-float16
```

Passing `-m anime-whisper` straight to the exe prints those same instructions
and exits 2. Built-in sizes and CTranslate2 models are unaffected.

## Exit codes

Meant to be checked when driving this from a batch script or Amatsukaze.

| Code | Meaning |
|---|---|
| `0` | Every file succeeded |
| `1` | **One or more files failed**, or no input was found |
| `2` | Startup error (config file, model resolution, VAD setup) |

With several inputs, one failure does not stop the rest. The count and the names
of the failed files go to stderr at the end and the exit code becomes 1.
`[whisp-carrier] All done.` is printed only when everything succeeded.

If an audio filter (`--ff_*`) fails, that file is counted as failed and no
transcript is written for it. A run that could not apply the requested filter
does not silently fall back to unfiltered audio.

## Config File (Profiles)

Editing the caller's extra-options string for every experiment is tedious, so
settings can live in a YAML file instead. Rename `whisp-carrier.yaml.example` to
`whisp-carrier.yaml` and place it next to `whisp_carrier.py` (or next to the exe).
It is picked up automatically, so **the calling application needs no changes**.

```yaml
override: true            # let this file win over command line options
active_profile: anime     # switch settings by editing this one line

beam_size: 10             # flat keys apply to every profile
best_of: 10

profiles:
  anime:
    language: ja
    standard_asia: true
```

Keys are the `--help` option names without the leading `--`. With
`override: false` (the code default) the file only fills in options the command
line did not set. Every applied value is echoed on `[CONFIG]` lines so results
can be traced back to the settings that produced them. An unknown key is an
error rather than a warning, since a silently ignored typo would invalidate a
comparison run.

Related options: `--config PATH`, `--no_config`, `--profile NAME`,
`--config_override`.

## Amatsukaze Integration

1. Set whisper path to: `C:\Users\<you>\whisp-carrier\whisp-carrier.bat`
2. Recommended extra options: `--beam_size 10 --best_of 10`
3. Or leave the options empty and drive everything from `whisp-carrier.yaml`
   with `override: true`.

## What the exe build cannot do

The exe covers normal transcription. Three options need the script version, and
each says so when invoked rather than failing obscurely.

| Option | Reason |
|---|---|
| `-m anime-whisper` (any transformers-format model) | `transformers` is not bundled; convert once with the script version, then pass the converted directory |
| `--ff_vocal_extract` | `audio-separator` is not bundled. Bundling it packaged cleanly but failed at runtime inside scipy, reporting a broken scipy installation that was not broken |
| `--vad_method pyannote_v3` / `pyannote_onnx_v3` | `pyannote.audio` is excluded on purpose: it pulls in pytorch-lightning and speechbrain, and measured worse than the built-in silero VAD |

None of this affects accuracy. Every figure this project reports was measured
without any `--ff_*` filter and with the built-in or external silero VAD.

`--realign` needs `stable-ts`, which is not in the default build either; a build
with `WHISP_CARRIER_FULL=1` includes it.

## Architecture

```
whisp_carrier.py            — Main CLI, transcription logic, output writers
audio_filter.py             — ffmpeg filters + vocal extraction (MDX/Roformer)
vad.py                      — Custom VAD backends (pyannote, auditok, webrtc)
loop_filter.py              — Detects and drops looping output
subtitle_format.py          — Subtitle splitting, wrapping and re-timing
whisp_models.py             — Model aliases and CTranslate2 conversion
whisp_config.py             — YAML config file / profiles
whisp_vad_patch.py          — Swaps the built-in VAD model / segment source
whisp-carrier.bat           — Launcher for Amatsukaze compatibility
whisp-carrier.yaml.example  — Sample config file
THIRD-PARTY-NOTICES.md      — Everything bundled, with licences
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
| Anime Whisper | Japanese anime dialogue model (`-m anime-whisper`) | https://huggingface.co/litagin/anime-whisper |
| Kotoba-Whisper | Japanese distilled Whisper, base of Anime Whisper | https://huggingface.co/kotoba-tech/kotoba-whisper-v2.0 |
| PyTorch | GPU computation (CUDA 12.8 / sm_120 support) | https://pytorch.org/ |
| silero-vad | Voice Activity Detection model | https://github.com/snakers4/silero-vad |
| audio-separator | Vocal extraction (MDX / Mel-Band-Roformer) | https://github.com/karaokenerds/python-audio-separator |
| stable-ts | Timestamp realignment (experimental) | https://github.com/jianfch/stable-ts |
| CTranslate2 | Efficient transformer inference | https://github.com/OpenNMT/CTranslate2 |
| ffmpeg | Audio preprocessing & filtering | https://ffmpeg.org/ |

Inspired by [Faster-Whisper-XXL](https://github.com/Purfview/whisper-standalone-win) (Purfview) — a proprietary Whisper CLI with RTX 5090 support.  
whisp-carrier reimplements equivalent functionality using only open-source components.

## License

MIT
