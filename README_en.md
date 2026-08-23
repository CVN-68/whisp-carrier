# whisp-carrier

[日本語](README.md) | **English**

RTX 5090 (Blackwell / sm_120) native faster-whisper CLI.  
Drop-in replacement for Faster-Whisper-XXL, built on open-source components.

**This is the manual for the distributed exe build.** Running it from Python is a
development build with no guarantees; its setup lives in `HANDOVER.md`. What the
exe cannot do is listed [below](#what-the-exe-build-cannot-do).

> The Japanese [README.md](README.md) is the fuller document: it also carries the
> complete option reference, the Amatsukaze field examples and the caveat list.
> `HANDOVER.md` holds the design decisions and the raw measurements, and is
> Japanese only (its commands are copy-pasteable regardless).

## Features

- **RTX 5090 native** — torch 2.8.0+cu128, no compatibility fallback
- **Amatsukaze compatible** — same CLI interface as faster-whisper-xxl.exe
- **Model aliases** — `-m anime-whisper` for Japanese anime dialogue; any Hugging Face
  Whisper fine-tune is converted to CTranslate2 on first use
- **TEN VAD by default** (Apache-2.0) — measured better than silero on every one of
  15 TV recordings: 19.3% -> 16.1% whole-region CER over nine 24-minute episodes and
  31.1% -> 21.9% over four children's programmes. The built-in silero paths are still
  selectable with `--vad_method`
- **Hallucination loop prevention** — segments that are one phrase or character
  repeated are dropped, which measured 24.3% -> 22.0% CER against ARIB captions
  on nine TV recordings (measured under the silero default and the pre-fix scoring
  script; `--loop_filter false` to keep them)
- **Vocal extraction** — MelBand-Roformer (SOTA quality) and MDX Kim_Vocal_2
- **Audio filters** — loudnorm, bandpass, RNNoise, FFT denoise, noise gate, etc.
- **Subtitle formatting** — sentence splitting, line width and count limits, Japanese kinsoku, re-timing from word timestamps
- **YAML profiles** — switch settings from a config file without touching the caller's options
- **Multiple output formats** — SRT, VTT, JSON, TXT, TSV, LRC

## Getting started

- Windows 10/11 (x64)
- NVIDIA RTX GPU with a CUDA 12.8+ driver
- **No Python, no CUDA Toolkit and no ffmpeg needed** (an LGPL ffmpeg is bundled)

Unpack the archive and run the exe. `LICENSE`, `LICENSE.ffmpeg.txt`,
`LICENSE.ten-vad.*.txt`, `THIRD-PARTY-NOTICES.md` and
`whisp-carrier.yaml.example` sit next to it. The config file is optional.

## Usage

```powershell
# Basic
whisp-carrier.exe "video.mp4" -m large-v3 -o source -pp

# Standard subtitle formatting (42 chars, 2 lines) / Japanese (16 chars, 2 lines)
whisp-carrier.exe "video.mp4" -l en --standard -o source
whisp-carrier.exe "video.mp4" -l ja --standard_asia -o source

# Widening the beam is not an upgrade here: measured equal or worse than the
# default 5 in 8 of 9 episodes. Left as an example of a knob, not a recommendation.
whisp-carrier.exe "video.mp4" -m large-v3 --beam_size 10 --best_of 10 -o source -pp
```

## Models

`--model` takes a built-in Whisper size, an alias, a local directory or a Hugging
Face repo id. Run `--list_models` for the alias table.

```powershell
# A built-in size
whisp-carrier.exe "video.mp4" -m large-v3 -l ja -o source

# An already converted directory (converting itself needs the script version)
whisp-carrier.exe "video.mp4" -m _models\ct2-litagin-anime-whisper-float16 -o source
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

The exe does not bundle `transformers`. Conversion is a one-off step, and
shipping it would add a few hundred MB and an untested frozen code path to the
distribution. **To use anime-whisper, convert once with the development build and
hand the resulting `_models/ct2-*` directory to the exe**; the procedure is in
`HANDOVER.md`.

**Passing `-m anime-whisper` straight to the exe prints those instructions and
exits 2.** Built-in sizes, CTranslate2 models and already converted directories
all work as usual. The full list of what differs is in
[what the exe build cannot do](#what-the-exe-build-cannot-do).

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

1. Set whisper path to: `C:\Users\<you>\whisp-carrier\whisp-carrier.exe`
2. **Leave the whisper-option field empty.** The defaults are exactly the
   configuration the figures above were measured with: TEN VAD at 0.75,
   `beam_size` / `best_of` 5, clip routing, loop suppression on.
   Add `--standard_asia` if you want the subtitles broken into 16-character
   two-line cues, and `--language ja` if short split fragments are being
   misdetected. Model and output format are supplied by Amatsukaze.

   **Set whisper-model to `large-v3` explicitly.** Its `自動` / auto setting passes
   `-m large-v3-turbo` (verified), and turbo keeps large-v3's encoder while cutting
   the decoder from 32 layers to 4. Everything this project is measured on is
   decoder-side work: loop hallucinations, segments running past 30s, and the
   timestamp tokens that place each cue. `condition_on_previous_text` is off here
   by design, so there is no carried context to recover with. turbo is not
   measured against the reference set, so the figures above would not apply.
   Passing turbo or a distil model prints a `[MODEL] NOTE:` at startup.
   The `未指定` / unspecified setting omits `-m` altogether, which lets a `model:`
   entry in the config file apply without needing `override`.
3. **Do not pass `--beam_size 10 --best_of 10`.** An earlier revision of this
   document recommended it; measured over the same nine episodes it was equal to
   or worse than beam 5 in 8 of 9 files (22.0% -> 22.3% CER) for 2.8% more time.
   Clear it if it is already in the field.
4. A config file is optional. `whisp-carrier.yaml` next to the exe with
   `override: true` lets you switch settings without touching Amatsukaze, which
   is convenient for A/B runs but not needed for normal use.

## Scope: opening/ending themes and singing are out of scope

**The target is dialogue in the episode body. Themes, insert songs and singing
scenes are out of scope, and not suppressing them is not guaranteed either.**
Under some conditions the output does contain lyric-like text.

This is shared behaviour across Whisper-based transcription, not something
specific to this project. Characters produced inside the same song windows on
the same material:

| Implementation | Characters inside song windows |
|---|---|
| whisp-carrier (current default, TEN VAD) | 2,274 |
| Faster-Whisper-XXL r245.4 | 630 |
| whisp-carrier (previous default, silero) | 54 |

The old silero default produced almost nothing there, but that was a **detection
miss rather than a feature**, and the same miss dropped dialogue as well.
Recovering that dialogue with TEN VAD raises the song-window output as a
side effect.

**No song filter is shipped.** Dropping those regions was measured and made
accuracy worse (whole-region CER 15.5% -> 16.1%). Some broadcasts do caption
their songs and the practice is not consistent, so a mechanical filter also
removes wanted captions.

If lyrics are unwanted, deleting the cues for those time ranges afterwards is
the reliable route. The `[VAD]` log lines are the signal to look at.

For the same reason, **stock phrases such as 「ご視聴ありがとうございました」
("thank you for watching") occasionally appear over silence.** Whisper emits
high-frequency training phrases against non-speech, and Faster-Whisper-XXL does
the same (11 occurrences against this implementation's 43 over the same nine
references). No filter for them is shipped, deliberately: **removing 0.3 to
0.6 points of CER is not worth a mechanism that could delete a real line spoken
by an in-universe broadcaster or streamer.** Measurement #20 in HANDOVER.md has
the details.

## What the exe build cannot do

**The exe is the supported path.** Transcription, VAD, loop suppression, the
over-30s repair, subtitle formatting and the config file all work there. Three
options need the development build, which carries no guarantees and is documented
in `HANDOVER.md`. Each of them says so when invoked rather than failing obscurely.

| Option | exe | script | Reason it is not bundled |
|---|:---:|:---:|---|
| `-m anime-whisper` (any transformers-format model) | no | yes | `transformers` is not bundled; convert once with the script version, then pass the converted directory |
| `--ff_vocal_extract` | no | yes | `audio-separator` is not bundled. Bundling it packaged cleanly but failed at runtime inside scipy, reporting a broken scipy installation that was not broken |
| `--realign` | no | yes | `stable-ts` is not in the default build; `WHISP_CARRIER_FULL=1` includes it. It is skipped anyway whenever subtitle formatting is on, which the recommended settings enable |
| `--vad_method pyannote_v3` / `pyannote_onnx_v3` | no | no | `pyannote.audio` is excluded on purpose: it pulls in pytorch-lightning and speechbrain, and measured worse than the built-in silero VAD |

**None of this affects accuracy.** Every figure this project reports was measured
without any `--ff_*` filter, on `large-v3`, through the default TEN VAD path,
and none of the three appear in it.

## Architecture

```
whisp_carrier.py            — Main CLI, transcription logic, output writers
audio_filter.py             — ffmpeg filters + vocal extraction (MDX/Roformer)
vad.py                      — External VAD backends: ten (default), silero,
                              precomputed, pyannote, auditok, webrtc
loop_filter.py              — Detects and drops looping output
subtitle_format.py          — Subtitle splitting, wrapping, re-timing, sanitizing
whisp_models.py             — Model aliases and CTranslate2 conversion
whisp_config.py             — YAML config file / profiles
whisp_vad_patch.py          — Swaps the built-in VAD model / segment source
whisp-carrier.bat           — Launcher for Amatsukaze compatibility (script version)
whisp-carrier.yaml.example  — Sample config file
whisp_carrier.spec          — PyInstaller build; ffmpeg licence check lives here
eval/                       — Accuracy harness; independent of the CLI, not in the exe
THIRD-PARTY-NOTICES.md      — Everything bundled, with licences
```

`HANDOVER.md` has the fuller version of this, together with the dependency table
and the reasoning behind each default. It is Japanese only.

## Measured accuracy

**Scored against ARIB broadcast captions on 15 Japanese TV recordings**, default
settings on both sides, no audio filters, singing excluded from scoring.

| Material | Files | Reference chars | whisp-carrier (default) | Faster-Whisper-XXL r245.4 |
|---|---|---|---|---|
| 24-minute episodes | 9 | 35,036 | **16.1%** | 20.5% |
| Children's programmes | 4 | 16,704 | **21.9%** | 33.5% |
| Marathon broadcast (5h22m) | 1 | 37,987 | **16.1%** | not measured |
| Marathon broadcast (4h52m) | 1 | 50,255 | **14.1%** | not measured |

Figures are whole-region CER (character error rate, lower is better).
**No single number describes this.** Per file the spread is 8.7% to 30.5%,
and it tracks the content: 8.7 to 14.0% for dialogue-driven shows,
23 to 25% for children's programmes, 30.5% for the worst case
(a series where speech covers only 23% of the runtime).

**The error is mostly missed speech rather than wrong text.** Over the nine
episodes, 86.6% of the reference was recovered and 90.0% of the produced text
matched it. The weaker files are not less precise, they output less.

Structural differences, which matter more in practice for subtitles:

| | whisp-carrier | Faster-Whisper-XXL r245.4 |
|---|---|---|
| Segments longer than 30s (nine episodes) | **0** | 9, totalling 602s |
| Long segments carrying almost no text | 1 | 15 |
| Hallucination loop characters left (nine episodes) | **35** | 53 |
| Runs on RTX 5090 at defaults | **yes** (float16) | no, needs `-ct float32` (~2x slower) |
| One 24-minute episode | 70-100s | 175s |
| One 5h22m broadcast | 1214s (6.3% of runtime) | not measured |

There is no degradation on long material: the two marathon broadcasts scored
16.1% and **14.1%**, the latter being the best result across the whole corpus.

### How the comparison was set up

**Faster-Whisper-XXL ran at its defaults, and those defaults are the recommendation.**
The [upstream README](https://github.com/Purfview/whisper-standalone-win/blob/main/README.md)
states that its defaults are already tuned for transcribing movies and publishes no
separate list of recommended options; the usage examples only add `--sentence`,
`--standard` or `--batch_recursive`. (Summarised rather than quoted, for licence
compliance.)

**Those defaults are not bare.** Three hallucination countermeasures are on out of
the box.

| Setting | Faster-Whisper-XXL r245.4 default | whisp-carrier default |
|---|---|---|
| VAD | on (silero) | on (**TEN VAD**) |
| `vad_min_silence_duration_ms` / `vad_speech_pad_ms` | 3000 / 900 | 3000 / 900 (adopted from upstream) |
| `beam_size` / `best_of` | 5 / 5 | 5 / 5 |
| `patience` | 2.0 | 2.0 |
| `condition_on_previous_text` | **True** | **False** (measured, see below) |
| Known-hallucination list | **on** | no equivalent |
| large-v3 pseudo-VAD threshold offsets | **on** | no equivalent |
| Prompt re-injection / duplicate-prompt suppression | **on** | no equivalent |
| Audio filters (`--ff_*`) | all off | not used |
| Subtitle formatting presets | off | `standard_asia: true` recommended |
| Loop suppression | — | **on** (no upstream equivalent) |

The full command lines were:

```powershell
# Faster-Whisper-XXL r245.4
faster-whisper-xxl.exe <wav> -m large-v3 -ct float32 -f json --language ja --beep_off

# whisp-carrier (defaults; --no_config keeps any config file out of the run)
python whisp_carrier.py <wav> -m large-v3 -f json --no_config --beep_off
```

**Two departures, both in XXL's favour or neutral.** `-ct float32` is unavoidable
because float16 crashes in cuBLAS on an RTX 5090; it is the same weights, so
accuracy is if anything better (at roughly half the speed). `--language ja` was
passed explicitly; on this implementation naming the language did not change a
single segment, since detection already returned ja at 92-100%.

**Matching the formatting would not change the headline figure.** Formatting adds
line breaks and cue splits, and scoring strips all whitespace from *both* sides,
so whole-region CER is unaffected. Per-30s-block CER does move with formatting,
and both sides ran unformatted there.

**Why `condition_on_previous_text` is False here.** Enabling it was measured:
coverage on a low-speech-density episode fell from 59.7% to 21.3%, inference took
5.9x longer, and output stopped at 18:16 of a 30:00 runtime. The upstream default
cannot simply be borrowed.

**Not measured:** XXL with audio filters (`--ff_rnndn_sh` and friends) or
`--batched`. Both are off by default and absent from the author's recommendations,
and this project has settled on not using filters, so enabling them on one side
only would make the comparison asymmetric.

`HANDOVER.md` documents the corpus, the metrics and how they are computed.

## Status

**Active — Evaluation phase.** Feedback welcome.

Amatsukaze integration is verified on the exe build. The same episode that is in
the measurement corpus was run through Amatsukaze in production and matched the
harness exactly: same VAD region count, same seconds of detected speech, same
segment count, same suppressed loop. The figures above are what that path
produces. **Only the exe build is supported**; running it from Python is a
development build with no guarantees.

## Acknowledgements / Based On

This project builds on the following open-source projects. **The complete list of
everything bundled, with licence texts, is in
[THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md)** (which also ships in the archive).

| Project | Role | Licence | Link |
|---------|------|---------|------|
| OpenAI Whisper | Original speech recognition model | MIT | https://github.com/openai/whisper |
| faster-whisper | CTranslate2-based Whisper inference engine | MIT | https://github.com/SYSTRAN/faster-whisper |
| CTranslate2 | Efficient transformer inference | MIT | https://github.com/OpenNMT/CTranslate2 |
| **TEN VAD** | **Default voice activity detection** (`--vad_method ten`); its DLL is bundled | **Apache-2.0** | https://github.com/TEN-framework/ten-vad |
| silero-vad | Alternative VAD (the previous default); faster-whisper's built-in v6 ONNX is the same model | MIT | https://github.com/snakers4/silero-vad |
| onnxruntime | Runs that ONNX model | MIT | https://onnxruntime.ai |
| PyTorch | GPU computation (CUDA 12.8 / sm_120 support) | BSD-3-Clause | https://pytorch.org/ |
| PyAV | Audio decoding (through faster-whisper) | BSD-3-Clause | https://github.com/PyAV-Org/PyAV |
| ffmpeg | Audio preprocessing and filtering (run as a separate process) | **LGPL v3** (the bundled build) | https://ffmpeg.org/ |
| libsndfile | Audio I/O (through `soundfile`) | **LGPL-2.1-or-later** | https://github.com/libsndfile/libsndfile |
| PyInstaller | Freezing to an exe (its bootloader ships inside) | GPL-2.0 with an exception for distributing frozen apps | https://github.com/pyinstaller/pyinstaller |
| PyYAML | Config file parsing | MIT | https://pyyaml.org/ |
| Anime Whisper | Japanese anime dialogue model (`-m anime-whisper`) | MIT | https://huggingface.co/litagin/anime-whisper |
| Kotoba-Whisper | Japanese distilled Whisper, base of Anime Whisper | Apache-2.0 | https://huggingface.co/kotoba-tech/kotoba-whisper-v2.0 |
| audio-separator | Vocal extraction (MDX / Mel-Band-Roformer); **script version only** | MIT | https://github.com/karaokenerds/python-audio-separator |
| stable-ts | Timestamp realignment (experimental); **script version only** | MIT | https://github.com/jianfch/stable-ts |

NVIDIA CUDA / cuDNN and Intel OpenMP are also bundled, arriving inside the `torch`
and `ctranslate2` wheels; their redistribution terms are in THIRD-PARTY-NOTICES.md.

Inspired by [Faster-Whisper-XXL](https://github.com/Purfview/whisper-standalone-win) (Purfview) — a proprietary Whisper CLI with RTX 5090 support.  
whisp-carrier reimplements equivalent functionality using only open-source components.

## License

MIT
