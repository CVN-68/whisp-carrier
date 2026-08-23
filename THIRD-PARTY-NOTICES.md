# Third-party notices

whisp-carrier itself is MIT licensed (see `LICENSE`). This file lists everything
else that ships inside a built distribution, with its origin and licence.

Scope: the PyInstaller `onedir` build produced by `whisp_carrier.spec`, i.e. the
contents of `dist/whisp-carrier/` (4.78 GB, of which roughly 4.5 GB is PyTorch
and the CUDA libraries). Running from source pulls the same Python packages from
your environment and additionally enables model conversion; only the bundled
`ffmpeg.exe` is specific to the build.

Regenerating this list is described at the end.

---

## 1. ffmpeg (separate executable)

`ffmpeg.exe` ships in `_internal/` and is invoked as a child process by
`audio_filter.py` and `vad.py`. It is not linked into the executable.

| | |
|---|---|
| Version | FFmpeg `n8.1.2-44-g7c533d0f86-20260820` |
| Licence | **LGPL v3 or later** (`--enable-version3`, without `--enable-gpl`) |
| Build provider | [BtbN/FFmpeg-Builds](https://github.com/BtbN/FFmpeg-Builds) |
| Upstream project | [FFmpeg](https://ffmpeg.org/) |
| Source code | https://github.com/FFmpeg/FFmpeg (tag `n8.1.2`); build recipe at https://github.com/BtbN/FFmpeg-Builds |
| Licence text | shipped as `LICENSE.ffmpeg.txt` beside the executable |
| SHA-256 | `8ee152edc79f7ba99969b7fb590cfde438b46cb383952dec87e754db83788572` |

GPL-only and non-free components are disabled in this build (no libx264,
libx265, libxvid, libxavs2, libdavs2, librubberband, libvidstab, frei0r,
libfdk-aac or AviSynth). `whisp_carrier.spec` reads the `configuration:` line of
the binary at build time and aborts if `--enable-gpl` or `--enable-nonfree` is
present, so a GPL build cannot end up here by accident. The hash above is
pinned in the spec and reported on every build.

FFmpeg does not distribute Windows binaries itself; ffmpeg.org points to
third-party build providers, of which BtbN is one. This is therefore a binary
from a provider FFmpeg refers people to, not one signed by the FFmpeg project.
Full provenance, including verification against the publisher's
`checksums.sha256` and the GitHub release metadata, is in
`_tools/ffmpeg/PROVENANCE.txt`.

---

## 2. Native runtime libraries

These arrive inside the `torch`, `ctranslate2` and `soundfile` wheels rather
than being fetched separately. They dominate the size of the distribution.

| Component | Origin | Licence |
|---|---|---|
| CUDA Runtime (`cudart64_12.dll`) | NVIDIA, via `torch==2.8.0+cu128` | [NVIDIA CUDA Toolkit EULA](https://docs.nvidia.com/cuda/eula/) — redistribution permitted under its terms |
| cuBLAS / cuBLASLt | NVIDIA, via `torch` | NVIDIA CUDA Toolkit EULA |
| cuFFT, cuRAND, cuSOLVER, cuSPARSE | NVIDIA, via `torch` | NVIDIA CUDA Toolkit EULA |
| NVRTC (`nvrtc64_120_0.dll`) | NVIDIA, via `torch` | NVIDIA CUDA Toolkit EULA |
| cuDNN 9 (`cudnn*64_9.dll`) | NVIDIA, via `torch` and `ctranslate2` | [NVIDIA cuDNN licence](https://docs.nvidia.com/deeplearning/cudnn/latest/reference/eula.html) |
| Intel OpenMP (`libiomp5md.dll`) | Intel, via `ctranslate2` | [Intel Simplified Software License](https://www.intel.com/content/www/us/en/developer/articles/license/end-user-license-agreement.html) |
| CTranslate2 (`ctranslate2.dll`) | [OpenNMT/CTranslate2](https://github.com/OpenNMT/CTranslate2) | MIT |
| **libsndfile** (`_soundfile_data/libsndfile_x64.dll`) | [libsndfile](https://github.com/libsndfile/libsndfile), via `soundfile` | **LGPL-2.1-or-later** — see section 5 |
| Microsoft Visual C++ runtime, OpenSSL (`libssl-3.dll`, `libcrypto-3.dll`), SQLite, expat, libffi | CPython 3.11 redistributables | Respective upstream licences (MS runtime redistribution terms, Apache-2.0, public domain, MIT) |

---

## 3. Models and weights

| Item | Bundled? | Origin | Licence |
|---|---|---|---|
| TEN VAD (`ten_vad/lib/Windows/x64/ten_vad.dll`) | **Yes** | [TEN-framework/ten-vad](https://github.com/TEN-framework/ten-vad), the prebuilt library inside the `ten-vad` wheel | Apache-2.0 |
| Silero VAD v6 (`faster_whisper/assets/silero_vad_v6.onnx`) | **Yes** | [snakers4/silero-vad](https://github.com/snakers4/silero-vad), redistributed inside the `faster-whisper` package | MIT |
| Whisper `large-v3` and other sizes | No | Downloaded at first run from [Hugging Face](https://huggingface.co/Systran) | MIT (weights, OpenAI) |
| `litagin/anime-whisper` | No | [Hugging Face](https://huggingface.co/litagin/anime-whisper); the exe cannot convert it, see section 6 | See the model card |
| Audio separation models | No | Downloaded by `audio-separator` on demand | Varies per model; check before redistributing output |

No model weights are included in the executable.

---

## 4. Python packages

| Package | Version | Licence | Project |
|---|---|---|---|
| aiohappyeyeballs | 2.7.1 | PSF-2.0 | https://github.com/aio-libs/aiohappyeyeballs |
| aiohttp | 3.14.3 | Apache-2.0 AND MIT | https://github.com/aio-libs/aiohttp |
| aiosignal | 1.4.0 | Apache-2.0 | https://github.com/aio-libs/aiosignal |
| anyio | 4.14.2 | MIT | https://anyio.readthedocs.io/ |
| attrs | 26.1.0 | MIT | https://www.attrs.org/ |
| audio-separator | 0.44.5 | MIT | https://github.com/karaokenerds/python-audio-separator |
| audioread | 3.1.0 | MIT | https://github.com/beetbox/audioread |
| av (PyAV) | 18.0.0 | BSD-3-Clause | https://github.com/PyAV-Org/PyAV |
| certifi | 2026.7.22 | **MPL-2.0** | https://github.com/certifi/python-certifi |
| cffi | 2.1.0 | MIT-0 | https://cffi.readthedocs.io/ |
| charset-normalizer | 3.4.9 | MIT | https://github.com/jawah/charset_normalizer |
| click | 8.4.2 | BSD-3-Clause | https://click.palletsprojects.com/ |
| colorama | 0.4.6 | BSD-3-Clause | https://github.com/tartley/colorama |
| ctranslate2 | 4.8.1 | MIT | https://github.com/OpenNMT/CTranslate2 |
| Cython | 3.2.9 | Apache-2.0 | https://cython.org/ |
| einops | 0.8.2 | MIT | https://github.com/arogozhnikov/einops |
| faster-whisper | 1.2.1 | MIT | https://github.com/SYSTRAN/faster-whisper |
| filelock | 3.29.0 | MIT | https://py-filelock.readthedocs.io |
| frozenlist | 1.8.0 | Apache-2.0 | https://github.com/aio-libs/frozenlist |
| fsspec | 2026.4.0 | BSD-3-Clause | https://filesystem-spec.readthedocs.io/ |
| h11 | 0.16.0 | MIT | https://github.com/python-hyper/h11 |
| hf-xet | 1.5.2 | Apache-2.0 | https://huggingface.co/docs/hub/xet/index |
| httpcore | 1.0.9 | BSD-3-Clause | https://www.encode.io/httpcore |
| httpx | 0.28.1 | BSD-3-Clause | https://github.com/encode/httpx |
| huggingface_hub | 1.26.0 | Apache-2.0 | https://github.com/huggingface/huggingface_hub |
| idna | 3.18 | BSD-3-Clause | https://github.com/kjd/idna |
| Jinja2 | 3.1.6 | BSD-3-Clause | https://jinja.palletsprojects.com/ |
| markdown-it-py | 4.2.0 | MIT | https://markdown-it-py.readthedocs.io |
| MarkupSafe | 3.0.3 | BSD-3-Clause | https://palletsprojects.com/ |
| mdurl | 0.1.2 | MIT | https://github.com/executablebooks/mdurl |
| mpmath | 1.3.0 | BSD-3-Clause | http://mpmath.org/ |
| multidict | 6.7.1 | Apache-2.0 | https://github.com/aio-libs/multidict |
| networkx | 3.6.1 | BSD-3-Clause | https://networkx.org/ |
| numpy | 2.4.4 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | https://numpy.org |
| onnxruntime | 1.28.0 | MIT | https://onnxruntime.ai |
| openai-whisper | 20250625 | MIT | https://github.com/openai/whisper |
| packaging | 26.2 | Apache-2.0 OR BSD-2-Clause | https://packaging.pypa.io/ |
| pillow | 12.2.0 | MIT-CMU | https://python-pillow.github.io/ |
| propcache | 0.5.2 | Apache-2.0 | https://github.com/aio-libs/propcache |
| protobuf | 7.35.1 | BSD-3-Clause | https://protobuf.dev/ |
| pycparser | 3.0 | BSD-3-Clause | https://github.com/eliben/pycparser |
| pydub | 0.25.1 | MIT | http://pydub.com |
| Pygments | 2.20.0 | BSD-2-Clause | https://pygments.org |
| pywin32 | 312 | PSF-2.0 | https://github.com/mhammond/pywin32 |
| PyYAML | 6.0.3 | MIT | https://pyyaml.org/ |
| requests | 2.34.2 | Apache-2.0 | https://requests.readthedocs.io |
| rich | 15.0.0 | MIT | https://rich.readthedocs.io/ |
| safetensors | 0.8.0 | Apache-2.0 | https://github.com/huggingface/safetensors |
| scipy | 1.17.1 | BSD-3-Clause | https://scipy.org/ |
| setuptools | 65.5.0 | MIT | https://github.com/pypa/setuptools |
| silero-vad | 6.2.1 | MIT | https://github.com/snakers4/silero-vad |
| ten-vad | 1.0.6.8 | Apache-2.0 | https://github.com/TEN-framework/ten-vad |
| six | 1.17.0 | MIT | https://github.com/benjaminp/six |
| soundfile | 0.14.0 | BSD-3-Clause | https://github.com/bastibe/python-soundfile |
| stable-ts | 2.19.1 | MIT | https://github.com/jianfch/stable-ts |
| sympy | 1.14.0 | BSD-3-Clause | https://sympy.org |
| tensorboardX | 2.6.5 | MIT | https://github.com/lanpa/tensorboardX |
| threadpoolctl | 3.6.0 | BSD-3-Clause | https://github.com/joblib/threadpoolctl |
| tokenizers | 0.22.2 | Apache-2.0 | https://github.com/huggingface/tokenizers |
| torch | 2.8.0+cu128 | BSD-3-Clause | https://pytorch.org/ |
| torchaudio | 2.8.0+cu128 | BSD-2-Clause | https://github.com/pytorch/audio |
| tqdm | 4.67.1 | **MPL-2.0** AND MIT | https://tqdm.github.io |
| typing_extensions | 4.15.0 | PSF-2.0 | https://github.com/python/typing_extensions |
| urllib3 | 2.7.0 | MIT | https://github.com/urllib3/urllib3 |
| yarl | 1.24.5 | Apache-2.0 | https://github.com/aio-libs/yarl |

Some entries are only partially present. PyInstaller stores pure-Python modules
in the PYZ archive and includes only what the import graph reaches, so
`audio-separator` and `stable-ts` contribute their top-level `__init__` without
the rest of the package (their features need `WHISP_CARRIER_FULL=1`; see
section 6). They are listed anyway, because some of their code does ship.

`pyannote.audio` is excluded and confirmed absent from the archive.

---

## 5. Copyleft components

Everything above is permissively licensed except the following. None is
statically linked into whisp-carrier's own code.

| Component | Licence | How it is used | Obligation |
|---|---|---|---|
| ffmpeg | LGPL-3.0-or-later | Separate process | Ship the licence text (done: `LICENSE.ffmpeg.txt`) and point at the sources (done: section 1). Users may replace the binary. |
| libsndfile | LGPL-2.1-or-later | Dynamic library loaded by `soundfile` | Licence text and source availability: https://github.com/libsndfile/libsndfile |
| certifi | MPL-2.0 | CA bundle, data only | Source availability for the file itself; no effect on surrounding code |
| tqdm | MPL-2.0 AND MIT | Imported library | As above |

None of these makes the distribution GPL. That would have happened with a GPL
build of ffmpeg, which is why the build now refuses one.

`python-soxr` (LGPL-2.1-or-later) was present in earlier builds through
`librosa`. Excluding the conversion path removed `librosa`, and with it soxr;
it is no longer shipped.

---

## 6. What the exe deliberately cannot do

Not defects. Each is a packaging decision, and each reports itself clearly at
runtime rather than failing obscurely.

**Model conversion.** `transformers` is excluded, so the exe cannot convert a
transformers-format model to CTranslate2. Asking for `-m anime-whisper` exits 2
with instructions to convert once using the script version and then pass the
converted directory. Built-in sizes and CTranslate2 models are unaffected. The
division of labour is intentional: the exe is the normal path, and the script
version is what you use for anime-whisper — which measured 17pt worse than
large-v3 on this project's own test set, so little is lost.

Note that `ctranslate2.converters` must *not* be excluded even though it is
part of the same feature: `ctranslate2/__init__.py` imports it unconditionally,
so excluding it breaks `import ctranslate2` and the exe dies on startup. Only
`transformers` is excluded, which is safe because
`converters/transformers.py` guards its own imports with `try/except
ImportError`.

**Vocal extraction.** `--ff_vocal_extract` needs `audio-separator`, which is
excluded from every build including `WHISP_CARRIER_FULL=1`. It was bundled once:
the package itself packaged correctly, then failed at runtime inside scipy with
"The `scipy` install you are using seems to be broken ... please try
reinstalling" — advice that sends the user to repair an installation that is not
broken. `audio_filter._load_separator` now reports the actual reason and points
at the script version. Nothing measurable is lost, because every accuracy figure
this project reports was produced without any `--ff_*` filter.

Excluding it also removes what it pulled in: `librosa`, `onnx`, `opencv-python`,
`pandas`, `scikit-learn` and `python-soxr` — the last of which was the only
LGPL-licensed Python package in the distribution.

**Realignment.** `--realign` needs `stable-ts`, which the default build omits.
`WHISP_CARRIER_FULL=1` includes it and `--realign` was verified working in that
variant.

**External VAD backends.** `--vad_method pyannote_v3` / `pyannote_onnx_v3` are
unavailable because `pyannote.audio` is excluded on purpose (it drags in
pytorch-lightning and speechbrain, and measured worse than the built-in silero).
`vad._missing_backend` says so explicitly.

---

## Regenerating this list

Two sources have to be combined, because neither is complete on its own:

1. `dist/whisp-carrier/_internal/*.dist-info` plus the top-level directories
   there. Covers anything with a compiled extension or data files.
2. `build/whisp_carrier/PYZ-00.toc`, which lists the pure-Python modules that
   went into the archive. Without this, tqdm, rich, certifi and similar are
   missed entirely — they have no directory under `_internal`.

Match those names against the installed environment's metadata
(`importlib.metadata`) to recover versions and licences, since PyInstaller only
copies `dist-info` for packages that need it at runtime (7 of them here).

Also re-check `torch/lib` and `ctranslate2` for native libraries, `_internal`
for `ffmpeg.exe`, and `_tools/ffmpeg/PROVENANCE.txt` for the bundled ffmpeg's
provenance. Grep the TOCs directly to confirm an exclusion actually took
effect; a spec `excludes` entry that never matched fails silently.
