# -*- mode: python ; coding: utf-8 -*-
# whisp-carrier PyInstaller spec file

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

# ---------------------------------------------------------------------------
# TEN VAD (Apache-2.0), the default VAD since it beat silero on all nine
# references. Its wheel ships a prebuilt DLL that ten_vad/__init__.py loads with
#
#     CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)),
#                       "lib/Windows/x64/ten_vad.dll"))
#
# so the DLL has to keep its position *relative to the package*. That rules out
# collect_dynamic_libs(), which flattens libraries into _internal and would leave
# the ctypes lookup failing at runtime with a FileNotFoundError naming a path
# that does not exist. Frozen onedir sets the package __file__ under
# sys._MEIPASS, which is _internal, so shipping it as data at
# ten_vad/lib/Windows/x64 lands exactly where the lookup goes.
#
# Only the Windows x64 library is bundled. The wheel also carries a Linux .so
# and a macOS framework, and collect_data_files would take all three.
# ---------------------------------------------------------------------------
import ten_vad as _ten_vad

_TEN_VAD_SUBDIR = 'lib/Windows/x64'
_ten_vad_dll = Path(_ten_vad.__file__).parent / 'lib' / 'Windows' / 'x64' / 'ten_vad.dll'
if not _ten_vad_dll.is_file():
    raise SystemExit(
        f'[spec] ten_vad.dll not found at {_ten_vad_dll}.\n'
        'The default VAD needs it. Install the wheel with\n'
        '  pip install ten-vad==1.0.6.8\n'
        'or, if the layout changed, update _TEN_VAD_SUBDIR to match.'
    )
datas += [(str(_ten_vad_dll), f'ten_vad/{_TEN_VAD_SUBDIR}')]
print(f'[spec] ten_vad dll: {_ten_vad_dll} -> ten_vad/{_TEN_VAD_SUBDIR}')

# Apache-2.0 section 4 asks for the licence and any NOTICE file to travel with
# the work. The wheel carries both in its dist-info and they are copied next to
# the exe after COLLECT, the same way ffmpeg's licence is.
#
# rglob, not glob: this wheel puts them under dist-info/licenses/ rather than at
# the top of dist-info, which is where recent packaging tools place them. A
# non-recursive lookup found nothing and only printed a warning, so the first
# build shipped the DLL without its licence.
TEN_VAD_LICENCES = []
for _dist in Path(_ten_vad.__file__).parent.parent.glob('ten_vad-*.dist-info'):
    for _name in ('LICENSE', 'NOTICES'):
        for _candidate in _dist.rglob(_name):
            if _candidate.is_file():
                TEN_VAD_LICENCES.append(
                    (_candidate, f'LICENSE.ten-vad.{_name.lower()}.txt')
                )
                break
if not TEN_VAD_LICENCES:
    raise SystemExit(
        '[spec] no LICENSE/NOTICES found in the ten-vad dist-info.\n'
        'Apache-2.0 requires them to ship alongside the binary, and the DLL is '
        'bundled unconditionally, so this is a licensing defect rather than a '
        'warning. Locate them under site-packages/ten_vad-*.dist-info and '
        'update the lookup.'
    )
for _src, _name in TEN_VAD_LICENCES:
    print(f'[spec] ten-vad licence: {_src}')

# ---------------------------------------------------------------------------
# ffmpeg.exe
#
# This project is MIT. ffmpeg is not, so *which* ffmpeg build gets bundled is a
# licensing decision and not merely a packaging one.
#
#   LGPL build -> fine. ffmpeg is executed as a separate process, and LGPL only
#                 requires that the licence text and a pointer to the sources
#                 ship with it (see THIRD-PARTY-NOTICES.md).
#   GPL build  -> not fine. Bundling one places the entire distribution under
#                 the GPL, contradicting the MIT licence on our own code.
#
# This spec used to list the ffmpeg.exe inside a local Faster-Whisper-XXL
# download as its second candidate. That file is a gyan.dev "essentials" build
# of FFmpeg 6.1.1 configured with --enable-gpl --enable-version3, i.e. GPL v3.
# It sat ahead of PATH in the search order, so a successful build would have
# picked it up and quietly produced a GPL-encumbered distribution.
#
# The checks below make that class of mistake impossible to repeat silently:
# the configure banner of the actual binary is inspected, and a GPL or non-free
# build aborts the build outright.
# ---------------------------------------------------------------------------
import os
import shutil
import hashlib
import subprocess

# BtbN/FFmpeg-Builds, ffmpeg-n8.1-latest-win64-lgpl-8.1.zip -> bin/ffmpeg.exe
# (FFmpeg n8.1.2-44-g7c533d0f86-20260820, LGPL v3). Verified against the
# publisher's checksums.sha256; provenance in _tools/ffmpeg/PROVENANCE.txt.
FFMPEG_KNOWN_SHA256 = (
    '8ee152edc79f7ba99969b7fb590cfde438b46cb383952dec87e754db83788572'
)

ffmpeg_candidates = [
    os.environ.get('WHISP_CARRIER_FFMPEG'),
    Path(SPECPATH) / '_tools' / 'ffmpeg' / 'official' / 'ffmpeg.exe',
    shutil.which('ffmpeg'),
]
ffmpeg_src = next(
    (Path(c) for c in ffmpeg_candidates if c and Path(c).is_file()),
    None,
)
# Build-time messages are English for the same reason the runtime output is:
# the console code page mangles Japanese, and a redirected build log becomes
# unreadable exactly when it is needed. Verified by piping this spec's output
# to a file.
if ffmpeg_src is None:
    raise SystemExit(
        'ffmpeg.exe not found. Provide it in one of these ways:\n'
        '  1. place it at _tools/ffmpeg/official/ffmpeg.exe\n'
        '  2. put it on PATH\n'
        '  3. set WHISP_CARRIER_FFMPEG to its full path\n'
        'It must be an LGPL build. A GPL build cannot be bundled with this '
        'MIT-licensed distribution.\n'
        'Source: https://github.com/BtbN/FFmpeg-Builds/releases '
        '(files with -lgpl in the name)'
    )

_ffmpeg_digest = hashlib.sha256(ffmpeg_src.read_bytes()).hexdigest()
print(f'[spec] ffmpeg: {ffmpeg_src}')
print(f'[spec] ffmpeg sha256: {_ffmpeg_digest}')

# Read the configure line off the binary that is about to be bundled. The hash
# pin cannot be the only gate: it has to be updated whenever ffmpeg is, and a
# stale pin must not silently degrade into "no licence check at all".
try:
    _ffmpeg_banner = subprocess.run(
        [str(ffmpeg_src), '-hide_banner', '-version'],
        capture_output=True, text=True, timeout=120, check=True,
    ).stdout
except (OSError, subprocess.SubprocessError) as exc:
    raise SystemExit(f'[spec] failed to run "ffmpeg -version": {exc}')

_ffmpeg_config = next(
    (ln for ln in _ffmpeg_banner.splitlines()
     if ln.startswith('configuration:')),
    '',
)
if not _ffmpeg_config:
    raise SystemExit(
        '[spec] could not read the "configuration:" line from '
        '"ffmpeg -version". Refusing to bundle a binary whose licence '
        'cannot be determined.'
    )

_ffmpeg_forbidden = [
    flag for flag in ('--enable-gpl', '--enable-nonfree')
    if flag in _ffmpeg_config
]
if _ffmpeg_forbidden:
    raise SystemExit(
        f'[spec] this ffmpeg was built with '
        f'{" / ".join(_ffmpeg_forbidden)}:\n'
        f'  {ffmpeg_src}\n'
        'Bundling a GPL or non-free build would place this entire '
        'distribution under those terms, contradicting the project\'s MIT '
        'licence. Replace it with an LGPL build.\n'
        'Source: https://github.com/BtbN/FFmpeg-Builds/releases '
        '(files with -lgpl in the name)'
    )

_ffmpeg_version = (
    _ffmpeg_banner.splitlines()[0] if _ffmpeg_banner else '(unknown)'
)
if _ffmpeg_digest == FFMPEG_KNOWN_SHA256:
    print(f'[spec] ffmpeg verified (known LGPL build): {_ffmpeg_version}')
else:
    print('[spec] WARNING: ffmpeg hash does not match FFMPEG_KNOWN_SHA256.')
    print(f'[spec]   version: {_ffmpeg_version}')
    print('[spec]   The configure line carries no --enable-gpl / '
          '--enable-nonfree, so the build continues,')
    print('[spec]   but confirm the origin and update FFMPEG_KNOWN_SHA256 '
          'and _tools/ffmpeg/PROVENANCE.txt.')

datas += [(str(ffmpeg_src), '.')]

# LGPL asks for the licence text to travel with the binary. Kept next to the
# exe rather than buried in _internal; see the post-COLLECT copy below.
FFMPEG_LICENSE_SRC = ffmpeg_src.parent / 'LICENSE.txt'
if not FFMPEG_LICENSE_SRC.is_file():
    print('[spec] WARNING: no LICENSE.txt next to ffmpeg '
          f'({FFMPEG_LICENSE_SRC}). LGPL redistribution requires the licence '
          'text to ship with the binary.')

# ---------------------------------------------------------------------------
# Preserve a live config file across rebuilds.
#
# COLLECT deletes dist/whisp-carrier before repopulating it, which takes
# whisp-carrier.yaml with it. That file is not a build artifact: it is the
# user's settings, and on this machine it is what points Amatsukaze at the
# external VAD (worth 2.8pt of CER). Losing it silently means the next run
# quietly falls back to the built-in VAD path, which is exactly the kind of
# regression that does not announce itself.
#
# Read before COLLECT runs, written back after. Only whisp-carrier.yaml is
# treated this way; the .example is a build artifact and gets overwritten.
# ---------------------------------------------------------------------------
LIVE_CONFIG_PATH = Path(DISTPATH) / 'whisp-carrier' / 'whisp-carrier.yaml'
LIVE_CONFIG_DATA = None
if LIVE_CONFIG_PATH.is_file():
    LIVE_CONFIG_DATA = LIVE_CONFIG_PATH.read_bytes()
    print(f'[spec] preserving existing config: {LIVE_CONFIG_PATH}')

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
    # The default VAD. vad.py imports it inside a function behind
    # try/except ImportError, so name it here rather than relying on the
    # analyser following that branch.
    'ten_vad',
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
    'yaml',
    # Local modules, imported by name rather than discovered through packaging.
    'audio_filter',
    'vad',
    'loop_filter',
    'subtitle_format',
    'whisp_config',
    'whisp_models',
    'whisp_vad_patch',
    'winsound',
]

# Optional feature package: stable-ts, for --realign. Opt-in via
# WHISP_CARRIER_FULL=1 because it is heavy and because keeping the base build
# minimal makes it possible to tell a packaging regression apart from it.
# Measured at +300MB when it was bundled together with audio-separator.
#
# audio-separator (--ff_vocal_extract) used to be bundled here too and is now
# excluded outright. Bundling it was worse than leaving it out: it packaged
# cleanly, then failed at runtime inside scipy with "The `scipy` install you
# are using seems to be broken ... please try reinstalling", which points the
# user at an installation that is not broken. audio_filter._load_separator now
# reports the real reason instead. Nothing measurable is lost: every accuracy
# figure this project reports was produced without any --ff_* filter, so
# excluding it cannot move CER.
#
# pyannote.audio stays in excludes below on purpose as well: it drags in
# pytorch-lightning and speechbrain, and this project's own testing found
# external VAD worse than the built-in silero.
if os.environ.get('WHISP_CARRIER_FULL') == '1':
    print('[spec] WHISP_CARRIER_FULL=1: bundling stable-ts (--realign)')
    hiddenimports += ['stable_whisper']
    datas += collect_data_files('stable_whisper')

# Model conversion is deliberately left out of the exe. transformers is only
# needed to turn a transformers-format model into CTranslate2, which is a
# one-off step, so the exe is pointed at a model that is already CTranslate2: a
# converted directory (--model path\to\ct2-model, produced by running the
# script version once), a CTranslate2 repo, or a built-in size.
#
# The division of labour, decided deliberately: the exe is the normal path, and
# the script version is what you use when you want anime-whisper.
#
# Note that this needs the excludes below to be enforced. Merely leaving these
# out of hiddenimports does not work: PyInstaller follows the function-level
# import inside whisp_models._load_transformers and bundles transformers
# anyway. A build made before those excludes were added did contain
# transformers 5.15.0 along with torchvision, opencv-python, librosa, numba,
# llvmlite, pandas, SQLAlchemy, sentencepiece, tiktoken and openai-whisper.
#
# That measured 381MB out of a 4.95GB _internal, so the "would add gigabytes"
# claim this comment used to make was wrong. The reason to exclude it is not
# size: it is that shipping the conversion path would add an untested frozen
# code path (the alignment_heads repair in whisp_models kills the process
# outright when it is wrong, with no traceback) for the sake of a model that
# measured 17pt worse than large-v3.

excludes = [
    'matplotlib', 'tkinter', 'PyQt5', 'PyQt6',
    'PySide2', 'PySide6', 'wx', 'IPython',
    'jupyter', 'notebook', 'pyannote.audio',
    'speechbrain', 'pytorch_lightning',
    # --ff_vocal_extract. Excluded in every build, including FULL; see the note
    # above the WHISP_CARRIER_FULL branch. Whatever it dragged in (librosa,
    # onnx, opencv, pandas, scikit-learn) goes with it, because PyInstaller
    # only keeps what something still imports.
    'audio_separator',
    # The conversion toolchain. whisp_models reports it as missing with
    # instructions to use the script version instead.
    #
    # Only transformers is excluded, never ctranslate2.converters: that one is
    # imported unconditionally by ctranslate2/__init__.py, so excluding it
    # breaks `import ctranslate2` and with it the whole exe (verified: the
    # build succeeded and every run died with "cannot import name 'converters'
    # from partially initialized module 'ctranslate2'"). Dropping transformers
    # alone is safe because converters/transformers.py guards its
    # huggingface_hub / torch / transformers imports with try/except ImportError.
    'transformers',
]

# Packages that only the conversion path drags in. They have to be named
# explicitly, because excluding transformers does not automatically drop what
# transformers pulled in; this list is what the pre-exclude build actually
# contained. The optional feature packages need several of them, though
# (audio-separator uses librosa and onnx, stable-ts uses openai-whisper, which
# is the 'whisper' module), so in a full build they have to stay.
_conversion_adjacent = [
    'torchvision', 'cv2', 'librosa', 'numba', 'llvmlite',
    'optuna', 'pandas', 'sqlalchemy', 'sentencepiece', 'tiktoken',
    'onnx', 'whisper',
]
if os.environ.get('WHISP_CARRIER_FULL') == '1':
    print('[spec] WHISP_CARRIER_FULL=1: keeping '
          f'{len(_conversion_adjacent)} conversion-adjacent packages that '
          'audio-separator and stable-ts depend on')
else:
    excludes += _conversion_adjacent

a = Analysis(
    ['whisp_carrier.py'],
    # SPECPATH is injected by PyInstaller and always points at this spec's
    # directory, so the build does not break when the project folder is renamed.
    pathex=[SPECPATH],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
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
    name='whisp-carrier',
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
    name='whisp-carrier',
)

# Ship the config sample next to the exe rather than inside _internal, because
# that is where whisp_config.discover() looks for whisp-carrier.yaml and where a
# user can reasonably be expected to edit it. Anything passed through datas ends
# up under _internal in onedir mode, so this has to be a post-COLLECT copy.
_sample = Path(SPECPATH) / 'whisp-carrier.yaml.example'
if _sample.is_file():
    _target = Path(DISTPATH) / 'whisp-carrier' / _sample.name
    _target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(_sample, _target)
    print(f'[spec] copied config sample to: {_target}')

# Licence paperwork, for the same reason and by the same route. LGPL requires
# ffmpeg's licence text to accompany the binary, and a reader looking for it
# will look beside the exe, not inside _internal.
_dist_root = Path(DISTPATH) / 'whisp-carrier'
_dist_root.mkdir(parents=True, exist_ok=True)

if FFMPEG_LICENSE_SRC.is_file():
    _target = _dist_root / 'LICENSE.ffmpeg.txt'
    shutil.copy2(FFMPEG_LICENSE_SRC, _target)
    print(f'[spec] copied ffmpeg licence to: {_target}')

# TEN VAD, Apache-2.0. Same reasoning as ffmpeg: beside the exe, not buried in
# _internal, because that is where a reader looks.
for _src, _name in TEN_VAD_LICENCES:
    _target = _dist_root / _name
    shutil.copy2(_src, _target)
    print(f'[spec] copied ten-vad {_src.name} to: {_target}')

# README.md travels with the archive too. Someone who downloads the release and
# never visits the repository would otherwise have no instructions at all, and
# README.md is the Japanese user-facing manual written against this exe build
# (README_en.md is a summary and stays in the repository).
for _name in ('LICENSE', 'THIRD-PARTY-NOTICES.md', 'README.md'):
    _src = Path(SPECPATH) / _name
    if _src.is_file():
        shutil.copy2(_src, _dist_root / _name)
        print(f'[spec] copied {_name} to: {_dist_root / _name}')
    else:
        print(f'[spec] WARNING: {_name} not found ({_src})')

# Put the user's settings back (see LIVE_CONFIG_PATH above).
if LIVE_CONFIG_DATA is not None:
    LIVE_CONFIG_PATH.write_bytes(LIVE_CONFIG_DATA)
    print(f'[spec] restored config: {LIVE_CONFIG_PATH}')
