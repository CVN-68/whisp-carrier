#!/usr/bin/env python3
"""
whisp-carrier.py
RTX 5090 (sm_120 / Blackwell) native compatible faster-whisper CLI.
Drop-in replacement for Faster-Whisper-XXL with torch 2.8+cu128.

Author: whisp-carrier contributors
License: MIT
"""

from __future__ import annotations

import argparse
import codecs
import contextlib
import ctypes
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Optional

# huggingface_hub warns, on the first download of any model, that this machine
# cannot make symlinks so its cache will use more disk. It is four lines of
# English in the middle of the Amatsukaze log at the worst possible moment (a
# first run), and there is nothing the reader can do about it: symlinks need
# Developer Mode or an elevated shell on Windows. Caching still works.
#
# Set before faster_whisper pulls huggingface_hub in, and with setdefault so
# that anyone who deliberately asked for the warning still gets it.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Names of the CUDA_* variables dropped by _use_bundled_cuda(), reported with
# the startup banner so a support log says which machine this happened on.
CUDA_ENV_DROPPED: List[str] = []

# The CUDA libraries CTranslate2 resolves by name at first use, in dependency
# order (cublas imports cublasLt). Pinned by absolute path rather than found by
# search: see preload_bundled_cuda().
BUNDLED_CUDA_PRELOAD = ("cublasLt64_12.dll", "cublas64_12.dll")

# Test hook for the two layers below. Not documented for users -- it exists so
# that each layer can be shown to work on its own, and so that the failure this
# release fixes can still be reproduced with the shipped exe.
#
#   unset       both layers (normal)
#   'preload'   pinning only, CUDA_PATH left in place
#   'off'       neither, i.e. the 0.9.1 behaviour
_CUDA_FIX = os.environ.get("WHISP_CARRIER_CUDA_FIX", "").strip().lower()


def _use_bundled_cuda() -> None:
    r"""Keep CTranslate2 on the CUDA libraries this build ships.

    ctranslate2.dll resolves cuBLAS on first use rather than at load time, and
    when CUDA_PATH is set it looks under %CUDA_PATH%\bin. On a machine whose
    toolkit is 12.x that silently works -- and runs the machine's cuBLAS, not
    the one we shipped and measured. On any other machine it is fatal:

        [ERROR] <file>: Library cublas64_12.dll is not found or cannot be loaded

    CUDA 13 has cublas64_13.dll, CUDA 11 has cublas64_11.dll, and an
    uninstalled toolkit can leave CUDA_PATH behind pointing at nothing useful.
    None of them carry cublas64_12.dll, so the run dies even though _internal/
    holds it. Reported from the field 2026-08-24 and reproduced here.

    Dropping the variable from our own process restores the normal search
    order, which finds _internal. Nothing is lost: the exe ships every CUDA
    library CTranslate2 loads, borrowing the machine's copies was never
    intended, and child processes (ffmpeg) do not read this.

    Frozen builds only. From source there is no bundled copy to prefer, and
    torch registers its own directory on import.
    """
    if os.name != "nt" or not getattr(sys, "frozen", False):
        return
    if _CUDA_FIX in ("off", "preload"):
        return

    for var in ("CUDA_PATH", "CUDA_HOME"):
        if os.environ.pop(var, None) is None:
            continue
        CUDA_ENV_DROPPED.append(var)
        # os.environ.pop updates the CRT copy of the environment.
        # ctranslate2.dll reads the Win32 block, so clear that too instead of
        # trusting the two to stay in sync.
        try:
            ctypes.windll.kernel32.SetEnvironmentVariableW(var, None)
        except Exception:
            pass

    # Belt and braces: register _internal as a DLL directory in its own right,
    # so a SetDllDirectory call by any dependency cannot hide it. The
    # bootloader already makes it findable; this survives it being replaced.
    base = getattr(sys, "_MEIPASS", None)
    if base:
        try:
            os.add_dll_directory(base)
        except OSError:
            pass


_use_bundled_cuda()


def preload_bundled_cuda() -> List[str]:
    r"""Pin the bundled cuBLAS by absolute path, before CTranslate2 looks for it.

    Second layer under _use_bundled_cuda(), and the one that does not depend on
    the DLL search path being intact. CTranslate2 asks for `cublas64_12.dll` by
    name; a module with that base name already in the process satisfies the
    request immediately (measured: 0.0 ms), whichever directories are being
    searched by then. Loading them here by full path is therefore the difference
    between "we ship it" and "it is the one that runs".

    Order matters: cublas imports cublasLt, so cublasLt goes first and the
    dependency is met by name rather than by another search.

    Called only for device=cuda. Mapping the two files costs about a second, and
    on the GPU path that is not an extra second -- CTranslate2 maps the same
    files a moment later anyway. On CPU it would be pure waste.

    Absolute paths do not survive this trick: LoadLibrary with a full path to a
    file that does not exist fails even when a module of the same base name is
    loaded (measured). So this covers a name-based lookup, which is what
    CTranslate2 does -- confirmed in the field, where putting _internal on PATH
    was enough to make 0.9.1 work.
    """
    if os.name != "nt" or not getattr(sys, "frozen", False):
        return []
    if _CUDA_FIX == "off":
        return []

    base = getattr(sys, "_MEIPASS", None)
    if not base:
        return []

    loaded: List[str] = []
    for name in BUNDLED_CUDA_PRELOAD:
        path = Path(base) / name
        if not path.is_file():
            continue
        try:
            ctypes.WinDLL(str(path))
        except OSError:
            # Not fatal: the normal search may still find it, and a GPU failure
            # downstream reports itself with more context than we could here.
            continue
        loaded.append(name)
    return loaded


from faster_whisper import WhisperModel  # noqa: E402  (must follow the env var)
from tqdm import tqdm

import loop_filter
import subtitle_format
import whisp_config
import whisp_models
import whisp_vad_patch

def _setup_console_encoding() -> None:
    """Write UTF-8 on stdout/stderr, because that is what our callers read.

    On Windows, Python only uses UTF-8 for the console itself (3.6+ writes
    through the wide console API). As soon as the streams are a pipe or a file
    they fall back to the locale encoding, which is cp932 on a Japanese
    install. Measured here: isatty() -> 'utf-8', piped -> 'cp932'.

    That fallback is what broke the Amatsukaze logs. Amatsukaze decodes this
    process's output as UTF-8 unconditionally -- TranscodeManager.cpp sets
    `param.isUtf8Log = true` on both the synchronous and the parallel path, and
    ProcessThread.cpp then runs every line through
    `utf8ToString()` (CP_UTF8 -> UTF-16 -> CP_ACP). Feeding cp932 bytes into a
    UTF-8 decoder produced the mojibake we saw: 'ロック' went out as
    83 8D 83 62 83 4E and came back as '???b?N', the two trail bytes that
    happen to be ASCII ('b' = 0x62, 'N' = 0x4E) surviving.

    So the encoding was ours to fix all along. Setting it here also repairs
    every other line that quotes the source -- input paths, [OUT] paths, the
    live transcription preview -- none of which went through console_safe().

    PYTHONIOENCODING wins if the caller set it: eval/run.py relies on that, and
    someone who asked for a specific encoding should get it.
    """
    if os.environ.get("PYTHONIOENCODING"):
        return
    for stream in (sys.stdout, sys.stderr):
        # Frozen/windowed builds can hand us None, and reconfigure() needs 3.7.
        if stream is None or not hasattr(stream, "reconfigure"):
            continue
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            # Nothing here is worth failing a transcription over.
            pass


_setup_console_encoding()


def console_safe(text: str) -> str:
    r"""Escape text that the output stream cannot represent.

    Our own output is English on purpose, but some of it has to quote the
    source: the [LOOP] lines show the text that was discarded, and errors name
    the file that failed. Those are Japanese here.

    With _setup_console_encoding() in front of this, the streams are UTF-8 and
    the check below passes, so the Japanese goes out as-is and is readable in
    the Amatsukaze log. This used to escape unconditionally, which was the
    wrong fix for the right observation: the text really did arrive as '?', but
    because we were emitting cp932 into a UTF-8 reader, not because the reader
    was beyond our reach.

    The guard stays because it still earns its place, but the test is "are we
    writing UTF-8" rather than "can this encoding hold the character". Those
    differ, and the difference is the whole bug: cp932 represents 'ロック'
    perfectly well, so an encodability check passes and then the UTF-8 reader on
    the other end still garbles it. UTF-8 is the only encoding we know our
    consumer decodes, so anything else means the bytes are not ours to trust and
    \uXXXX is the safe form -- ugly, but reversible and never lossy.

    Reachable when PYTHONIOENCODING pins something narrower, which _setup_
    console_encoding() deliberately leaves alone. eval/ takes the same position
    for the same reason: raw glyphs in a cp932 pipe killed investigations with
    UnicodeEncodeError midway.
    """
    if text.isascii():
        return text
    for stream in (sys.stdout, sys.stderr):
        encoding = getattr(stream, "encoding", None)
        if not encoding:
            continue
        try:
            normalised = codecs.lookup(encoding).name
        except LookupError:
            return text.encode("unicode_escape").decode("ascii")
        if normalised != "utf-8":
            return text.encode("unicode_escape").decode("ascii")
    return text


# ─────────────────────────────────────────────
# Supported media extensions for batch mode
# ─────────────────────────────────────────────
MEDIA_EXTENSIONS = {
    ".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm",
    ".mp3", ".wav", ".flac", ".aac", ".m4a", ".ogg", ".wma",
    ".ts", ".m2ts", ".mts",
}

VERSION = "0.9.2"

# ─────────────────────────────────────────────
# Per-backend VAD threshold
#
# The speech probability a VAD emits is on a model-specific scale, so one number
# does not carry across. Measured on 死亡遊戯 #09, which is the reference that
# loses the most speech: TEN at silero's 0.45 recovers the missing speech
# (coverage 59.7% -> 73.1%) but drops precision to 74.6%, and 0.75 keeps the
# recall while holding precision at 83.2% (whole-region CER 39.3% -> 30.5%).
# 0.90 was measured on four references and lost on the total, so 0.75 it is.
#
# --vad_threshold defaults to None and is resolved here, rather than carrying a
# single default that would be wrong for whichever backend is not selected.
# ─────────────────────────────────────────────
VAD_THRESHOLD_DEFAULTS = {"ten": 0.75, "silero": 0.45}


def resolve_vad_threshold(method: str) -> float:
    """The default --vad_threshold for this backend.

    'precomputed' takes regions from a file, so the threshold only reaches the
    shared aggregation and any value behaves the same; it gets the silero number
    so that comparisons against silero stay like-for-like.
    """
    family = "ten" if method == "ten" else "silero"
    return VAD_THRESHOLD_DEFAULTS[family]
# Device reporting, and the source of the `--device auto` decision.
#
# torch is asked first because the script version has it and it answers all
# three questions, including the GPU name. It is not required though: inference
# runs on CTranslate2 and the default VAD is a native library reached with
# ctypes, so nothing on the normal path imports torch. The exe therefore does
# not bundle it -- torch and the CUDA kernels it carries were 4.3 GB of a 4.7 GB
# payload, spent on this banner and on silero, which lost to TEN VAD on all
# fifteen references.
#
# ctranslate2 answers the one question that changes behaviour ("is there a usable
# CUDA device"), so falling back to it keeps `--device auto` correct without
# torch. The GPU name is cosmetic and is reported as unknown instead.
TORCH_VERSION = ""
CUDA_AVAILABLE = False
CUDA_DEVICE_NAME = "N/A"
try:
    import torch
    TORCH_VERSION = torch.__version__
    CUDA_AVAILABLE = torch.cuda.is_available()
    CUDA_DEVICE_NAME = torch.cuda.get_device_name(0) if CUDA_AVAILABLE else "N/A"
except Exception:
    try:
        import ctranslate2
        CUDA_AVAILABLE = ctranslate2.get_cuda_device_count() > 0
        if CUDA_AVAILABLE:
            CUDA_DEVICE_NAME = "(name unavailable without torch)"
    except Exception:
        pass


def runtime_banner() -> str:
    """One line naming the inference runtime actually in use.

    Printed instead of the torch version, which was misleading even when torch
    was bundled: it is not what runs the model.
    """
    try:
        import ctranslate2
        return f"ctranslate2 {ctranslate2.__version__}"
    except Exception:
        return "ctranslate2 (version unknown)"


# ─────────────────────────────────────────────
# SRT / VTT / TXT writers
# ─────────────────────────────────────────────

def format_timestamp(seconds: float, vtt: bool = False) -> str:
    ms = int(round(seconds * 1000))
    h = ms // 3_600_000
    ms %= 3_600_000
    m = ms // 60_000
    ms %= 60_000
    s = ms // 1000
    ms %= 1000
    sep = "." if vtt else ","
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def one_line(text: str) -> str:
    """Collapse a possibly multi-line cue into a single line.

    Subtitle formatting can put newlines inside a cue. SRT and VTT render those
    natively, but TXT, TSV and LRC are line-oriented and would break.
    """
    return " ".join(str(text).split())


def write_srt(segments, output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            f.write(f"{i}\n")
            f.write(f"{format_timestamp(seg['start'])} --> {format_timestamp(seg['end'])}\n")
            f.write(f"{seg['text'].strip()}\n\n")


def write_vtt(segments, output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for i, seg in enumerate(segments, 1):
            f.write(f"{i}\n")
            f.write(f"{format_timestamp(seg['start'], vtt=True)} --> {format_timestamp(seg['end'], vtt=True)}\n")
            f.write(f"{seg['text'].strip()}\n\n")


def write_txt(segments, output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        for seg in segments:
            f.write(one_line(seg["text"]) + "\n")


def write_tsv(segments, output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("start\tend\ttext\n")
        for seg in segments:
            # Tabs and newlines would corrupt the column layout.
            text = one_line(seg["text"]).replace("\t", " ")
            f.write(f"{seg['start']:.3f}\t{seg['end']:.3f}\t{text}\n")


def write_lrc(segments, output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        for seg in segments:
            s = seg["start"]
            m = int(s // 60)
            sec = s - m * 60
            f.write(f"[{m:02d}:{sec:05.2f}]{one_line(seg['text'])}\n")


def write_json(segments, output_path: str, info: dict) -> None:
    data = {
        "language": info.get("language", ""),
        "duration": info.get("duration", 0),
        "segments": segments,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# Format name -> file extension. Single source of truth so that the paths we
# report can never drift from the paths we actually write. Note that "text" is
# an alias of "txt" and therefore shares its extension.
FORMAT_EXTENSIONS = {
    "srt":  ".srt",
    "vtt":  ".vtt",
    "txt":  ".txt",
    "text": ".txt",
    "tsv":  ".tsv",
    "lrc":  ".lrc",
    "json": ".json",
}

# What "all" expands to. Excludes the "text" alias to avoid writing .txt twice.
ALL_FORMATS = ("srt", "vtt", "txt", "tsv", "lrc", "json")


def expand_formats(formats: List[str]) -> List[str]:
    """Resolve the requested formats into concrete writer names, deduplicated."""
    resolved: List[str] = []
    for fmt in formats or []:
        names = ALL_FORMATS if fmt == "all" else (fmt,)
        for name in names:
            if name in FORMAT_EXTENSIONS and name not in resolved:
                resolved.append(name)
    # Drop the alias when the canonical name is already present.
    if "txt" in resolved and "text" in resolved:
        resolved.remove("text")
    return resolved


def output_paths(base_path: str, formats: List[str]) -> List[str]:
    """Paths that write_outputs() would produce, in write order."""
    return [base_path + FORMAT_EXTENSIONS[name] for name in expand_formats(formats)]


def write_outputs(segments, info: dict, base_path: str, formats: List[str]) -> List[str]:
    """Write every requested format. Returns the paths actually written."""
    writers = {
        "srt":  write_srt,
        "vtt":  write_vtt,
        "txt":  write_txt,
        "text": write_txt,
        "tsv":  write_tsv,
        "lrc":  write_lrc,
        "json": lambda s, p: write_json(s, p, info),
    }
    written: List[str] = []
    for name in expand_formats(formats):
        path = base_path + FORMAT_EXTENSIONS[name]
        writers[name](segments, path)
        written.append(path)
    return written


# ─────────────────────────────────────────────
# Core transcription
# ─────────────────────────────────────────────

def transcribe_file(
    audio_path: str,
    engine,
    args: argparse.Namespace,
) -> tuple[List[dict], dict]:
    """Run transcription on a pre-processed audio file.

    'engine' is either a WhisperModel or a BatchedInferencePipeline wrapping one.
    In faster-whisper 1.2.1 the batched pipeline accepts every argument the plain
    model does plus batch_size, so the same kwargs work for both.
    """

    transcribe_kwargs = dict(
        language=args.language,
        task=args.task,
        temperature=args.temperature,
        best_of=args.best_of,
        beam_size=args.beam_size,
        patience=args.patience,
        length_penalty=args.length_penalty,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        # "None"/"auto" are accepted as explicit no-prompt values so that
        # existing Amatsukaze option strings keep working.
        initial_prompt=None if args.initial_prompt in (None, "None", "auto") else args.initial_prompt,
        condition_on_previous_text=args.condition_on_previous_text,
        compression_ratio_threshold=args.compression_ratio_threshold,
        log_prob_threshold=args.logprob_threshold,
        no_speech_threshold=args.no_speech_threshold,
        word_timestamps=args.word_timestamps,
        hallucination_silence_threshold=getattr(args, "hallucination_silence_threshold", 0) if getattr(args, "hallucination_silence_threshold", 0) > 0 else None,
        # Accepted by both WhisperModel.transcribe and BatchedInferencePipeline.
        max_new_tokens=args.max_new_tokens,
        chunk_length=args.chunk_length,
        hotwords=args.hotwords,
        vad_filter=args.vad_filter,
        vad_parameters=dict(
            threshold=args.vad_threshold,
            min_speech_duration_ms=args.vad_min_speech_duration_ms,
            max_speech_duration_s=args.vad_max_speech_duration_s or float("inf"),
            min_silence_duration_ms=args.vad_min_silence_duration_ms,
            speech_pad_ms=args.vad_speech_pad_ms,
            # VadOptions carries this field but nothing was setting it, so the
            # end-of-speech test was always the derived max(threshold-0.15,0.01).
            # Only sent when asked for, so the derived value stays the default.
            **({} if args.vad_neg_threshold is None
               else {"neg_threshold": args.vad_neg_threshold}),
        ) if args.vad_filter else None,
    )

    # Use custom VAD if method is not the built-in silero.
    # Note: faster-whisper 1.2+ uses silero_vad_v6.onnx internally, so the
    # *_fw names all resolve to the same model unless --vad_onnx replaces it.
    builtin_vad_methods = {"silero_v4_fw", "silero_v5_fw", "silero_v6", "silero_v6_fw"}
    vad_ctx = contextlib.nullcontext()
    if args.vad_filter and args.vad_method not in builtin_vad_methods:
        from vad import get_speech_segments
        print(f"  [VAD] Running {args.vad_method}...", flush=True)
        # Timed because the cost of the VAD itself is invisible otherwise: it
        # runs inside this function, so it is folded into the [STT] Done total.
        # TEN steps a native call every 256 samples from Python, which is fine
        # on a 24 minute file and unknown on a 5 hour one, and no threshold or
        # routing choice can change it. See HANDOVER 次の着手 I.
        vad_started = time.time()
        speech_segs = get_speech_segments(
            audio_path,
            method=args.vad_method,
            threshold=args.vad_threshold,
            min_speech_ms=args.vad_min_speech_duration_ms,
            max_speech_s=args.vad_max_speech_duration_s,
            min_silence_ms=args.vad_min_silence_duration_ms,
            speech_pad_ms=args.vad_speech_pad_ms,
            window_size=args.vad_window_size_samples,
            vad_device=args.vad_device,
            neg_threshold=args.vad_neg_threshold,
            segments_json=args.vad_segments_json,
        )
        vad_elapsed = time.time() - vad_started
        if speech_segs:
            # Speech seconds were only reported on the collect path, by
            # describe_external() after the fact, so the clip path gave no way
            # to read a speech ratio out of a production log.
            speech_seconds = sum(end - start for start, end in speech_segs)
            print(
                f"  [VAD] {args.vad_method}: {len(speech_segs)} regions | "
                f"{speech_seconds:.1f}s of speech | detected in {vad_elapsed:.1f}s",
                flush=True,
            )
        if not speech_segs:
            # Previously this produced an empty clip_timestamps string, which
            # faster-whisper reads as "no restriction" and quietly transcribes
            # the whole file. Say what happened instead.
            print(
                f"  [VAD] {args.vad_method} found no speech; falling back to the "
                "built-in VAD for this file",
                flush=True,
            )
        elif args.vad_segment_mode == "clip":
            clip_ts = ",".join(f"{s},{e}" for s, e in speech_segs)
            transcribe_kwargs["clip_timestamps"] = clip_ts
            transcribe_kwargs["vad_filter"] = False
            print(
                f"  [VAD] {len(speech_segs)} segments via clip_timestamps "
                "(clips are zero-padded to 30s)",
                flush=True,
            )
        else:
            # Route the external segments through the built-in code path so the
            # silence is cut out of the waveform instead of zero-padded. This
            # was the default until the 5h22m measurement, where it lost on
            # every exactly-computed metric; see HANDOVER 測定結果 #6.
            vad_ctx = whisp_vad_patch.external_segments(speech_segs)

    # batch_size only exists on BatchedInferencePipeline.transcribe. Everything
    # else in transcribe_kwargs is accepted by both engines.
    if args.batched:
        transcribe_kwargs["batch_size"] = args.batch_size

    # The VAD hook fires inside transcribe() before the segment generator is
    # returned, so the patch only has to be live for this call.
    with vad_ctx as vad_info:
        segments_gen, info = engine.transcribe(audio_path, **transcribe_kwargs)

    for line in whisp_vad_patch.describe_external(args.vad_method, vad_info):
        print(f"  {line}", flush=True)

    segments = []
    last_text = ""
    dupe_count = 0
    MAX_DUPES = 2  # 同じテキストが連続2回以上出たらスキップ
    seg_count = 0

    def is_hallucination(text: str, prev: str) -> bool:
        """テキストがハルシネーション（繰り返し）かどうか判定"""
        if not text or not prev:
            return False
        # 完全一致
        if text == prev:
            return True
        # 同じ文字の繰り返しが大部分を占める（例：ぬぬぬぬ...）
        stripped = text.replace(" ", "").replace("　", "")
        if len(stripped) > 10 and len(set(stripped)) <= 2:
            return True
        return False

    for seg in segments_gen:
        text = seg.text.strip()

        # ハルシネーションループ検出
        if is_hallucination(text, last_text):
            dupe_count += 1
            if dupe_count >= MAX_DUPES:
                continue  # ループしている重複をスキップ
        else:
            dupe_count = 0
            last_text = text

        seg_count += 1
        entry = {
            "id": seg.id,
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "text": seg.text,
        }
        if args.word_timestamps and seg.words:
            entry["words"] = [
                {"word": w.word, "start": round(w.start, 3), "end": round(w.end, 3), "probability": round(w.probability, 4)}
                for w in seg.words
            ]
        segments.append(entry)

        # 常に進捗を出力（-pp なしでも）
        elapsed_ts = time.strftime("%H:%M:%S", time.gmtime(seg.end))
        if args.print_progress:
            print(f"\r  [{elapsed_ts}] ({seg_count} segs) {text[:60]}", end="", flush=True)
        elif seg_count % 10 == 0:
            print(f"  [STT] {elapsed_ts} | {seg_count} segments processed...", flush=True)

    if args.print_progress:
        print()

    info_dict = {
        "language": info.language,
        "language_probability": round(info.language_probability, 4),
        "duration": round(info.duration, 3),
    }
    return segments, info_dict


def process_single_file(
    input_path: str,
    engine,
    args: argparse.Namespace,
    device: str = "cuda",
    compute_type: str = "float16",
    model_dir: str = None,
    model_path: str = None,
) -> None:
    input_path = str(Path(input_path).resolve())
    print(f"\n[whisp-carrier] Processing: {input_path}", flush=True)

    # Determine output base path
    if args.output_dir == "source":
        out_dir = str(Path(input_path).parent)
    elif args.output_dir == "default":
        out_dir = str(Path(input_path).parent) if args.batch_recursive else str(Path(sys.argv[0]).parent)
    else:
        out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    stem = Path(input_path).stem
    # With --postfix the detected language is appended, so the final base path
    # can only be built after transcription.
    base_out = os.path.join(out_dir, stem)

    # Check if output already exists (--skip). Uses the real extension of the
    # first requested format rather than the format name.
    if args.skip:
        candidates = output_paths(base_out, args.output_format)
        primary = candidates[0] if candidates else base_out + ".srt"
        if os.path.exists(primary):
            print(f"  Skipping (output exists): {primary}", flush=True)
            return

    # Preprocessing
    tmp_audio = None
    try:
        audio_input = input_path
        has_ff = any([
            getattr(args, "ff_rnndn_sh", False),
            getattr(args, "ff_rnndn_xiph", False),
            getattr(args, "ff_fftdn", 0) > 0,
            getattr(args, "ff_gate", False),
            getattr(args, "ff_speechnorm", False),
            getattr(args, "ff_loudnorm", False),
            getattr(args, "ff_lowhighpass", False),
            getattr(args, "ff_fc", False),
            getattr(args, "ff_lc", False),
            getattr(args, "ff_invert", False),
            getattr(args, "ff_vocal_extract", None) is not None,
            getattr(args, "ff_tempo", 1.0) != 1.0,
            (getattr(args, "ff_silence_suppress", [0])[0] or 0) != 0,
        ])

        if has_ff:
            print("  [FF] Running audio filters...", flush=True)
            from audio_filter import preprocess
            tmp_audio = preprocess(input_path, args)
            audio_input = tmp_audio

        # Transcription
        print(f"  [STT] Transcribing with model={args.model}...", flush=True)
        t0 = time.time()
        segments, info = transcribe_file(audio_input, engine, args)
        elapsed = time.time() - t0
        # Audio duration is echoed so the speech seconds on the [VAD] line above
        # can be read as a ratio without opening the JSON.
        print(f"  [STT] Done in {elapsed:.1f}s | dur={info['duration']:.1f}s | lang={info['language']} ({info['language_probability']:.2%}) | {len(segments)} segments", flush=True)

        # Drop segments that are one thing repeated. This runs before the
        # duration repair below because a loop never needs splitting, and
        # because the repair would otherwise turn one broken 30s cue into
        # several. Measured on nine recordings, dropping these took the total
        # CER from 24.3% to 22.0% at no cost in coverage (HANDOVER 測定結果 #11).
        if getattr(args, "loop_filter", True):
            before_loop = len(segments)
            segments, loop_stats = loop_filter.filter_segments(segments)
            loop_line = loop_filter.describe(
                loop_stats, before_loop, len(segments)
            )
            if loop_line:
                print(loop_line, flush=True)
                for start, end, text, why in loop_stats["examples"]:
                    stamp = time.strftime("%H:%M:%S", time.gmtime(start))
                    # `why` quotes the repeated unit, so it carries source text
                    # too. repr() does not escape non-ASCII (PEP 3138), so this
                    # needs the same treatment as the text -- it was the one
                    # place that missed it, which is why the reason field showed
                    # '???b?N' while the text beside it came out fine.
                    print(
                        f"         {stamp} {end - start:5.1f}s "
                        f"{console_safe(why)}: "
                        f"{console_safe(text[:40])}",
                        flush=True,
                    )

        # Repair segments that straddle a pause the VAD cut out of the waveform.
        # This runs whatever the formatting options are, because the damage is a
        # broken timestamp rather than a layout preference: measured with
        # large-v3, the default VAD path produced cues up to 370s long.
        before_fix = len(segments)
        segments, fix_stats = subtitle_format.sanitize_segments(
            segments, max_gap=getattr(args, "max_gap", 3.0)
        )
        fix_line = subtitle_format.describe_sanitize(
            fix_stats, before_fix, len(segments)
        )
        if fix_line:
            print(fix_line, flush=True)

        # Subtitle line formatting (--sentence / --max_line_width / --max_line_count)
        if subtitle_format.is_enabled(args):
            before = len(segments)
            segments = subtitle_format.format_segments(
                segments, args, language=info.get("language")
            )
            print(
                f"  [FMT] width={args.max_line_width} lines={args.max_line_count} "
                f"sentence={bool(args.sentence)} gap={args.max_gap} | "
                f"{before} -> {len(segments)} cues",
                flush=True,
            )

        # Postfix language to filename
        if args.postfix:
            base_out = os.path.join(out_dir, f"{stem}.{info['language']}")

        # Write outputs
        written = write_outputs(segments, info, base_out, args.output_format)

        # --realign: タイムスタンプ再調整
        if getattr(args, "realign", False):
            srt_path = base_out + ".srt"
            if subtitle_format.is_enabled(args):
                # stable-ts rewrites the whole SRT, which would throw away the
                # line layout produced just above.
                print("  [REALIGN] Skipped: incompatible with the subtitle formatting "
                      "options, which it would overwrite. Drop either --realign or "
                      "the formatting options.", flush=True)
            elif not os.path.exists(srt_path):
                print("  [REALIGN] Skipped: no SRT was written "
                      "(add srt to --output_format).", flush=True)
            else:
                print("  [REALIGN] Realigning timestamps...", flush=True)
                try:
                    import stable_whisper
                    realign_device = getattr(args, "realign_device", None) or device
                    # stable-ts でSRTを読み込んでタイムスタンプを調整
                    # The resolved path matters here: an alias or a converted
                    # model is not a name stable-ts could load by itself.
                    result = stable_whisper.load_faster_whisper(
                        model_path or args.model,
                        device=realign_device,
                        compute_type=compute_type,
                        download_root=model_dir,
                    )
                    # alignではなくtranscribe_stableで直接取得する方が安定
                    stable_result = result.transcribe(
                        audio_input,
                        language=info["language"],
                        regroup=True,
                    )
                    stable_result.to_srt_vtt(srt_path, word_level=False)
                    print(f"  [REALIGN] Done: {srt_path}", flush=True)
                except Exception as e:
                    print(f"  [REALIGN] Failed (skipped): {e}", flush=True)
        for w in written:
            print(f"  [OUT] {w}", flush=True)

    finally:
        if tmp_audio and os.path.exists(tmp_audio):
            try:
                os.unlink(tmp_audio)
            except Exception:
                pass


def collect_files(paths: List[str], recursive: bool) -> List[str]:
    """Collect all media files from file/dir/wildcard inputs."""
    import glob
    result = []
    for p in paths:
        expanded = glob.glob(p)
        if not expanded:
            expanded = [p]
        for item in expanded:
            item_path = Path(item)
            if item_path.is_file():
                if item_path.suffix.lower() in MEDIA_EXTENSIONS:
                    result.append(str(item_path))
            elif item_path.is_dir() and recursive:
                for ext in MEDIA_EXTENSIONS:
                    result.extend(str(f) for f in item_path.rglob(f"*{ext}"))
            elif item_path.suffix.lower() in {".txt", ".m3u", ".m3u8", ".lst"}:
                with open(item_path, encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            result.append(line)
    return result


# ─────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="whisp-carrier",
        description="whisp-carrier - RTX 5090 native faster-whisper CLI (torch 2.8+cu128)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("audio", nargs="*",
                   help="Audio/video file(s), wildcard, filelist, or directory.")

    # Model
    p.add_argument("--model", "-m", default="large-v3",
                   help="Whisper model name, alias, local directory or Hugging Face "
                        "repo id. Run --list_models for the aliases. A repo in "
                        "transformers format is converted to CTranslate2 on first use.")
    p.add_argument("--model_dir", default=None,
                   help="Directory to cache/load models. Defaults to _models/ next to exe.")
    p.add_argument("--list_models", action="store_true",
                   help="List the model aliases and exit.")
    p.add_argument("--reconvert", action="store_true",
                   help="Convert the model again even if a converted copy is cached.")
    p.add_argument("--device", "-d", default="auto",
                   help="Device: cuda / cpu / auto.")
    p.add_argument("--compute_type", "-ct",
                   default="default",
                   choices=["default", "auto", "int8", "int8_float16", "int8_float32",
                            "int8_bfloat16", "int16", "float16", "float32", "bfloat16"],
                   help="Quantization type.")

    # Output
    p.add_argument("--output_dir", "-o", default="default",
                   help="Output directory. 'source'=same as input, 'default'=exe dir.")
    p.add_argument("--output_format", "-f", nargs="*",
                   default=["srt"],
                   choices=["json", "lrc", "txt", "text", "vtt", "srt", "tsv", "all"],
                   help="Output format(s).")

    # Transcription
    p.add_argument("--language", "-l", default=None,
                   help="Language code (e.g. ja, en). None=auto-detect.")
    p.add_argument("--task", default="transcribe",
                   choices=["transcribe", "translate"])
    p.add_argument("--temperature", type=float, default=0)
    p.add_argument("--best_of", "-bo", type=int, default=5)
    p.add_argument("--beam_size", "-bs", type=int, default=5)
    p.add_argument("--patience", "-p", type=float, default=2.0)
    p.add_argument("--length_penalty", type=float, default=1.0)
    p.add_argument("--repetition_penalty", type=float, default=1.0)
    p.add_argument("--no_repeat_ngram_size", type=int, default=0)
    p.add_argument("--initial_prompt", "-prompt", default=None,
                   help="Initial prompt text passed to the decoder. "
                        "Omit, or pass 'None'/'auto', to send no prompt.")
    p.add_argument("--condition_on_previous_text", "-condition",
                   type=lambda x: x.lower() != "false", default=False)
    p.add_argument("--compression_ratio_threshold", type=float, default=2.4)
    p.add_argument("--logprob_threshold", type=float, default=-1.0)
    p.add_argument("--no_speech_threshold", type=float, default=0.6)
    p.add_argument("--word_timestamps", "-wt",
                   type=lambda x: x.lower() != "false", default=True)
    p.add_argument("--max_new_tokens", type=int, default=None)
    p.add_argument("--chunk_length", type=int, default=None)
    p.add_argument("--hotwords", default=None)

    p.add_argument("--hallucination_silence_threshold", "-hst", type=float, default=0,
                   help="Skip silent periods longer than this (seconds) when possible hallucination detected. 0=disabled.")
    p.add_argument("--hallucination_silence_th_temp", "-hst_temp", type=float, default=0.5,
                   help="Accepted for Faster-Whisper-XXL command line compatibility "
                        "and ignored: faster-whisper has no matching parameter.")
    p.add_argument("--loop_filter",
                   type=lambda x: x.lower() != "false", default=True,
                   help="Drop segments that are one phrase or character repeated "
                        "(measured: total CER 24.3%% -> 22.0%% on nine "
                        "recordings, with no loss of coverage). Pass false to "
                        "keep them, for instance when a long scream is wanted "
                        "in the subtitle. Dropped segments are listed on the "
                        "[LOOP] line.")

    # Batched inference
    p.add_argument("--batched", action="store_true",
                   help="Enable batched inference (~2x-8x faster, slight quality trade-off).")
    p.add_argument("--batch_size", type=int, default=8)

    # VAD
    p.add_argument("--vad_filter", "-vad",
                   type=lambda x: x.lower() != "false", default=True)
    p.add_argument("--vad_threshold", type=float, default=None,
                   help="Speech probability above which the VAD opens. Left "
                        "unset it resolves per backend, because the scale is "
                        f"model-specific: {VAD_THRESHOLD_DEFAULTS['ten']} for "
                        f"TEN VAD and {VAD_THRESHOLD_DEFAULTS['silero']} for "
                        "silero. The same number does not mean the same "
                        "strictness across the two, so a value tuned on one is "
                        "wrong on the other.")
    p.add_argument("--vad_neg_threshold", type=float, default=None,
                   help="End-of-speech probability for the silero VAD. Speech "
                        "starts above --vad_threshold and only ends below this, "
                        "so lowering it makes the VAD hold through a quiet "
                        "passage instead of cutting there. Defaults to silero's "
                        "own max(threshold - 0.15, 0.01), i.e. 0.30 at the "
                        "default threshold. Silero backends only.")
    p.add_argument("--vad_min_speech_duration_ms", type=int, default=250)
    p.add_argument("--vad_max_speech_duration_s", type=float, default=None)
    p.add_argument("--vad_min_silence_duration_ms", type=int, default=3000)
    p.add_argument("--vad_speech_pad_ms", type=int, default=900)
    p.add_argument("--vad_window_size_samples", type=int, default=1536,
                   help="Silero window size. Only reaches the torch.hub fallback "
                        "in vad.py: the built-in VAD hardcodes 512 samples and "
                        "silero-vad 5.x/6.x no longer accept the argument.")
    p.add_argument("--vad_method",
                   default="ten",
                   choices=["silero_v4_fw", "silero_v5_fw", "silero_v3", "silero_v4",
                            "silero_v5", "silero_v6", "silero_v6_fw",
                            "ten", "precomputed",
                            "pyannote_v3", "pyannote_onnx_v3",
                            "auditok", "webrtc"],
                   help="VAD backend. Defaults to 'ten' (TEN VAD, Apache-2.0), "
                        "which beat silero on all nine references: 19.3% -> "
                        "16.1% whole-region CER, coverage 82.6% -> 86.6%, and "
                        "it wins on eight of the nine files. Segmentation is "
                        "silero's aggregation either way, so only the model "
                        "differs. 'silero_v5' is the previous default. The *_fw "
                        "names all run faster-whisper's bundled model "
                        "regardless of the version in the name; use --vad_onnx "
                        "to actually change it. 'precomputed' reads regions "
                        "from --vad_segments_json, for segmenters that cannot "
                        "be installed here.")
    p.add_argument("--vad_segments_json", default=None, metavar="PATH",
                   help="Speech regions for --vad_method precomputed, as "
                        "{\"<wav stem>\": [[start, end], ...]}. Rasterised onto "
                        "the same frame grid and put through the same "
                        "aggregation as the other backends.")
    p.add_argument("--vad_device", default="cpu",
                   help="Device for VAD model (cpu/cuda). The built-in VAD always "
                        "runs on CPU; this applies to the backends in vad.py.")
    p.add_argument("--vad_onnx", default=None, metavar="PATH",
                   help="Replace the ONNX model used by the built-in VAD. A "
                        "relative path is resolved next to the script or exe. "
                        "Must be a silero v5/v6 style graph (inputs: input, h, c).")
    p.add_argument("--vad_segment_mode", default="clip",
                   choices=["clip", "collect"],
                   help="How segments from an external --vad_method reach the "
                        "model. 'clip' (default) passes clip_timestamps, so each "
                        "clip is zero-padded to 30s. 'collect' cuts the silence "
                        "out of the waveform like the built-in VAD does. "
                        "Measured on a 5h22m recording, 'collect' produced 3.3x "
                        "more looping text and 2.2pt worse CER, and it is the "
                        "only path that needs the >30s segment repair.")

    # Subtitle formatting
    p.add_argument("--standard", action="store_true",
                   help="Standard subtitle preset: --max_line_width=42 --max_line_count=2 --sentence.")
    p.add_argument("--standard_asia", action="store_true",
                   help="Standard preset for Asian languages: width=16, count=2.")
    p.add_argument("--sentence", action="store_true",
                   help="Split output at sentence boundaries.")
    p.add_argument("--max_line_width", type=int, default=1000)
    p.add_argument("--max_line_count", type=int, default=1)
    p.add_argument("--max_gap", type=float, default=3.0)

    # Batch processing
    p.add_argument("--batch_recursive", "-br", action="store_true")
    p.add_argument("--skip", action="store_true",
                   help="Skip if output already exists.")
    p.add_argument("--check_files", action="store_true",
                   help="Check input files before processing.")
    p.add_argument("--print_progress", "-pp", action="store_true")
    p.add_argument("--postfix", action="store_true",
                   help="Add detected language as filename postfix.")
    p.add_argument("--beep_off", action="store_true")

    # Audio filters
    p.add_argument("--ff_track", type=int, default=1, choices=[1, 2, 3, 4, 5, 6],
                   help="Audio track selector.")
    p.add_argument("--ff_fc", action="store_true", help="Select front-center channel only.")
    p.add_argument("--ff_lc", action="store_true", help="Select left channel only.")
    p.add_argument("--ff_invert", action="store_true", help="Invert left channel and mix to mono.")
    p.add_argument("--ff_rnndn_sh", action="store_true", help="RNNoise denoising (SH model).")
    p.add_argument("--ff_rnndn_xiph", action="store_true", help="RNNoise denoising (Xiph model).")
    p.add_argument("--ff_fftdn", type=int, default=0, metavar="[0 - 97]",
                   help="FFT denoising strength (0=off, 12=normal).")
    p.add_argument("--ff_gate", action="store_true", help="Noise gate filter.")
    p.add_argument("--ff_speechnorm", action="store_true", help="Speech normalization.")
    p.add_argument("--ff_loudnorm", action="store_true", help="EBU R128 loudness normalization.")
    p.add_argument("--ff_lowhighpass", action="store_true", help="50Hz-7800Hz bandpass filter.")
    p.add_argument("--ff_tempo", type=float, default=1.0, metavar="[0.5 - 2.0]",
                   help="Adjust audio tempo.")
    p.add_argument("--ff_silence_suppress", nargs=2, type=float,
                   default=[0, 3.0], metavar=("noise", "duration"),
                   help="Silence suppression: noise_dB duration_s.")
    p.add_argument("--ff_vocal_extract", default=None,
                   choices=["mdx_kim2", "mb-roformer"],
                   help="Vocal extraction model.")
    p.add_argument("--mdx_chunk", type=int, default=15,
                   help="Chunk size (seconds) for MDX vocal extraction.")
    p.add_argument("--voc_device", default="cuda",
                   help="Device for vocal extraction.")

    # Config file / profiles
    p.add_argument("--config", default=None, metavar="PATH",
                   help="YAML profile file. Auto-detected as whisp-carrier.yaml "
                        "next to the script or exe when omitted.")
    p.add_argument("--no_config", action="store_true",
                   help="Ignore any YAML profile file.")
    p.add_argument("--profile", default=None, metavar="NAME",
                   help="Profile to activate, overriding active_profile in the YAML file.")
    p.add_argument("--config_override", action="store_true",
                   help="Let YAML values win over options given on the command line. "
                        "Equivalent to 'override: true' in the YAML file.")

    # Misc
    p.add_argument("--model_preload", default=None,
                   type=lambda x: None if x == "None" else x.lower() != "false",
                   help="Accepted for Faster-Whisper-XXL command line compatibility "
                        "and ignored: the model is always loaded once before the "
                        "file loop, so it is effectively always preloaded.")
    p.add_argument("--realign", action="store_true",
                   help="Realign SRT timestamps using stable-ts for improved accuracy.")
    p.add_argument("--realign_device", default=None,
                   help="Device for --realign (auto if not set).")
    p.add_argument("--verbose", "-v",
                   type=lambda x: x.lower() != "false", default=False)
    p.add_argument("--version", action="store_true",
                   help="Show version and exit.")
    p.add_argument("--checkcuda", "-cc", action="store_true",
                   help="Print CUDA device count and exit.")

    return p


# ─────────────────────────────────────────────
# Presets
# ─────────────────────────────────────────────

# Subtitle presets expand into the individual formatting options.
SUBTITLE_PRESETS = {
    "standard": {"max_line_width": 42, "max_line_count": 2, "sentence": True},
    "standard_asia": {"max_line_width": 16, "max_line_count": 2, "sentence": True},
}


def apply_presets(args: argparse.Namespace, explicit: set) -> None:
    """Expand --standard / --standard_asia into the formatting options.

    Values the user set explicitly, on the command line or in the config file,
    are left alone so that '--standard_asia --max_line_width 20' keeps the 20.
    """
    for flag, preset in SUBTITLE_PRESETS.items():
        if not getattr(args, flag, False):
            continue
        for key, value in preset.items():
            if key not in explicit:
                setattr(args, key, value)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main() -> None:
    parser = build_parser()
    argv = sys.argv[1:]
    args = parser.parse_args(argv)

    if args.version:
        _torch = f"torch {TORCH_VERSION}" if TORCH_VERSION else "torch not bundled"
        print(f"whisp-carrier {VERSION} | {runtime_banner()} | {_torch} | "
              f"CUDA: {CUDA_AVAILABLE} | GPU: {CUDA_DEVICE_NAME}")
        sys.exit(0)

    if args.checkcuda:
        # Reports the device count CTranslate2 will actually use, which is the
        # number that matters, and works without torch.
        import ctranslate2
        print(ctranslate2.get_cuda_device_count())
        sys.exit(0)

    if args.list_models:
        for line in whisp_models.describe_aliases():
            print(line)
        sys.exit(0)

    if not args.audio:
        print("Nothing to do. Usage: whisp-carrier [audio ...] [options]")
        print("Run with --help for full option list.")
        sys.exit(1)

    # Config file. Depending on override mode its values either fill in the gaps
    # left by the command line, or win against it outright.
    try:
        cfg = whisp_config.apply(args, parser, build_parser, argv)
    except whisp_config.ConfigError as e:
        print(f"[CONFIG] error: {e}", file=sys.stderr, flush=True)
        sys.exit(2)

    for line in whisp_config.describe(cfg):
        print(line, flush=True)

    # Presets run after the config file so that a profile can set standard_asia,
    # and so explicit width/count values survive the expansion.
    explicit = whisp_config.cli_specified(build_parser, argv) | set(cfg.applied)
    apply_presets(args, explicit)

    # Resolve the VAD threshold after the config file and presets, so that a
    # value set in whisp-carrier.yaml still wins and only an unset one is filled
    # in. See VAD_THRESHOLD_DEFAULTS for why this is per backend.
    if args.vad_threshold is None:
        args.vad_threshold = resolve_vad_threshold(args.vad_method)
        print(f"[VAD] threshold {args.vad_threshold} "
              f"(default for {args.vad_method})", flush=True)

    # Device selection
    device = args.device
    if device == "auto":
        device = "cuda" if CUDA_AVAILABLE else "cpu"

    # Compute type defaults
    compute_type = args.compute_type
    if compute_type == "default":
        compute_type = "float16" if device == "cuda" else "int8"

    # Model directory
    model_dir = args.model_dir
    if model_dir is None:
        exe_dir = Path(getattr(__import__("sys"), "_MEIPASS", Path(sys.argv[0]).parent))
        candidate = exe_dir / "_models"
        model_dir = str(candidate) if candidate.exists() else None

    # Where converted models are cached. Unlike model_dir this is created on
    # demand, because a conversion has to write somewhere.
    cache_root = Path(args.model_dir) if args.model_dir else whisp_config.base_dir() / "_models"

    print(f"whisp-carrier {VERSION}", flush=True)
    print(f"{runtime_banner()} | device={device} | compute={compute_type}", flush=True)
    if CUDA_AVAILABLE:
        print(f"GPU: {CUDA_DEVICE_NAME}", flush=True)
    if CUDA_ENV_DROPPED:
        # Only prints on machines with a CUDA toolkit installed, and says why
        # the bundled libraries are used instead of it.
        print(f"[CUDA] ignoring {' / '.join(CUDA_ENV_DROPPED)}; "
              f"using the bundled CUDA libraries", flush=True)

    # Replacing the built-in VAD model has to happen before anything runs it,
    # because faster-whisper caches the instance on first use.
    if args.vad_onnx:
        try:
            for line in whisp_vad_patch.install_model(args.vad_onnx):
                print(line, flush=True)
        except whisp_vad_patch.VadPatchError as e:
            print(f"[VAD] error: {e}", file=sys.stderr, flush=True)
            sys.exit(2)
    elif args.verbose:
        print(whisp_vad_patch.describe_builtin_model(), flush=True)

    # Collect input files
    files = collect_files(args.audio, args.batch_recursive)

    if not files:
        print("No media files found.", file=sys.stderr)
        sys.exit(1)

    # Validate files if requested
    if args.check_files:
        valid = []
        for f in files:
            if os.path.exists(f):
                valid.append(f)
            else:
                print(f"  [WARN] File not found: {f}", flush=True)
        files = valid

    print(f"\nFiles to process: {len(files)}", flush=True)

    # Resolve --model. Aliases expand, and a transformers-format source is
    # converted to CTranslate2 here rather than failing inside faster-whisper.
    try:
        resolved = whisp_models.resolve(
            args.model,
            cache_root=cache_root,
            compute_type=compute_type,
            device=device,
            force_convert=args.reconvert,
        )
    except whisp_models.ModelError as e:
        print(f"[MODEL] error: {e}", file=sys.stderr, flush=True)
        sys.exit(2)

    for line in resolved.lines:
        print(line, flush=True)

    # Option defaults that belong to the model itself. Applied after the config
    # file so that anything the caller set explicitly still wins.
    for line in whisp_models.apply_model_defaults(args, resolved.spec, explicit):
        print(line, flush=True)

    # Pin the bundled CUDA libraries before CTranslate2 goes looking for them.
    # Must happen before the model is constructed, which is where the first
    # cuBLAS call happens.
    if device == "cuda":
        _pinned = preload_bundled_cuda()
        if _pinned and args.verbose:
            print(f"[CUDA] pinned bundled {', '.join(_pinned)}", flush=True)

    # Load model
    print(f"\nLoading model: {resolved.path}...", flush=True)
    t0 = time.time()
    model = WhisperModel(
        resolved.path,
        device=device,
        compute_type=compute_type,
        download_root=model_dir,
        local_files_only=False,
    )
    print(f"Model loaded in {time.time() - t0:.1f}s", flush=True)

    # Batched inference needs a pipeline wrapper. Passing batch_size straight to
    # WhisperModel.transcribe raises TypeError, which is what --batched used to do.
    # BatchedInferencePipeline.transcribe accepts every argument the plain model
    # takes plus batch_size, so nothing downstream has to change.
    engine = model
    if args.batched:
        from faster_whisper import BatchedInferencePipeline
        engine = BatchedInferencePipeline(model=model)
        print(f"Batched inference enabled (batch_size={args.batch_size})", flush=True)

    # Process files.
    #
    # One bad file does not stop the batch, but it must not be reported as
    # success either. This used to print "[ERROR] ..." and then "All done" and
    # exit 0, so a caller driving this in a batch, Amatsukaze included, could
    # not tell a failed run from a good one: the only signal was a line in the
    # log. An audio filter that fails takes the whole file down with it (no
    # transcript is written), which is exactly the case that has to be visible.
    failures = []
    for i, f in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}]", flush=True)
        try:
            process_single_file(
                f, engine, args,
                device=device,
                compute_type=compute_type,
                model_dir=model_dir,
                model_path=resolved.path,
            )
        except Exception as e:
            failures.append(f)
            print(
                f"  [ERROR] {console_safe(str(f))}: {console_safe(str(e))}",
                file=sys.stderr, flush=True,
            )
            if args.verbose:
                import traceback
                traceback.print_exc()

    if not args.beep_off:
        try:
            import winsound
            winsound.MessageBeep()
        except Exception:
            pass

    if failures:
        print(
            f"\n[whisp-carrier] Failed: {len(failures)} of {len(files)} file(s).",
            file=sys.stderr, flush=True,
        )
        for f in failures:
            print(f"  {console_safe(str(f))}", file=sys.stderr, flush=True)
        sys.exit(1)

    print("\n[whisp-carrier] All done.", flush=True)


if __name__ == "__main__":
    main()
