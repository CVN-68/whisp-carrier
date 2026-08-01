#!/usr/bin/env python3
"""
whisp-carier.py
RTX 5090 (sm_120 / Blackwell) native compatible faster-whisper CLI.
Drop-in replacement for Faster-Whisper-XXL with torch 2.8+cu128.

Author: whisp-carier contributors
License: MIT
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Optional

from faster_whisper import WhisperModel
from tqdm import tqdm

# ─────────────────────────────────────────────
# Supported media extensions for batch mode
# ─────────────────────────────────────────────
MEDIA_EXTENSIONS = {
    ".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm",
    ".mp3", ".wav", ".flac", ".aac", ".m4a", ".ogg", ".wma",
    ".ts", ".m2ts", ".mts",
}

VERSION = "0.1.0"
TORCH_VERSION = ""
try:
    import torch
    TORCH_VERSION = torch.__version__
    CUDA_AVAILABLE = torch.cuda.is_available()
    CUDA_DEVICE_NAME = torch.cuda.get_device_name(0) if CUDA_AVAILABLE else "N/A"
except Exception:
    CUDA_AVAILABLE = False
    CUDA_DEVICE_NAME = "N/A"


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
            f.write(seg["text"].strip() + "\n")


def write_tsv(segments, output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("start\tend\ttext\n")
        for seg in segments:
            f.write(f"{seg['start']:.3f}\t{seg['end']:.3f}\t{seg['text'].strip()}\n")


def write_lrc(segments, output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        for seg in segments:
            s = seg["start"]
            m = int(s // 60)
            sec = s - m * 60
            f.write(f"[{m:02d}:{sec:05.2f}]{seg['text'].strip()}\n")


def write_json(segments, output_path: str, info: dict) -> None:
    data = {
        "language": info.get("language", ""),
        "duration": info.get("duration", 0),
        "segments": segments,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_outputs(segments, info: dict, base_path: str, formats: List[str]) -> None:
    writers = {
        "srt":  (write_srt,  ".srt"),
        "vtt":  (write_vtt,  ".vtt"),
        "txt":  (write_txt,  ".txt"),
        "text": (write_txt,  ".txt"),
        "tsv":  (write_tsv,  ".tsv"),
        "lrc":  (write_lrc,  ".lrc"),
        "json": (lambda s, p: write_json(s, p, info), ".json"),
    }
    for fmt in formats:
        if fmt == "all":
            for f2, (fn, ext) in writers.items():
                if f2 not in ("text", "all"):
                    fn(segments, base_path + ext)
        elif fmt in writers:
            fn, ext = writers[fmt]
            fn(segments, base_path + ext)


# ─────────────────────────────────────────────
# Core transcription
# ─────────────────────────────────────────────

def transcribe_file(
    audio_path: str,
    model: WhisperModel,
    args: argparse.Namespace,
) -> tuple[List[dict], dict]:
    """Run transcription on a pre-processed audio file."""

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
        initial_prompt=None if args.initial_prompt == "None" else args.initial_prompt,
        condition_on_previous_text=args.condition_on_previous_text,
        compression_ratio_threshold=args.compression_ratio_threshold,
        log_prob_threshold=args.logprob_threshold,
        no_speech_threshold=args.no_speech_threshold,
        word_timestamps=args.word_timestamps,
        hallucination_silence_threshold=getattr(args, "hallucination_silence_threshold", 0) if getattr(args, "hallucination_silence_threshold", 0) > 0 else None,
        vad_filter=args.vad_filter,
        vad_parameters=dict(
            threshold=args.vad_threshold,
            min_speech_duration_ms=args.vad_min_speech_duration_ms,
            max_speech_duration_s=args.vad_max_speech_duration_s or float("inf"),
            min_silence_duration_ms=args.vad_min_silence_duration_ms,
            speech_pad_ms=args.vad_speech_pad_ms,
        ) if args.vad_filter else None,
    )

    # Use custom VAD if method is not the built-in silero
    # Note: faster-whisper 1.2+ uses silero_vad_v6.onnx internally
    builtin_vad_methods = {"silero_v4_fw", "silero_v5_fw", "silero_v6", "silero_v6_fw"}
    if args.vad_filter and args.vad_method not in builtin_vad_methods:
        from vad import get_speech_segments
        print(f"  [VAD] Running {args.vad_method}...", flush=True)
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
        )
        clip_ts = ",".join(f"{s},{e}" for s, e in speech_segs)
        transcribe_kwargs["clip_timestamps"] = clip_ts
        transcribe_kwargs["vad_filter"] = False

    if args.batched:
        transcribe_kwargs["batch_size"] = args.batch_size

    segments_gen, info = model.transcribe(audio_path, **transcribe_kwargs)

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
    model: WhisperModel,
    args: argparse.Namespace,
    device: str = "cuda",
    compute_type: str = "float16",
    model_dir: str = None,
) -> None:
    input_path = str(Path(input_path).resolve())
    print(f"\n[whisp-carier] Processing: {input_path}", flush=True)

    # Determine output base path
    if args.output_dir == "source":
        out_dir = str(Path(input_path).parent)
    elif args.output_dir == "default":
        out_dir = str(Path(input_path).parent) if args.batch_recursive else str(Path(sys.argv[0]).parent)
    else:
        out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    stem = Path(input_path).stem
    if args.postfix:
        # Language postfix added after transcription
        base_out = os.path.join(out_dir, stem)
    else:
        base_out = os.path.join(out_dir, stem)

    # Check if output already exists (--skip / --check_files)
    first_fmt = args.output_format[0] if args.output_format else "srt"
    if first_fmt == "all":
        first_fmt = "srt"
    existing = base_out + f".{first_fmt}"
    if args.skip and os.path.exists(existing):
        print(f"  Skipping (output exists): {existing}", flush=True)
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
        segments, info = transcribe_file(audio_input, model, args)
        elapsed = time.time() - t0
        print(f"  [STT] Done in {elapsed:.1f}s | lang={info['language']} ({info['language_probability']:.2%}) | {len(segments)} segments", flush=True)

        # Postfix language to filename
        if args.postfix:
            base_out = os.path.join(out_dir, f"{stem}.{info['language']}")

        # Write outputs
        write_outputs(segments, info, base_out, args.output_format)

        # --realign: タイムスタンプ再調整
        if getattr(args, "realign", False):
            srt_path = base_out + ".srt"
            if os.path.exists(srt_path):
                print("  [REALIGN] Realigning timestamps...", flush=True)
                try:
                    import stable_whisper
                    realign_device = getattr(args, "realign_device", None) or device
                    # stable-ts でSRTを読み込んでタイムスタンプを調整
                    result = stable_whisper.load_faster_whisper(
                        args.model,
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
        written = [base_out + f".{fmt}" for fmt in args.output_format if fmt != "all"]
        for w in written:
            if os.path.exists(w):
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
        prog="whisp-carier",
        description="whisp-carier - RTX 5090 native faster-whisper CLI (torch 2.8+cu128)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("audio", nargs="*",
                   help="Audio/video file(s), wildcard, filelist, or directory.")

    # Model
    p.add_argument("--model", "-m", default="large-v3",
                   help="Whisper model name or path.")
    p.add_argument("--model_dir", default=None,
                   help="Directory to cache/load models. Defaults to _models/ next to exe.")
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
    p.add_argument("--initial_prompt", "-prompt", default="auto")
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
                   help="Temperature threshold for hallucination detection.")

    # Batched inference
    p.add_argument("--batched", action="store_true",
                   help="Enable batched inference (~2x-8x faster, slight quality trade-off).")
    p.add_argument("--batch_size", type=int, default=8)

    # VAD
    p.add_argument("--vad_filter", "-vad",
                   type=lambda x: x.lower() != "false", default=True)
    p.add_argument("--vad_threshold", type=float, default=0.45)
    p.add_argument("--vad_min_speech_duration_ms", type=int, default=250)
    p.add_argument("--vad_max_speech_duration_s", type=float, default=None)
    p.add_argument("--vad_min_silence_duration_ms", type=int, default=3000)
    p.add_argument("--vad_speech_pad_ms", type=int, default=900)
    p.add_argument("--vad_window_size_samples", type=int, default=1536)
    p.add_argument("--vad_method",
                   default="silero_v5_fw",
                   choices=["silero_v4_fw", "silero_v5_fw", "silero_v3", "silero_v4",
                            "silero_v5", "silero_v6", "silero_v6_fw",
                            "pyannote_v3", "pyannote_onnx_v3",
                            "auditok", "webrtc"],
                   help="VAD backend.")
    p.add_argument("--vad_device", default="cpu",
                   help="Device for VAD model (cpu/cuda).")

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

    # Misc
    p.add_argument("--model_preload", default=None,
                   type=lambda x: None if x == "None" else x.lower() != "false")
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
# Main
# ─────────────────────────────────────────────

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.version:
        print(f"whisp-carier {VERSION} | torch {TORCH_VERSION} | CUDA: {CUDA_AVAILABLE} | GPU: {CUDA_DEVICE_NAME}")
        sys.exit(0)

    if args.checkcuda:
        import torch
        print(torch.cuda.device_count())
        sys.exit(0)

    if not args.audio:
        print("Nothing to do. Usage: whisp-carier [audio ...] [options]")
        print("Run with --help for full option list.")
        sys.exit(1)

    # Apply presets
    if args.standard:
        args.max_line_width = 42
        args.max_line_count = 2
        args.sentence = True

    if args.standard_asia:
        args.max_line_width = 16
        args.max_line_count = 2
        args.sentence = True

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

    print(f"whisp-carier {VERSION}", flush=True)
    print(f"torch {TORCH_VERSION} | device={device} | compute={compute_type}", flush=True)
    if CUDA_AVAILABLE:
        print(f"GPU: {CUDA_DEVICE_NAME}", flush=True)

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

    # Load model
    print(f"\nLoading model: {args.model}...", flush=True)
    t0 = time.time()
    model = WhisperModel(
        args.model,
        device=device,
        compute_type=compute_type,
        download_root=model_dir,
        local_files_only=False,
    )
    print(f"Model loaded in {time.time() - t0:.1f}s", flush=True)

    # Process files
    for i, f in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}]", flush=True)
        try:
            process_single_file(f, model, args, device=device, compute_type=compute_type, model_dir=model_dir)
        except Exception as e:
            print(f"  [ERROR] {f}: {e}", file=sys.stderr, flush=True)
            if args.verbose:
                import traceback
                traceback.print_exc()

    if not args.beep_off:
        try:
            import winsound
            winsound.MessageBeep()
        except Exception:
            pass

    print("\n[whisp-carier] All done.", flush=True)


if __name__ == "__main__":
    main()
