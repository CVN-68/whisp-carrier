#!/usr/bin/env python3
"""
eval/ext_vad_dump.py
Dump speech regions from a segmenter that cannot live in the main environment.

Why this exists
    inaSpeechSegmenter wants tensorflow[and-cuda] plus onnxruntime-gpu, and
    funasr-onnx pins numpy<=1.26.4. Installing either into the environment every
    recorded number was measured in would put the whole HANDOVER out of date. So
    each one gets its own virtualenv, runs this, and writes the regions it found
    to JSON. whisp_carrier then reads them with --vad_method precomputed, which
    rasterises them onto the same frame grid and applies the same aggregation as
    the built-in backends, so the comparison is of detection rather than of each
    project's smoothing.

Usage (from the isolated virtualenv, not the main one)
    _venv_inass\\Scripts\\python.exe eval/ext_vad_dump.py --backend inass \\
        --wav-dir _eval/wav --out _eval/vad-inass.json --only 2026030522

    _venv_fsmn\\Scripts\\python.exe eval/ext_vad_dump.py --backend fsmn \\
        --wav-dir _eval/wav --out _eval/vad-fsmn.json --only 2026030522

Output
    {"<wav stem>": [[start_seconds, end_seconds], ...], ...}
    plus a sibling .log recording what each backend reported, because the label
    mix is the interesting part for inaSpeechSegmenter: it separates music and
    noise from speech, which is the reason it is worth testing at all.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

Region = Tuple[float, float]

# inaSpeechSegmenter labels. Only speech counts; the rest is what it is being
# tested for its ability to set aside.
INASS_SPEECH_LABELS = {"speech", "male", "female"}


def die(message: str) -> None:
    print(f"[ext-vad] error: {message}", file=sys.stderr)
    raise SystemExit(2)


def merge(regions: List[Region]) -> List[Region]:
    """Sort and merge touching regions. Downstream expects them disjoint."""
    if not regions:
        return []
    regions = sorted(regions)
    merged = [list(regions[0])]
    for start, end in regions[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(a, b) for a, b in merged]


def run_inass(paths: List[Path], notes: List[str]) -> Dict[str, List[Region]]:
    try:
        from inaSpeechSegmenter import Segmenter
    except ImportError as exc:
        die(f"inaSpeechSegmenter not importable in this interpreter: {exc}")

    # vad_engine='smn' gives the speech / music / noise split, which is the
    # whole point here. Gender detection is off: it would subdivide speech
    # without changing which regions are speech.
    segmenter = Segmenter(vad_engine="smn", detect_gender=False)

    table: Dict[str, List[Region]] = {}
    for path in paths:
        started = time.time()
        segmentation = segmenter(str(path))
        by_label: Dict[str, float] = {}
        regions: List[Region] = []
        for label, start, end in segmentation:
            by_label[label] = by_label.get(label, 0.0) + (end - start)
            if label in INASS_SPEECH_LABELS:
                regions.append((float(start), float(end)))
        regions = merge(regions)
        table[path.stem] = regions
        spread = " ".join(f"{k}={v:.0f}s" for k, v in sorted(by_label.items()))
        notes.append(
            f"{path.name}: {len(regions)} speech region(s), "
            f"{sum(e - s for s, e in regions):.0f}s | {spread} "
            f"| {time.time() - started:.0f}s"
        )
        print(f"  {notes[-1]}", flush=True)
    return table


def run_fsmn(paths: List[Path], notes: List[str]) -> Dict[str, List[Region]]:
    try:
        from funasr_onnx import Fsmn_vad
    except ImportError as exc:
        die(f"funasr_onnx not importable in this interpreter: {exc}")

    # Pulled from ModelScope on first use and cached. quantize=True is not a
    # choice: the ONNX repo ships only model_quant.onnx (495 kB), and asking for
    # the full-precision graph makes funasr_onnx tell you to install funasr and
    # export it yourself. So this measures the quantised graph, which is what the
    # published artefact is.
    model = Fsmn_vad("iic/speech_fsmn_vad_zh-cn-16k-common-onnx", quantize=True)

    table: Dict[str, List[Region]] = {}
    for path in paths:
        started = time.time()
        result = model(str(path))
        # The wrapper returns a list (one entry per input) of [start_ms, end_ms].
        raw = result[0] if result and isinstance(result[0], list) else result
        regions = merge([
            (float(a) / 1000.0, float(b) / 1000.0)
            for a, b in raw
            if float(b) > float(a)
        ])
        table[path.stem] = regions
        notes.append(
            f"{path.name}: {len(regions)} speech region(s), "
            f"{sum(e - s for s, e in regions):.0f}s "
            f"| {time.time() - started:.0f}s"
        )
        print(f"  {notes[-1]}", flush=True)
    return table


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(
        description="Dump speech regions from an isolated segmenter.",
    )
    parser.add_argument("--backend", required=True, choices=["inass", "fsmn"])
    parser.add_argument("--wav-dir", default="_eval/wav")
    parser.add_argument("--out", required=True)
    parser.add_argument("--only", action="append", default=[], metavar="SUBSTR")
    parser.add_argument("--exclude", action="append", default=[], metavar="SUBSTR")
    args = parser.parse_args()

    wav_dir = Path(args.wav_dir)
    if not wav_dir.is_dir():
        die(f"not a directory: {wav_dir}")
    paths = sorted(wav_dir.glob("*.wav"))
    if args.only:
        paths = [p for p in paths if any(t in p.name for t in args.only)]
    if args.exclude:
        paths = [p for p in paths if not any(t in p.name for t in args.exclude)]
    if not paths:
        die(f"no WAVs selected under {wav_dir}")

    print(f"[ext-vad] backend={args.backend} | {len(paths)} file(s)", flush=True)
    notes: List[str] = []
    runner = run_inass if args.backend == "inass" else run_fsmn
    table = runner(paths, notes)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table, ensure_ascii=False, indent=1), encoding="utf-8")
    out.with_suffix(".log").write_text(
        f"backend={args.backend}\n" + "\n".join(notes) + "\n", encoding="utf-8"
    )
    print(f"[ext-vad] wrote {out} ({len(table)} file(s))", flush=True)


if __name__ == "__main__":
    main()
