#!/usr/bin/env python3
"""
eval/xxl_convert.py
Convert Faster-Whisper-XXL JSON output into the schema eval/score.py reads.

Why this exists
    XXL writes {segments, language, text} with no duration, and its words use
    the key "word" where whisp-carrier writes "word" too but the rest of the
    envelope differs. score.py needs duration, language and segment
    start/end/text, so the gap is small but it has to be closed before the two
    can go into the same report.

    duration is not in XXL's output at all. It is taken from the matching
    whisp-carrier JSON when one exists, because that is the number the rest of
    the report is already normalised against, and falling back to ffprobe would
    risk a different value for the same file.

Usage
    python eval/xxl_convert.py --src _eval/hyp-xxl-raw --dst _eval/hyp/xxl
                               --duration-from _eval/hyp/clip-fixed
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


def die(message: str) -> None:
    print(f"[xxl] error: {message}", file=sys.stderr)
    raise SystemExit(2)


def probe_duration(media: Path) -> Optional[float]:
    """Last resort when no reference JSON carries the duration."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(media)],
            capture_output=True, text=True, check=True,
        )
        return float(result.stdout.strip())
    except Exception:
        return None


def convert(source: Path, duration: float) -> Dict:
    data = json.loads(source.read_text(encoding="utf-8"))
    segments: List[Dict] = []
    for index, segment in enumerate(data.get("segments") or [], 1):
        words = []
        for word in segment.get("words") or []:
            # XXL keeps the leading space in "word"; strip it so the text
            # reconstructed from words matches the segment text the same way
            # whisp-carrier's does.
            words.append({
                "start": float(word["start"]),
                "end": float(word["end"]),
                "word": str(word.get("word") or "").strip(),
                "probability": float(word.get("probability") or 0.0),
            })
        segments.append({
            "id": int(segment.get("id") or index),
            "start": float(segment["start"]),
            "end": float(segment["end"]),
            "text": str(segment.get("text") or "").strip(),
            "words": words,
        })
    return {
        "duration": duration,
        "language": str(data.get("language") or ""),
        "segments": segments,
    }


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(
        description="Convert Faster-Whisper-XXL JSON to the eval/score.py schema.",
    )
    parser.add_argument("--src", required=True, help="Folder of XXL JSON output.")
    parser.add_argument("--dst", required=True, help="Folder to write converted JSON.")
    parser.add_argument("--duration-from", default=None, metavar="DIR",
                        help="Folder of whisp-carrier JSON to take duration from.")
    parser.add_argument("--wav-dir", default="_eval/wav",
                        help="Used only if a duration cannot be found otherwise.")
    args = parser.parse_args()

    src = Path(args.src).expanduser()
    dst = Path(args.dst).expanduser()
    if not src.is_dir():
        die(f"not a directory: {src}")
    dst.mkdir(parents=True, exist_ok=True)

    reference = Path(args.duration_from).expanduser() if args.duration_from else None
    wav_dir = Path(args.wav_dir).expanduser()

    sources = sorted(src.glob("*.json"))
    if not sources:
        die(f"no JSON under {src}")

    written = 0
    for source in sources:
        duration = None
        if reference is not None:
            twin = reference / source.name
            if twin.is_file():
                duration = float(
                    json.loads(twin.read_text(encoding="utf-8")).get("duration") or 0.0
                ) or None
        if duration is None:
            duration = probe_duration(wav_dir / f"{source.stem}.wav")
        if duration is None:
            print(f"[xxl] skip (no duration): {source.name}", flush=True)
            continue

        converted = convert(source, duration)
        target = dst / source.name
        target.write_text(
            json.dumps(converted, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        written += 1
        speech = sum(s["end"] - s["start"] for s in converted["segments"])
        print(
            f"[xxl] {source.stem[:34]:34} {len(converted['segments']):5d} segments | "
            f"speech {speech / 60:6.1f}m | duration {duration / 60:6.1f}m",
            flush=True,
        )

    print(f"[xxl] wrote {written} file(s) to {dst}", flush=True)


if __name__ == "__main__":
    main()
