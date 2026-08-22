#!/usr/bin/env python3
"""
eval/strip_loops.py
Copy a hypothesis folder with the loop segments removed, to get the ceiling of
loop suppression without running inference again.

Why this exists
    Measurement #7 left one open line: novad has the best coverage of every
    configuration (85.0%) and its only weakness is precision (79.0%), which is
    mostly loops. Whether removing them would beat the current default is a
    question the existing JSON can answer, so it should be answered before any
    detector goes into whisp_carrier.py.

What the number means
    An upper bound, and an optimistic one. Dropping a whole segment also drops
    whatever correct text sits inside it, and a real detector has to decide
    while decoding rather than with the finished file in hand. So read the
    result as "the best that perfect segment-level detection could do", not as
    a prediction of what an implementation will reach.

    The detector is score.py's is_loop(), applied to the same plain-normalised
    text score.py charges, so the segments removed here are exactly the ones the
    report counts under "loops". That detector is tuned for scoring and is more
    trigger-happy than production can afford (known issue #1 records a
    phrase-repeat detector that was implemented and disabled for false
    positives), which is another reason this is a ceiling.

    Run it on the baseline as well as on the candidate. Comparing "novad with
    loops removed" against "clip with loops kept" measures two changes at once.

Two detectors
    --detector score (default) uses score.py's is_loop(), which is what the
    report counts, so the result is the ceiling that motivated the feature.

    --detector production uses loop_filter.py, the one that actually ships. Its
    rule 3 requires the repeated stretch to be at least 12 characters, so short
    screams and stutters survive. Running both answers a question the ceiling
    cannot: how much of the available gain the shipped code collects.

Usage
    python eval/strip_loops.py --src _eval/hyp/novad --dst _eval/hyp/novad-noloop
    python eval/strip_loops.py --src _eval/hyp/clip-fixed --dst _eval/hyp/clip-prodloop
                               --detector production
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

import loop_filter  # noqa: E402  (the shipped detector, one level up)
import normalize  # noqa: E402
import score  # noqa: E402


def die(message: str) -> None:
    print(f"[strip] error: {message}", file=sys.stderr)
    raise SystemExit(2)


def detect_score(text: str) -> bool:
    """What the report counts: score.py's detector on plain-level text."""
    folded = normalize.normalize(text, "plain")
    return bool(folded) and score.is_loop(folded)


def detect_production(text: str) -> bool:
    """What whisp-carrier actually does."""
    return loop_filter.is_loop(text)


DETECTORS = {"score": detect_score, "production": detect_production}


def strip(data: Dict, detector) -> Tuple[Dict, List[Dict]]:
    kept: List[Dict] = []
    dropped: List[Dict] = []
    for segment in data.get("segments") or []:
        if detector(str(segment.get("text") or "")):
            dropped.append(segment)
        else:
            kept.append(segment)

    result = dict(data)
    result["segments"] = kept
    return result, dropped


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(
        description="Write a copy of a hypothesis folder without its loop segments.",
    )
    parser.add_argument("--src", required=True, help="Folder of hypothesis JSON.")
    parser.add_argument("--dst", required=True, help="Folder to write the copy to.")
    parser.add_argument("--examples", type=int, default=2,
                        help="Loop examples to print per file.")
    parser.add_argument("--detector", default="score", choices=sorted(DETECTORS),
                        help="'score' for the ceiling, 'production' for the "
                             "detector that ships in loop_filter.py.")
    args = parser.parse_args()
    detector = DETECTORS[args.detector]

    src = Path(args.src).expanduser()
    dst = Path(args.dst).expanduser()
    if not src.is_dir():
        die(f"not a directory: {src}")
    if dst.resolve() == src.resolve():
        die("--dst must differ from --src")
    dst.mkdir(parents=True, exist_ok=True)

    sources = sorted(src.glob("*.json"))
    if not sources:
        die(f"no JSON under {src}")

    print(f"[strip] detector={args.detector}", flush=True)
    total_segments = total_dropped = total_chars = total_dropped_chars = 0
    for source in sources:
        data = json.loads(source.read_text(encoding="utf-8"))
        stripped, dropped = strip(data, detector)

        before = len(data.get("segments") or [])
        chars = sum(
            len(normalize.normalize(str(s.get("text") or ""), "plain"))
            for s in data.get("segments") or []
        )
        dropped_chars = sum(
            len(normalize.normalize(str(s.get("text") or ""), "plain"))
            for s in dropped
        )
        total_segments += before
        total_dropped += len(dropped)
        total_chars += chars
        total_dropped_chars += dropped_chars

        (dst / source.name).write_text(
            json.dumps(stripped, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        share = (dropped_chars / chars * 100.0) if chars else 0.0
        print(
            f"[strip] {source.stem[:34]:34} {before:5d} -> "
            f"{len(stripped['segments']):5d} segments | dropped {len(dropped):3d} "
            f"({dropped_chars:5d} chars, {share:5.1f}% of text)",
            flush=True,
        )
        for segment in dropped[:args.examples]:
            text = normalize.normalize(str(segment.get("text") or ""), "plain")
            print(
                f"           {float(segment['start']):8.1f}-"
                f"{float(segment['end']):8.1f}  {text[:56]}",
                flush=True,
            )

    share = (total_dropped_chars / total_chars * 100.0) if total_chars else 0.0
    print(
        f"[strip] total {total_segments} -> {total_segments - total_dropped} "
        f"segments | dropped {total_dropped} ({total_dropped_chars} chars, "
        f"{share:.1f}% of text) -> {dst}",
        flush=True,
    )


if __name__ == "__main__":
    main()
