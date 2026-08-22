#!/usr/bin/env python3
"""
eval/prep.py
Survey an evaluation set of Amatsukaze output (MKV + ARIB caption VTT) and
extract audio.

Not part of the CLI. This is the preparation step for the quality comparison in
HANDOVER.md ("今後やるなら" #1 and #6).

Extraction
    Each media file becomes a 16kHz mono pcm_s16le WAV, which is exactly the
    form faster-whisper's decode_audio() and vad.py both want, so measuring
    against the WAV changes nothing about the result. What it removes is a
    decode per run: vad.py shells out to ffmpeg for any input that is not
    already .wav, so an MKV is decoded twice per run as soon as an external
    --vad_method is in play.

Survey
    Reports what the audio tracks really are, since --ff_track's default would
    otherwise pick up a secondary or 5.1 track silently, and reports what the
    reference captions really contain. See eval/arib_vtt.py for why the caption
    files need a parser rather than a tag strip.

VTT lookup
    Captions are matched to media by file stem, searching the whole input tree
    plus any --vtt-dir. The recording chain does not necessarily put them next
    to the media.

Usage
    python eval/prep.py sample --out _eval/wav
    python eval/prep.py sample --survey-only
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import arib_vtt  # noqa: E402

MEDIA_SUFFIXES = (".mkv", ".mp4", ".ts", ".m2ts", ".mts")


def die(message: str) -> None:
    print(f"[prep] error: {message}", file=sys.stderr)
    raise SystemExit(2)


def find_tool(name: str) -> str:
    found = shutil.which(name)
    if not found:
        die(
            f"{name} not found on PATH. Install ffmpeg/ffprobe or add the folder "
            "holding them to PATH."
        )
    return found


# ─────────────────────────────────────────────
# Media probing
# ─────────────────────────────────────────────

@dataclass
class AudioStream:
    index: int
    codec: str
    channels: int
    layout: str
    sample_rate: str
    language: str
    title: str

    def describe(self) -> str:
        bits = [f"a:{self.index} {self.codec} {self.channels}ch"]
        if self.layout and self.layout != "unknown":
            bits.append(self.layout)
        bits.append(f"{self.sample_rate}Hz")
        if self.language:
            bits.append(self.language)
        if self.title:
            bits.append(f"'{self.title}'")
        return " ".join(bits)


@dataclass
class MediaInfo:
    path: Path
    duration: float
    audio: List[AudioStream] = field(default_factory=list)


def probe(ffprobe: str, path: Path) -> MediaInfo:
    result = subprocess.run(
        [
            ffprobe, "-v", "error", "-of", "json",
            "-show_format", "-show_streams", "-select_streams", "a",
            str(path),
        ],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if result.returncode != 0:
        die(f"ffprobe failed on {path.name}: {result.stderr.strip()}")

    data = json.loads(result.stdout or "{}")
    duration = float((data.get("format") or {}).get("duration") or 0.0)

    streams = []
    for index, stream in enumerate(data.get("streams") or []):
        tags = stream.get("tags") or {}
        streams.append(
            AudioStream(
                index=index,
                codec=str(stream.get("codec_name") or "?"),
                channels=int(stream.get("channels") or 0),
                layout=str(stream.get("channel_layout") or "unknown"),
                sample_rate=str(stream.get("sample_rate") or "?"),
                language=str(tags.get("language") or ""),
                title=str(tags.get("title") or ""),
            )
        )
    return MediaInfo(path=path, duration=duration, audio=streams)


def extract(ffmpeg: str, src: Path, dst: Path, track: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    partial = dst.with_suffix(".wav.part")
    result = subprocess.run(
        [
            ffmpeg, "-y", "-v", "error",
            "-i", str(src),
            "-map", f"0:a:{track - 1}",
            "-ar", "16000", "-ac", "1",
            "-c:a", "pcm_s16le", "-f", "wav",
            str(partial),
        ],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if result.returncode != 0:
        partial.unlink(missing_ok=True)
        die(f"extraction failed for {src.name}: {result.stderr.strip()}")
    # Write through a .part file so an interrupted run cannot leave a truncated
    # WAV that the next run would skip as already done.
    partial.replace(dst)


# ─────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────

def format_clock(seconds: float) -> str:
    # Rounded before the split, otherwise 1499.996 renders as 0:24:60.00.
    total = round(max(0.0, seconds), 2)
    hours = int(total // 3600)
    minutes = int((total % 3600) // 60)
    return f"{hours:d}:{minutes:02d}:{total % 60:05.2f}"


def audio_lines(media: MediaInfo, track: int) -> List[str]:
    lines: List[str] = []
    if not media.audio:
        return ["  audio       NONE FOUND"]

    for stream in media.audio:
        marker = " <- --ff_track" if stream.index == track - 1 else ""
        lines.append(f"  audio       {stream.describe()}{marker}")

    if len(media.audio) > 1:
        lines.append(
            f"  WARNING     {len(media.audio)} audio tracks; --ff_track {track} "
            f"selects a:{track - 1}. Confirm this is the main audio and not a "
            "secondary or commentary track."
        )
    chosen = next((s for s in media.audio if s.index == track - 1), None)
    if chosen is None:
        lines.append(f"  WARNING     no track a:{track - 1} in this file")
    elif chosen.channels > 2:
        lines.append(
            f"  NOTE        {chosen.channels}ch source; the downmix to mono "
            "happens here, so --ff_fc / --ff_lc would be a no-op afterwards"
        )
    return lines


def caption_lines(vtt: Path, media_duration: float) -> List[str]:
    cues, stats = arib_vtt.parse(vtt)
    lines = [f"  vtt         {vtt.name}"]

    if not cues:
        lines.append("  WARNING     no cues parsed; check the file format")
        return lines

    lines.append(
        f"  cues        {stats.cues} total | {stats.text_cues} with text | "
        f"{stats.empty_cues} clear-screen"
    )
    coverage = (stats.text_seconds / media_duration * 100) if media_duration else 0.0
    lines.append(
        f"  captioned   {format_clock(stats.text_seconds)} "
        f"({coverage:.1f}% of the file)"
    )

    last_end = max(c.end for c in cues)
    lines.append(
        f"  last cue    ends {format_clock(last_end)} "
        f"(media {format_clock(media_duration)})"
    )
    # Catches a caption/media timeline mismatch. A few seconds of overhang is
    # normal, because the last caption's display time runs past the end of the
    # trimmed programme; only a large overhang means a different edit.
    if media_duration:
        overhang = last_end - media_duration
        if overhang > 10.0:
            lines.append(
                f"  WARNING     captions run {overhang:.1f}s past the media. The "
                "VTT and the MKV are probably not the same edit."
            )
        elif overhang > 0.5:
            lines.append(
                f"  NOTE        last caption overhangs the media end by "
                f"{overhang:.1f}s (benign; clamp cue ends when scoring)"
            )

    holes = arib_vtt.gaps(cues, minimum=20.0, total_duration=media_duration or None)
    hole_seconds = sum(end - start for start, end in holes)
    lines.append(
        f"  gaps >20s   {len(holes)} spans | {format_clock(hole_seconds)} total "
        "<- scorable no-caption region"
    )
    for start, end in sorted(holes, key=lambda g: g[1] - g[0], reverse=True)[:4]:
        lines.append(
            f"                {format_clock(start)} -> {format_clock(end)} "
            f"({end - start:.0f}s)"
        )

    marks = []
    if stats.speaker_paren:
        marks.append(f"paren speaker labels x{stats.speaker_paren}")
    if stats.speaker_colon:
        marks.append(f"colon speaker labels x{stats.speaker_colon}")
    if stats.ruby_runs:
        marks.append(f"ruby runs x{stats.ruby_runs} ({stats.ruby_chars} chars)")
    if stats.drcs:
        marks.append(f"DRCS glyph defs x{stats.drcs}")
    if stats.music_note:
        marks.append(f"U+266A music note x{stats.music_note}")
    if stats.fullwidth_digits:
        marks.append(f"fullwidth digits x{stats.fullwidth_digits}")
    if stats.halfwidth_space:
        marks.append(f"inline spaces x{stats.halfwidth_space}")
    lines.append("  to normalise " + (", ".join(marks) if marks else "(nothing found)"))

    if stats.speakers:
        top = ", ".join(f"{name} x{n}" for name, n in stats.speakers.most_common(5))
        lines.append(f"  speakers    {len(stats.speakers)} distinct | top: {top}")

    if stats.suspect:
        top = ", ".join(
            f"{arib_vtt.describe_char(ch)} x{n}"
            for ch, n in stats.suspect.most_common(6)
        )
        lines.append(f"  non-plain   {sum(stats.suspect.values())} chars: {top}")

    for note in stats.notes:
        lines.append(f"  NOTE        {note}")
    return lines


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def index_vtts(roots: List[Path]) -> Tuple[Dict[str, Path], List[str]]:
    """Map file stem -> VTT path, searching each root recursively."""
    index: Dict[str, Path] = {}
    notes: List[str] = []
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() != ".vtt":
                continue
            existing = index.get(path.stem)
            if existing is not None and existing != path:
                notes.append(
                    f"two VTTs share the stem '{path.stem}': "
                    f"{existing} and {path}; using the first"
                )
                continue
            index[path.stem] = path
    return index, notes


def main() -> None:
    # Output stays ASCII. Runtime output has to survive any console codepage,
    # and a gaiji written straight out would abort the run on cp932.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(
        description="Survey an MKV + ARIB VTT evaluation set and extract 16kHz mono WAV.",
    )
    parser.add_argument("input_dir", help="Folder holding the media (searched recursively).")
    parser.add_argument("--vtt-dir", action="append", default=[], metavar="DIR",
                        help="Extra folder to search for VTTs. Repeatable.")
    parser.add_argument("--out", default="_eval/wav", help="Where to write the WAVs.")
    parser.add_argument("--track", type=int, default=1,
                        help="Audio track to extract, 1-based, matching --ff_track.")
    parser.add_argument("--survey-only", action="store_true",
                        help="Report only, extract nothing.")
    parser.add_argument("--report", default="_eval/prep-report.txt",
                        help="Where to write the report. '-' for stdout only.")
    args = parser.parse_args()

    source = Path(args.input_dir).expanduser()
    if not source.is_dir():
        die(f"not a directory: {source}")

    ffprobe = find_tool("ffprobe")
    ffmpeg = "" if args.survey_only else find_tool("ffmpeg")

    media_files = sorted(
        p for p in source.rglob("*")
        if p.is_file() and p.suffix.lower() in MEDIA_SUFFIXES
    )
    if not media_files:
        die(f"no media files ({', '.join(MEDIA_SUFFIXES)}) under {source}")

    roots = [source] + [Path(d).expanduser() for d in args.vtt_dir]
    vtt_index, index_notes = index_vtts(roots)

    out_dir = Path(args.out).expanduser()
    report: List[str] = [
        f"[prep] {len(media_files)} media file(s) under {source}",
        f"[prep] {len(vtt_index)} VTT(s) indexed from: "
        + ", ".join(str(r) for r in roots),
        f"[prep] track={args.track} "
        + ("survey only" if args.survey_only else f"-> {out_dir}"),
    ]
    for note in index_notes:
        report.append(f"[prep] NOTE {note}")
    report.append("")
    for line in report:
        print(line, flush=True)

    missing: List[str] = []
    for path in media_files:
        info = probe(ffprobe, path)
        block = [f"=== {path.name}", f"  duration    {format_clock(info.duration)}"]
        block.extend(audio_lines(info, args.track))

        if not args.survey_only:
            wav = out_dir / (path.stem + ".wav")
            if wav.is_file():
                block.append(f"  wav         {wav} (exists, skipped)")
            else:
                extract(ffmpeg, path, wav, args.track)
                block.append(f"  wav         {wav}")
            drift = probe(ffprobe, wav).duration - info.duration
            flag = "  <- CHECK" if abs(drift) > 0.5 else ""
            block.append(f"  wav drift   {drift:+.2f}s{flag}")

        vtt = vtt_index.get(path.stem)
        if vtt is None:
            block.append("  vtt         MISSING (no VTT with a matching stem)")
            missing.append(path.name)
        else:
            block.extend(caption_lines(vtt, info.duration))

        report.extend(block)
        report.append("")
        for line in block:
            print(line, flush=True)
        print(flush=True)

    if missing:
        tail = f"[prep] {len(missing)} media file(s) without a VTT: " + ", ".join(missing)
        report.append(tail)
        print(tail, flush=True)

    if args.report != "-":
        target = Path(args.report).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(report) + "\n", encoding="utf-8")
        print(f"[prep] report written: {target}", flush=True)


if __name__ == "__main__":
    main()
