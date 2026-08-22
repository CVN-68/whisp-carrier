"""
eval/arib_vtt.py
Reader for the ARIB B24 caption dumps that come out of the recording chain as
.vtt files.

These are not ordinary WebVTT. A cue looks like this:

    00:01:42.536 --> 00:01:46.874
    <v b24caption0><c>%00%00%00%3F%01%1Ajpn%84%={%=}</c></v>
    <v b24caption1><c>...%^PQ</c>（レロニラ）<c>%^J</c>では<c>%^I</c> <c>%^J</c>...

Three things follow from that, and all three change the numbers.

Visible text is what sits *outside* the <c> blocks. Everything inside them is
percent-escaped ARIB control code, so stripping tags alone leaves the control
soup in the text and a line-anchored speaker-name match never fires.

A cue whose visible text is empty is a clear-screen event, not speech. The cues
form a contiguous chain (each end equals the next start) because a caption stays
up until something replaces it, so these empty cues are what actually marks the
gaps.

Ruby is emitted as ordinary positioned text runs drawn at small size. Taken
naively, 「私」's ruby 「わたし」 lands in the reference string next to 私 itself
and inflates it. Ruby runs are identified through the size controls: %^H is SSZ
(small), %^I is MSZ (half width) and %^J is NSZ (normal), so tracking the last
size control seen before each text run separates ruby from body text.

Cue end times are not speech end times. Because of the contiguous chain an end
time only means "this is when the next caption replaced it", so it must not be
used to score timestamp accuracy. Gaps between text-bearing cues are meaningful,
since those come from a deliberate clear.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

# Cue timing, with the trailing cue settings WebVTT permits.
TIMING_RE = re.compile(
    r"(?P<start>\d{1,3}:\d{2}:\d{2}[.,]\d{1,3}|\d{1,2}:\d{2}[.,]\d{1,3})"
    r"\s*-->\s*"
    r"(?P<end>\d{1,3}:\d{2}:\d{2}[.,]\d{1,3}|\d{1,2}:\d{2}[.,]\d{1,3})"
)

VOICE_BLOCK_RE = re.compile(r"<v\b[^>]*>(?P<body>.*?)</v>", re.DOTALL)
CONTROL_BLOCK_RE = re.compile(r"<c>(?P<payload>.*?)</c>", re.DOTALL)
ANY_TAG_RE = re.compile(r"</?[a-zA-Z][^>]*>")

# DRCS glyph definitions. The bitmap is inline, so a music note or any other
# custom character arrives as data rather than as a codepoint.
DRCS_RE = re.compile(r"%\+\{.*?%\+\}", re.DOTALL)

# Character size controls. SSZ marks ruby, MSZ marks half width, NSZ is normal.
SIZE_RE = re.compile(r"%\^([HIJ])")
SIZE_SSZ, SIZE_MSZ, SIZE_NSZ = "H", "I", "J"

# Clear screen. Present in the cues that carry no visible text.
CLEAR_RE = re.compile(r"%0C")

# 「（レロニラ）」 style speaker labels, which these captions use throughout.
SPEAKER_PAREN_RE = re.compile(r"^[（(]\s*([^）)]{1,14})\s*[）)]")
# 「ぐみ：ミラーパクト」 style labels.
SPEAKER_COLON_RE = re.compile(r"^([^\s:：（）()]{1,10})[:：]")

FULLWIDTH_DIGITS = "０１２３４５６７８９"


def parse_timestamp(value: str) -> float:
    value = value.replace(",", ".")
    parts = value.split(":")
    if len(parts) == 2:
        parts = ["0"] + parts
    hours, minutes, seconds = parts
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


@dataclass
class Cue:
    start: float
    end: float
    text: str = ""
    ruby: List[str] = field(default_factory=list)
    drcs: int = 0
    cleared: bool = False

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def has_text(self) -> bool:
        return bool(self.text.strip())


@dataclass
class Stats:
    cues: int = 0
    text_cues: int = 0
    empty_cues: int = 0
    text_seconds: float = 0.0
    ruby_runs: int = 0
    ruby_chars: int = 0
    drcs: int = 0
    music_note: int = 0
    speaker_paren: int = 0
    speaker_colon: int = 0
    fullwidth_digits: int = 0
    halfwidth_space: int = 0
    suspect: Counter = field(default_factory=Counter)
    speakers: Counter = field(default_factory=Counter)
    notes: List[str] = field(default_factory=list)


def _extract_voice(body: str) -> Tuple[str, List[str], int, bool]:
    """Split one <v> block into body text, ruby runs, DRCS count and clear flag.

    Walks the block so that the size control active at each text run is known.
    Size state is local to the block: the management voice is processed by the
    same code and must not leak state into the caption voice.
    """
    text_parts: List[str] = []
    ruby_parts: List[str] = []
    drcs = 0
    cleared = False
    size = SIZE_NSZ
    position = 0

    for match in CONTROL_BLOCK_RE.finditer(body):
        plain = body[position:match.start()]
        if plain:
            (ruby_parts if size == SIZE_SSZ else text_parts).append(plain)

        payload = match.group("payload")
        drcs += len(DRCS_RE.findall(payload))
        if CLEAR_RE.search(payload):
            cleared = True
        # Controls apply in order, so the run that follows is governed by the
        # last size control in this block.
        sizes = SIZE_RE.findall(payload)
        if sizes:
            size = sizes[-1]
        position = match.end()

    trailing = body[position:]
    if trailing:
        (ruby_parts if size == SIZE_SSZ else text_parts).append(trailing)

    text = ANY_TAG_RE.sub("", "".join(text_parts))
    ruby = [ANY_TAG_RE.sub("", part) for part in ruby_parts]
    return text, [r for r in ruby if r.strip()], drcs, cleared


def _is_suspect_char(ch: str) -> bool:
    """Characters that are neither plain Japanese nor ASCII.

    Surfaces gaiji substitutes and decorative symbols so the normalisation step
    can be written against what the files actually contain.
    """
    if ch in "\n\r\t ":
        return False
    code = ord(ch)
    if code < 128:
        return False
    ranges = (
        (0x3040, 0x30FF),  # hiragana / katakana
        (0x4E00, 0x9FFF),  # CJK unified
        (0x3000, 0x303F),  # CJK punctuation
        (0xFF00, 0xFFEF),  # halfwidth / fullwidth forms
    )
    return not any(low <= code <= high for low, high in ranges)


def parse(path: Path) -> Tuple[List[Cue], Stats]:
    """Read an ARIB caption VTT into cues plus a survey of its contents."""
    stats = Stats()
    raw = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    if not raw.lstrip().startswith("WEBVTT"):
        stats.notes.append("no WEBVTT header")

    lines = raw.splitlines()
    cues: List[Cue] = []
    index = 0
    while index < len(lines):
        timing = TIMING_RE.search(lines[index])
        if not timing:
            index += 1
            continue
        cue = Cue(
            start=parse_timestamp(timing.group("start")),
            end=parse_timestamp(timing.group("end")),
        )
        index += 1
        block: List[str] = []
        while index < len(lines) and lines[index].strip():
            block.append(lines[index])
            index += 1

        body = "\n".join(block)
        voices = VOICE_BLOCK_RE.findall(body)
        # A cue without <v> wrappers is treated as one implicit voice, so a
        # plainer dump still parses.
        segments = voices if voices else [body]

        texts: List[str] = []
        for segment in segments:
            text, ruby, drcs, cleared = _extract_voice(segment)
            if text.strip():
                texts.append(text)
            cue.ruby.extend(ruby)
            cue.drcs += drcs
            cue.cleared = cue.cleared or cleared
        cue.text = "\n".join(texts)
        cues.append(cue)

    _survey(cues, stats)
    return cues, stats


def _survey(cues: List[Cue], stats: Stats) -> None:
    stats.cues = len(cues)
    for cue in cues:
        stats.drcs += cue.drcs
        stats.ruby_runs += len(cue.ruby)
        stats.ruby_chars += sum(len(r) for r in cue.ruby)

        if not cue.has_text:
            stats.empty_cues += 1
            continue

        stats.text_cues += 1
        stats.text_seconds += cue.duration

        for line in cue.text.splitlines():
            line = line.strip()
            if not line:
                continue
            paren = SPEAKER_PAREN_RE.match(line)
            colon = SPEAKER_COLON_RE.match(line)
            if paren:
                stats.speaker_paren += 1
                stats.speakers[paren.group(1)] += 1
            elif colon:
                stats.speaker_colon += 1
                stats.speakers[colon.group(1)] += 1

        stats.music_note += cue.text.count("\u266a")
        stats.halfwidth_space += cue.text.count(" ")
        for ch in cue.text:
            if ch in FULLWIDTH_DIGITS:
                stats.fullwidth_digits += 1
            if _is_suspect_char(ch):
                stats.suspect[ch] += 1

    out_of_order = sum(1 for a, b in zip(cues, cues[1:]) if b.start < a.start)
    if out_of_order:
        stats.notes.append(f"{out_of_order} cues out of chronological order")
    inverted = sum(1 for c in cues if c.end <= c.start)
    if inverted:
        stats.notes.append(f"{inverted} cues with end <= start")


def text_regions(cues: List[Cue]) -> List[Tuple[float, float]]:
    """Merged (start, end) spans of the cues that actually carry text.

    These are the regions a caption was on screen. The complement is where a
    clear was issued, which is what the hallucination count is scored against.
    """
    spans = [(c.start, c.end) for c in cues if c.has_text and c.end > c.start]
    spans.sort()
    merged: List[List[float]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(a, b) for a, b in merged]


def gaps(
    cues: List[Cue],
    minimum: float = 0.0,
    total_duration: Optional[float] = None,
) -> List[Tuple[float, float]]:
    """Spans with no caption text, i.e. candidate silence or music regions.

    Pass total_duration to include the span after the last caption. That one
    matters: these captions run as a contiguous chain, so on a single episode
    the only gaps are usually the opening before the first cue and the tail
    after the last, and leaving the tail out hides half the scorable region.
    """
    regions = text_regions(cues)
    result: List[Tuple[float, float]] = []
    previous_end = 0.0
    for start, end in regions:
        if start - previous_end > minimum:
            result.append((previous_end, start))
        previous_end = max(previous_end, end)

    if total_duration is not None and total_duration - previous_end > minimum:
        result.append((previous_end, total_duration))
    return result


def describe_char(ch: str) -> str:
    """Codepoint and name, never the raw glyph.

    The console codepage cannot be assumed to render these, and writing one that
    cp932 lacks would abort a run part way through a survey.
    """
    return f"U+{ord(ch):04X} {unicodedata.name(ch, 'UNNAMED')}"
