#!/usr/bin/env python3
"""
eval/score.py
Score whisp-carrier output against the ARIB captions.

Three measurements, because they answer different questions.

Hallucination
    Counts hypothesis segments that land in a region the captions leave empty.

    Do not interpret this number. It was meant to be the primary metric for the
    collect/clip comparison and it does not work on this material: a region
    without captions is not a silent region. The 24 minute recordings put
    sponsor credits, promos and the next programme in the tail, and the marathon
    broadcast puts opening and closing themes and next-episode previews between
    episodes, so correct transcription lands here and is counted as invention.
    On the marathon the two configurations scored 5735 and 5820 characters, and
    the widest examples were the closing theme's lyrics repeated once per
    episode. The reference-free anomalies below are what the conclusions rest on.

    The scorable region is small and unevenly distributed, which is a property
    of the captions rather than a choice: they form a contiguous chain, so the
    only gaps are before the first cue, after the last, and between episodes in
    a marathon broadcast. Files whose captions cover 97% or more contribute
    nothing here and are scored by the paired difference instead.

Paired difference
    Where two configurations disagree about whether anything was said. Needs no
    reference at all, so it covers the files with no scorable gap, and it gives a
    short list of timestamps to listen to rather than a number to trust.

Coverage and precision
    The CER split into its two failure modes, from the LCS of reference and
    hypothesis over the captioned region:

        coverage  = LCS / reference length
        precision = LCS / hypothesis length

    A CER cannot tell these apart, because a deletion and a substitution both
    cost 1. Two files can score 42% and 54% with completely different problems:
    one missing half the dialogue, the other inventing half again as much. Only
    printed for the whole-region CER, where it costs a second banded pass.

CER
    Character error rate over the captioned regions, at three normalisation
    levels. This is for #6, anime-whisper against large-v3.

    Hypothesis segments are assigned to reference blocks by midpoint, so a
    timing error shows up as a substitution. That makes the number slightly
    pessimistic and it is the reason the paired difference exists alongside it.
    Cue end times are never used as speech end times: in a contiguous chain an
    end time only marks when the next caption replaced this one.

Usage
    python eval/score.py --hyp _eval/hyp/ext-collect --hyp _eval/hyp/ext-clip
    python eval/score.py --hyp _eval/hyp/large-v3 --hyp _eval/hyp/anime-whisper --cer
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import arib_vtt  # noqa: E402
import normalize  # noqa: E402

# A gap has to be at least this long to be scored. Shorter ones are dominated by
# the caption author's timing slack rather than by real silence.
DEFAULT_MIN_GAP = 20.0

# Reference cues are grouped into blocks of about this length before CER, which
# keeps a small timing offset from being charged as a whole-block substitution.
DEFAULT_BLOCK_SECONDS = 30.0


def die(message: str) -> None:
    print(f"[score] error: {message}", file=sys.stderr)
    raise SystemExit(2)


# ─────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────

@dataclass
class Segment:
    start: float
    end: float
    text: str

    @property
    def mid(self) -> float:
        return (self.start + self.end) / 2.0

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class Hypothesis:
    name: str
    segments: List[Segment]
    duration: float
    language: str = ""


def load_hypothesis(path: Path, name: str) -> Hypothesis:
    data = json.loads(path.read_text(encoding="utf-8"))
    segments = [
        Segment(float(s["start"]), float(s["end"]), str(s.get("text") or ""))
        for s in data.get("segments") or []
    ]
    segments.sort(key=lambda s: s.start)
    return Hypothesis(
        name=name,
        segments=segments,
        duration=float(data.get("duration") or 0.0),
        language=str(data.get("language") or ""),
    )


def index_vtts(roots: Sequence[Path]) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.vtt")):
            index.setdefault(path.stem, path)
    return index


# ─────────────────────────────────────────────
# Metric 1: hallucination in no-caption regions
# ─────────────────────────────────────────────

@dataclass
class GapScore:
    gap_seconds: float = 0.0
    gap_count: int = 0
    segments: int = 0
    seconds: float = 0.0
    chars: int = 0
    worst: List[Tuple[float, float, str]] = field(default_factory=list)

    @property
    def rate_per_minute(self) -> float:
        minutes = self.gap_seconds / 60.0
        return (self.segments / minutes) if minutes else 0.0


def score_gaps(
    gaps: Sequence[Tuple[float, float]],
    hypothesis: Hypothesis,
) -> GapScore:
    result = GapScore(
        gap_seconds=sum(end - start for start, end in gaps),
        gap_count=len(gaps),
    )
    for segment in hypothesis.segments:
        for start, end in gaps:
            # Midpoint containment, so a segment straddling the boundary of a
            # captioned region is not counted as invented.
            if start <= segment.mid < end:
                text = normalize.normalize(segment.text, "markup")
                result.segments += 1
                result.seconds += segment.duration
                result.chars += len(text)
                if text:
                    result.worst.append((segment.start, segment.end, text))
                break
    result.worst.sort(key=lambda item: item[1] - item[0], reverse=True)
    return result


# ─────────────────────────────────────────────
# Metric 2: paired difference between two configurations
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# Metric 1b: reference-free anomalies
# ─────────────────────────────────────────────
#
# The no-caption metric above turned out to be confounded on this material: a
# region without captions is not a silent region. Sponsor credits, promos, the
# next programme and news all sit in the tail of a recording and are real
# speech, so transcribing them correctly was being counted as invention.
#
# These detectors need no reference at all, so they apply to the whole file and
# cannot be fooled that way. They look for the two things that are actually
# wrong with the output rather than for text in the wrong place.

# Whisper's window is 30s, so a segment longer than that cannot come from the
# decoder. In collect mode it comes from timestamp restoration: a segment
# straddling removed silence has its start and end mapped back separately, so
# the span grows by the silence between them while the text stays short.
IMPOSSIBLE_DURATION = 30.0

# Japanese speech runs roughly 5-8 characters a second. Well under 1 is a span
# holding almost no text.
THIN_CHARS_PER_SECOND = 1.0
THIN_MIN_DURATION = 20.0

LOOP_MIN_CHARS = 12
LOOP_MAX_DISTINCT = 2
LOOP_MIN_RUN = 8
LOOP_MIN_UNIT_REPEAT = 4


def longest_char_run(text: str) -> int:
    best = run = 1
    for previous, current in zip(text, text[1:]):
        run = run + 1 if current == previous else 1
        best = max(best, run)
    return best if text else 0


def max_unit_repeat(text: str, max_size: int = 6, limit: int = 600) -> int:
    """Longest run of one repeated substring, e.g. 4 for 'はぁっ' x4.

    Near linear per unit size: once a run is found the scan jumps past it.
    """
    text = text[:limit]
    length = len(text)
    best = 1
    for size in range(1, max_size + 1):
        if length < size * 2:
            break
        index = 0
        while index + size <= length:
            unit = text[index:index + size]
            count = 1
            cursor = index + size
            while text[cursor:cursor + size] == unit:
                count += 1
                cursor += size
            best = max(best, count)
            index = cursor if count > 1 else index + 1
    return best


def is_loop(text: str) -> bool:
    if len(text) >= LOOP_MIN_CHARS and len(set(text)) <= LOOP_MAX_DISTINCT:
        return True
    if longest_char_run(text) >= LOOP_MIN_RUN:
        return True
    return max_unit_repeat(text) >= LOOP_MIN_UNIT_REPEAT


@dataclass
class AnomalyScore:
    segments: int = 0
    loops: int = 0
    loop_chars: int = 0
    impossible: int = 0
    impossible_seconds: float = 0.0
    thin: int = 0
    max_duration: float = 0.0
    examples: List[Tuple[str, float, float, str]] = field(default_factory=list)


def score_anomalies(hypothesis: Hypothesis) -> AnomalyScore:
    result = AnomalyScore(segments=len(hypothesis.segments))
    for segment in hypothesis.segments:
        text = normalize.normalize(segment.text, "plain")
        duration = segment.duration
        result.max_duration = max(result.max_duration, duration)

        if text and is_loop(text):
            result.loops += 1
            result.loop_chars += len(text)
            result.examples.append(("loop", segment.start, segment.end, text))

        if duration > IMPOSSIBLE_DURATION:
            result.impossible += 1
            result.impossible_seconds += duration
            result.examples.append(("over30s", segment.start, segment.end, text))

        if (duration >= THIN_MIN_DURATION
                and len(text) / duration < THIN_CHARS_PER_SECOND):
            result.thin += 1
            result.examples.append(("thin", segment.start, segment.end, text))
    return result


def occupancy(hypothesis: Hypothesis) -> List[Tuple[float, float]]:
    spans = [(s.start, s.end) for s in hypothesis.segments if s.end > s.start]
    spans.sort()
    merged: List[List[float]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(a, b) for a, b in merged]


def subtract(
    spans: Sequence[Tuple[float, float]],
    other: Sequence[Tuple[float, float]],
    minimum: float = 1.0,
) -> List[Tuple[float, float]]:
    """Parts of `spans` not covered by `other`, keeping pieces above `minimum`."""
    result: List[Tuple[float, float]] = []
    for start, end in spans:
        cursor = start
        for other_start, other_end in other:
            if other_end <= cursor:
                continue
            if other_start >= end:
                break
            if other_start > cursor and other_start - cursor >= minimum:
                result.append((cursor, min(other_start, end)))
            cursor = max(cursor, other_end)
            if cursor >= end:
                break
        if end - cursor >= minimum:
            result.append((cursor, end))
    return result


def texts_in(hypothesis: Hypothesis, span: Tuple[float, float]) -> str:
    """Text of every segment overlapping the span.

    Overlap rather than midpoint containment: these spans are the leftover
    pieces after subtracting the other configuration's occupancy, so they are
    often shorter than the segment that produced them and no midpoint falls
    inside. Midpoint matching printed blank examples, which defeats the point
    of listing them.
    """
    start, end = span
    parts = [
        normalize.normalize(s.text, "markup")
        for s in hypothesis.segments
        if s.start < end and s.end > start
    ]
    return " / ".join(p for p in parts if p)


# ─────────────────────────────────────────────
# Metric 3: CER over captioned regions
# ─────────────────────────────────────────────

@dataclass
class Block:
    start: float
    end: float
    reference: str


def build_blocks(
    cues: Sequence[arib_vtt.Cue],
    block_seconds: float = DEFAULT_BLOCK_SECONDS,
    skip_music: bool = True,
    skip_songs: bool = False,
) -> List[Block]:
    """Group consecutive text-bearing cues into blocks for alignment.

    Block edges and the gap that separates two blocks both come from
    Cue.speech_end rather than from the display end. An ARIB cue is on screen
    until the next caption replaces it, so on the display end a caption held
    through a theme song produced one block of ~110s carrying 11 reference
    characters, and every character a recogniser emitted during the song was
    charged against it. See the arib_vtt module docstring.

    Cues carrying a music symbol are dropped by default: the captions represent
    a sung passage with a mark rather than with the lyrics, so scoring a
    recogniser that does transcribe the singing against them measures nothing.

    skip_songs widens that to the two markers this material actually uses, a
    wave dash on its own and a fully bracketed cue (normalize.is_sung). It is
    off by default because every recorded number in HANDOVER was measured
    without it, and turning it on silently would make those unreproducible.
    Pass --exclude-songs to get it.
    """
    blocks: List[Block] = []
    current: List[arib_vtt.Cue] = []

    def flush() -> None:
        if not current:
            return
        blocks.append(
            Block(
                start=current[0].start,
                end=max(c.speech_end for c in current),
                reference="".join(c.text for c in current),
            )
        )
        current.clear()

    previous_end: Optional[float] = None
    for cue in cues:
        if not cue.has_text:
            flush()
            previous_end = None
            continue
        if skip_music and normalize.has_music_mark(cue.text):
            flush()
            previous_end = None
            continue
        if skip_songs and normalize.is_sung(cue.text):
            flush()
            previous_end = None
            continue
        if previous_end is not None and cue.start - previous_end > 2.0:
            flush()
        current.append(cue)
        # max(), not assignment: a cue carrying a lot of text can have a later
        # speech_end than the cue that follows it, and letting the boundary walk
        # backwards would invent a gap. These captions form a contiguous chain so
        # it does not arise here, but the block edges should not depend on that.
        previous_end = (cue.speech_end if previous_end is None
                        else max(previous_end, cue.speech_end))
        if previous_end - current[0].start >= block_seconds:
            flush()
            previous_end = None
    flush()
    return blocks


def score_cer_whole(
    blocks: Sequence[Block],
    hypothesis: Hypothesis,
) -> Dict[str, normalize.CerResult]:
    """CER over the captioned region as a single string.

    The block scoring below assigns each hypothesis segment to one block by its
    midpoint, which silently punishes a model that segments coarsely: if one 30s
    segment carries what the captions split across five cues, all of its text
    lands in one block and the neighbours score as deletions. anime-whisper
    produces roughly a quarter as many segments as large-v3, so on that
    comparison the block number measures segmentation granularity more than it
    measures recognition.

    Concatenating removes alignment granularity from the measurement entirely.
    Read the two together: when this number is close and the block number is
    not, the difference is timing and segmentation rather than words.

    Hypothesis text is taken from the blocks themselves, not from the outer span
    between the first and last block. On a marathon broadcast the no-caption
    regions sit *between* episodes rather than after the last cue, so the outer
    span swallowed 47 minutes of sponsor credits, next-episode previews and
    theme songs and charged all of it as insertions: length ratio 1.11 here
    against 0.97 for the same output scored per block, a 13pt inflation of the
    CER. Regions dropped for carrying a music symbol are excluded for the same
    reason.
    """
    reference = "".join(b.reference for b in blocks)
    hypothesis_text = "".join(
        segment.text
        for segment in hypothesis.segments
        if any(segment.start < b.end and segment.end > b.start for b in blocks)
    )
    # Only the plain level, which is the headline number. Running all three over
    # whole-episode strings triples an already expensive banded alignment for no
    # extra insight: the level-to-level drop is already visible per block.
    #
    # with_lcs doubles the cost of this pair and buys the coverage/precision
    # split, which is the only thing that separates missed speech from invention.
    # A CER on its own cannot: a deletion and a substitution both cost 1.
    return {
        "plain": normalize.score_pairs(
            [(reference, hypothesis_text)], "plain", large=True, with_lcs=True
        )
    }


def score_cer(
    blocks: Sequence[Block],
    hypothesis: Hypothesis,
) -> Tuple[Dict[str, normalize.CerResult], int]:
    """CER at every level, plus the character count left outside every block."""
    pairs: List[Tuple[str, str]] = []
    assigned: set = set()
    for index, block in enumerate(blocks):
        parts = []
        for position, segment in enumerate(hypothesis.segments):
            if block.start <= segment.mid < block.end:
                parts.append(segment.text)
                assigned.add(position)
        pairs.append((block.reference, "".join(parts)))

    outside = sum(
        len(normalize.normalize(segment.text, "plain"))
        for position, segment in enumerate(hypothesis.segments)
        if position not in assigned
    )
    return normalize.score_all_levels(pairs), outside


# ─────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────

def clock(seconds: float) -> str:
    # Rounded before the split, otherwise 1499.96 renders as 0:24:60.0.
    total = round(max(0.0, seconds), 1)
    hours = int(total // 3600)
    minutes = int((total % 3600) // 60)
    return f"{hours:d}:{minutes:02d}:{total % 60:04.1f}"


def short(name: str, width: int = 26) -> str:
    return name if len(name) <= width else name[: width - 1] + "\u2026"


# A file can reach the same CER by missing half the dialogue or by inventing
# half again as much, and the fix is different in each case. This turns the two
# ratios into the one-word reading so the tables do not have to be interpreted
# by hand every time.
def diagnose(result: normalize.CerResult) -> str:
    if not result.has_lcs:
        return "no LCS"
    unrecovered = 1.0 - result.coverage
    unmatched = 1.0 - result.precision
    if unrecovered < 0.05 and unmatched < 0.05:
        return "close on both"
    if unrecovered > unmatched * 1.5:
        return f"MISSED SPEECH dominates ({unrecovered * 100:.0f}% of ref lost)"
    if unmatched > unrecovered * 1.5:
        return f"INVENTION dominates ({unmatched * 100:.0f}% of output unmatched)"
    return (f"both ({unrecovered * 100:.0f}% of ref lost, "
            f"{unmatched * 100:.0f}% of output unmatched)")


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(
        description="Score whisp-carrier output against ARIB captions.",
    )
    parser.add_argument("--hyp", action="append", required=True, metavar="DIR",
                        help="Directory of hypothesis JSON. Repeatable; the "
                             "directory name is used as the configuration name.")
    parser.add_argument("--ref", action="append", default=["sample"], metavar="DIR",
                        help="Directory to search for reference VTTs.")
    parser.add_argument("--min-gap", type=float, default=DEFAULT_MIN_GAP,
                        help="Shortest no-caption region to score.")
    parser.add_argument("--block-seconds", type=float, default=DEFAULT_BLOCK_SECONDS,
                        help="Approximate reference block length for CER.")
    parser.add_argument("--keep-music", action="store_true",
                        help="Include cues carrying a music symbol in CER.")
    parser.add_argument("--exclude-songs", action="store_true",
                        help="Also drop instrumental cues (a wave dash on its "
                             "own) and printed lyrics (a fully bracketed cue) "
                             "from CER, on both sides. The music-symbol test "
                             "never fires on this material because the note is "
                             "a DRCS bitmap. Off by default so the recorded "
                             "numbers stay reproducible.")
    parser.add_argument("--only", action="append", default=[], metavar="SUBSTR",
                        help="Only score files whose name contains this. Repeatable.")
    parser.add_argument("--exclude", action="append", default=[], metavar="SUBSTR",
                        help="Skip files whose name contains this. Applied after "
                             "--only. Repeatable. Useful for setting aside the "
                             "files whose reference carries sung lyrics or sound "
                             "effect notes, which no recogniser can match.")
    parser.add_argument("--examples", type=int, default=3,
                        help="How many example spans to print per file.")
    parser.add_argument("--report", default="_eval/score-report.txt",
                        help="Where to write the report. '-' for stdout only.")
    args = parser.parse_args()

    hyp_dirs = [Path(d).expanduser() for d in args.hyp]
    for path in hyp_dirs:
        if not path.is_dir():
            die(f"not a directory: {path}")
    names = [p.name for p in hyp_dirs]
    if len(set(names)) != len(names):
        die("hypothesis directory names must be unique; they name the configurations")

    vtts = index_vtts([Path(d).expanduser() for d in args.ref])
    if not vtts:
        die(f"no VTTs found under: {', '.join(args.ref)}")

    stems = sorted(
        set.intersection(*[{p.stem for p in d.glob("*.json")} for d in hyp_dirs])
    )
    if not stems:
        die("no stem is present in every hypothesis directory")

    if args.only:
        stems = [s for s in stems if any(token in s for token in args.only)]
    if args.exclude:
        stems = [s for s in stems if not any(token in s for token in args.exclude)]
    if not stems:
        die("no file left after --only/--exclude")

    selection = ""
    if args.only or args.exclude:
        parts = []
        if args.only:
            parts.append("only " + "|".join(args.only))
        if args.exclude:
            parts.append("excluding " + "|".join(args.exclude))
        selection = " | " + ", ".join(parts)

    lines: List[str] = [
        f"[score] configs: {', '.join(names)}",
        f"[score] {len(stems)} file(s) scored | min gap {args.min_gap:.0f}s | "
        f"block {args.block_seconds:.0f}s | "
        f"music cues {'kept' if args.keep_music else 'excluded'} from CER"
        + (" | songs (instrumental + printed lyrics) excluded"
           if args.exclude_songs else "")
        + f"{selection}",
        "",
    ]

    totals_gap: Dict[str, GapScore] = {name: GapScore() for name in names}
    totals_anomaly: Dict[str, AnomalyScore] = {name: AnomalyScore() for name in names}
    totals_cer: Dict[str, Dict[str, normalize.CerResult]] = {
        name: {level: normalize.CerResult() for level in normalize.LEVELS}
        for name in names
    }
    totals_whole: Dict[str, Dict[str, normalize.CerResult]] = {
        name: {level: normalize.CerResult() for level in normalize.LEVELS}
        for name in names
    }
    missing_ref: List[str] = []

    for stem in stems:
        vtt = vtts.get(stem)
        hypotheses = {
            d.name: load_hypothesis(d / f"{stem}.json", d.name) for d in hyp_dirs
        }
        duration = max((h.duration for h in hypotheses.values()), default=0.0)

        lines.append(f"=== {stem}")
        lines.append(f"  duration    {clock(duration)}")
        for name, hypothesis in hypotheses.items():
            total = sum(s.duration for s in hypothesis.segments)
            lines.append(
                f"  {short(name, 11):11} {len(hypothesis.segments)} segments | "
                f"speech {clock(total)} | lang={hypothesis.language}"
            )

        if vtt is None:
            lines.append("  reference   MISSING, only the paired difference applies")
            missing_ref.append(stem)
            cues: List[arib_vtt.Cue] = []
        else:
            cues, stats = arib_vtt.parse(vtt)
            # "captioned" is speech time, not screen time. held_* reports the
            # difference, which is captions left up after the speech ended; that
            # time used to be scored as if the reference asserted speech there.
            held = ""
            if stats.held_cues:
                held = (f" | {stats.held_cues} cue(s) held open past the speech,"
                        f" {clock(stats.held_seconds)} total")
            lines.append(
                f"  reference   {stats.text_cues} text cues | "
                f"captioned {clock(stats.text_seconds)}"
                f" of {clock(stats.screen_seconds)} on screen{held}"
            )

        # --- hallucination ---
        if cues:
            gaps = arib_vtt.gaps(cues, minimum=args.min_gap,
                                 total_duration=duration or None)
            gap_seconds = sum(end - start for start, end in gaps)
            lines.append(
                f"  no-caption  {len(gaps)} span(s) | {clock(gap_seconds)}"
            )
            if gaps:
                for name, hypothesis in hypotheses.items():
                    score = score_gaps(gaps, hypothesis)
                    accumulate_gap(totals_gap[name], score)
                    lines.append(
                        f"    {short(name, 11):11} {score.segments} segs | "
                        f"{score.seconds:.1f}s | {score.chars} chars | "
                        f"{score.rate_per_minute:.2f} segs/min"
                    )
                    for start, end, text in score.worst[: args.examples]:
                        lines.append(
                            f"      {clock(start)}-{clock(end)} {short(text, 60)}"
                        )

        # --- reference-free anomalies ---
        lines.append("  anomalies (no reference needed)")
        for name, hypothesis in hypotheses.items():
            anomaly = score_anomalies(hypothesis)
            accumulate_anomaly(totals_anomaly[name], anomaly)
            lines.append(
                f"    {short(name, 11):11} loops {anomaly.loops} "
                f"({anomaly.loop_chars} chars) | >30s {anomaly.impossible} "
                f"({anomaly.impossible_seconds:.0f}s) | thin {anomaly.thin} | "
                f"longest {anomaly.max_duration:.0f}s"
            )
            seen = 0
            for kind, start, end, text in anomaly.examples:
                if kind == "loop" or end - start > IMPOSSIBLE_DURATION:
                    lines.append(
                        f"      {kind:8} {clock(start)}-{clock(end)} "
                        f"{short(text, 46)}"
                    )
                    seen += 1
                    if seen >= args.examples:
                        break

        # --- paired difference ---
        if len(names) >= 2:
            lines.append("  paired diff")
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    left, right = names[i], names[j]
                    a, b = occupancy(hypotheses[left]), occupancy(hypotheses[right])
                    only_a = subtract(a, b)
                    only_b = subtract(b, a)
                    lines.append(
                        f"    {left} only: {len(only_a)} span(s) / "
                        f"{sum(e - s for s, e in only_a):.1f}s | "
                        f"{right} only: {len(only_b)} span(s) / "
                        f"{sum(e - s for s, e in only_b):.1f}s"
                    )
                    for label, spans, source in (
                        (left, only_a, hypotheses[left]),
                        (right, only_b, hypotheses[right]),
                    ):
                        widest = sorted(spans, key=lambda s: s[1] - s[0],
                                        reverse=True)[: args.examples]
                        for span in widest:
                            text = texts_in(source, span)
                            lines.append(
                                f"      {label} {clock(span[0])}-{clock(span[1])} "
                                f"{short(text, 56)}"
                            )

        # --- CER ---
        if cues:
            blocks = build_blocks(cues, args.block_seconds,
                                  skip_music=not args.keep_music,
                                  skip_songs=args.exclude_songs)
            lines.append(f"  CER over {len(blocks)} block(s)")
            for name, hypothesis in hypotheses.items():
                whole = score_cer_whole(blocks, hypothesis)
                plain_whole = whole["plain"]
                accumulate_cer(totals_whole[name]["plain"], plain_whole)
                lines.append(
                    f"    {short(name, 11):11} whole-region plain "
                    f"{plain_whole.cer * 100:5.1f}%"
                    f"{'' if plain_whole.exact else ' (UPPER BOUND)'} | "
                    f"ref {plain_whole.ref_chars} chars | "
                    f"len ratio {plain_whole.length_ratio:.2f}"
                )
                lines.append(
                    f"    {short(name, 11):11}   coverage "
                    f"{plain_whole.coverage * 100:5.1f}% | precision "
                    f"{plain_whole.precision * 100:5.1f}%"
                    f"{'' if plain_whole.lcs_exact else ' (LOWER BOUND)'} | "
                    f"{plain_whole.matches} of {plain_whole.ref_chars} ref chars "
                    f"recovered | {diagnose(plain_whole)}"
                )

                levels, outside = score_cer(blocks, hypothesis)
                for level in normalize.LEVELS:
                    accumulate_cer(totals_cer[name][level], levels[level])
                plain = levels["plain"]
                lines.append(
                    f"    {short(name, 11):11} per-block    "
                    + " | ".join(
                        f"{level} {levels[level].cer * 100:5.1f}%"
                        for level in normalize.LEVELS
                    )
                    + f" | plain ref {plain.ref_chars} chars | len ratio "
                    f"{plain.length_ratio:.2f} | {outside} chars outside blocks"
                )
        lines.append("")

    # --- totals ---
    lines.append("========== TOTAL")
    lines.append("  no-caption regions")
    for name in names:
        score = totals_gap[name]
        lines.append(
            f"    {short(name, 11):11} {score.segments} segs | {score.seconds:.1f}s | "
            f"{score.chars} chars | over {clock(score.gap_seconds)} of gap | "
            f"{score.rate_per_minute:.2f} segs/min"
        )
    lines.append("  anomalies (no reference needed)")
    for name in names:
        anomaly = totals_anomaly[name]
        lines.append(
            f"    {short(name, 11):11} {anomaly.segments} segments | "
            f"loops {anomaly.loops} ({anomaly.loop_chars} chars) | "
            f">30s {anomaly.impossible} ({anomaly.impossible_seconds:.0f}s) | "
            f"thin {anomaly.thin} | longest {anomaly.max_duration:.0f}s"
        )
    # Whole-region first: it is the number to lead with, because it does not
    # depend on how coarsely either side segments.
    lines.append("  CER, whole captioned region as one string (plain level)")
    for name in names:
        plain = totals_whole[name]["plain"]
        lines.append(
            f"    {short(name, 11):11} plain {plain.cer * 100:5.1f}% | "
            f"ref {plain.ref_chars} chars | len ratio {plain.length_ratio:.2f}"
            + (f" | {plain.approximate} file(s) hit the band cap, UPPER BOUND"
               if not plain.exact else "")
        )
    lines.append("  Coverage and precision (LCS based; splits the CER above)")
    for name in names:
        plain = totals_whole[name]["plain"]
        lines.append(
            f"    {short(name, 11):11} coverage {plain.coverage * 100:5.1f}% | "
            f"precision {plain.precision * 100:5.1f}%"
            + (f" | {plain.lcs_approximate} file(s) hit the band cap, LOWER BOUND"
               if not plain.lcs_exact else "")
            + f" | {diagnose(plain)}"
        )
    lines.append("  CER, per 30s block (also charges timing and segmentation)")
    for name in names:
        plain = totals_cer[name]["plain"]
        lines.append(
            f"    {short(name, 11):11} "
            + " | ".join(
                f"{level} {totals_cer[name][level].cer * 100:5.1f}%"
                for level in normalize.LEVELS
            )
            + f" | plain ref {plain.ref_chars} chars | "
            f"len ratio {plain.length_ratio:.2f}"
        )
    if missing_ref:
        lines.append(f"  files without a reference: {', '.join(missing_ref)}")

    for line in lines:
        print(line, flush=True)

    if args.report != "-":
        target = Path(args.report).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[score] report written: {target}", flush=True)


def accumulate_gap(total: GapScore, part: GapScore) -> None:
    total.gap_seconds += part.gap_seconds
    total.gap_count += part.gap_count
    total.segments += part.segments
    total.seconds += part.seconds
    total.chars += part.chars


def accumulate_anomaly(total: AnomalyScore, part: AnomalyScore) -> None:
    total.segments += part.segments
    total.loops += part.loops
    total.loop_chars += part.loop_chars
    total.impossible += part.impossible
    total.impossible_seconds += part.impossible_seconds
    total.thin += part.thin
    total.max_duration = max(total.max_duration, part.max_duration)


def accumulate_cer(total: normalize.CerResult, part: normalize.CerResult) -> None:
    total.ref_chars += part.ref_chars
    total.hyp_chars += part.hyp_chars
    total.edits += part.edits
    total.blocks += part.blocks
    total.empty_ref_blocks += part.empty_ref_blocks
    total.approximate += part.approximate
    total.matches += part.matches
    total.lcs_pairs += part.lcs_pairs
    total.lcs_approximate += part.lcs_approximate


if __name__ == "__main__":
    main()
