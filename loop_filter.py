#!/usr/bin/env python3
"""
loop_filter.py
Drop the runaway repetition Whisper emits on screams, music and noise.

Why this exists
    Measured against ARIB captions on nine 24 minute recordings, looping output
    was the single thing keeping this project behind Faster-Whisper-XXL. One
    file (a children's show, full of shouting) produced 1091 characters of
    「あああ...」 and 「やったーーー...」, which cost 19 CER points on that file and
    turned an otherwise clear win into a tie overall. Removing those segments
    from the recorded output ahead of time put the total at 22.0% against XXL's
    23.7%, so the ceiling was worth implementing (HANDOVER 測定結果 #11).

    faster-whisper's own guards do not catch this. `condition_on_previous_text`
    is already false here, `no_repeat_ngram_size` operates on tokens inside one
    decode step, and `hallucination_silence_threshold` was measured to take
    valid segments with it (212 -> 104). The repetition that survives all three
    is a whole segment that says one thing many times, which is cheap to spot
    after the fact.

What counts as a loop
    Three rules, on text with whitespace, punctuation and width folded away:

        1. at least 12 characters drawn from at most 2 distinct ones
        2. the same character 8 or more times in a row
        3. a unit of 1-6 characters repeated 4 or more times, *and* the
           repeated stretch is at least 12 characters long

    Rules 1 and 2 are what eval/score.py counts under "loops", so the numbers
    here line up with the report that motivated the feature.

    The span requirement in rule 3 is the one deliberate difference. Without it
    the rule fires on 「きゃあああああ」 and 「そそそそんなわけ」, which are a scream and
    a stutter: real speech, and the sort of line this project is supposed to be
    better at than a generic model. The measured CER cost of keeping them is
    0.1 points either way, so the tie-breaker is that a subtitle should not
    silently lose a cue an ASR got right.

Scope
    Detection only looks at one segment. A phrase repeated across consecutive
    segments is handled separately by the duplicate check in whisp_carrier.py.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

# Rule 1: a long segment spelled with almost no distinct characters.
LOOP_MIN_CHARS = 12
LOOP_MAX_DISTINCT = 2

# Rule 2: one character held down.
LOOP_MIN_RUN = 8

# Rule 3: a repeated unit, and how much text it has to cover. The span is what
# separates 「うっうっうっうっ...」 running for 30 seconds from a four-mora scream.
LOOP_MIN_UNIT_REPEAT = 4
LOOP_MIN_UNIT_SPAN = 12
LOOP_MAX_UNIT_SIZE = 6

# Detection is linear in this, and a 30 second segment cannot hold much more.
# Whatever a loop is, it shows itself well before the cap.
LOOP_SCAN_LIMIT = 4000

# Same character class eval/normalize.py drops at its "plain" level, so the
# detector sees the same text the measurement scored. Long vowel marks stay:
# they are phonemic, and 「やったーーーーー」 is caught by the run rule anyway.
_PUNCT_CHARS = (
    "、。，．・：；！？!?,.:;"
    "…‥"
    "「」『』〈〉《》【】〔〕［］[]｛｝{}"
    "\u201c\u201d\u2018\u2019\"'"
    "\u301c\uff5e~"
    "\u2010\u2011\u2012\u2013\u2014\u2015-"
    "\u3000 \t\r\n"
)
_PUNCT_RE = re.compile("[" + re.escape(_PUNCT_CHARS) + "]")
_SPACE_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Fold away everything that would hide a repetition."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _PUNCT_RE.sub("", text)
    return _SPACE_RE.sub("", text)


def longest_char_run(text: str) -> int:
    if not text:
        return 0
    best = run = 1
    for previous, current in zip(text, text[1:]):
        run = run + 1 if current == previous else 1
        if run > best:
            best = run
    return best


def max_unit_repeat(
    text: str,
    max_size: int = LOOP_MAX_UNIT_SIZE,
) -> Tuple[int, int, str]:
    """Longest run of one repeated substring: (count, span, unit).

    Near linear per unit size, because a run that repeats is skipped past
    rather than rescanned from the next character.
    """
    text = text[:LOOP_SCAN_LIMIT]
    length = len(text)
    best = (1, 0, "")
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
            if count > 1 and count * size > best[1]:
                best = (count, count * size, unit)
            index = cursor if count > 1 else index + 1
    return best


def reason(text: str) -> Optional[str]:
    """Why this text is a loop, or None. The string goes into the log."""
    folded = normalize(text)
    if not folded:
        return None

    if len(folded) >= LOOP_MIN_CHARS and len(set(folded)) <= LOOP_MAX_DISTINCT:
        return f"{len(folded)} chars over {len(set(folded))} distinct"

    run = longest_char_run(folded)
    if run >= LOOP_MIN_RUN:
        return f"{run}x the same character in a row"

    count, span, unit = max_unit_repeat(folded)
    if count >= LOOP_MIN_UNIT_REPEAT and span >= LOOP_MIN_UNIT_SPAN:
        return f"{unit!r} x{count} spanning {span} chars"

    return None


def is_loop(text: str) -> bool:
    return reason(text) is not None


def filter_segments(
    segments: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Drop looping segments from a list.

    Returns (segments, stats). The same list object comes back when nothing
    matched, so ids stay untouched in the common case.

    Whole segments go, rather than the repeated stretch inside them. Measured
    on the evaluation set, doing so cost 0.1 points of coverage: a looping
    segment holds almost no text that matches the captions, so there is nothing
    to salvage and no reason to pay for the complexity of trying.
    """
    stats: Dict[str, Any] = {"dropped": 0, "chars": 0, "examples": []}
    if not segments:
        return segments, stats

    kept: List[Dict[str, Any]] = []
    for segment in segments:
        text = str(segment.get("text") or "")
        why = reason(text)
        if why is None:
            kept.append(segment)
            continue
        stats["dropped"] += 1
        stats["chars"] += len(normalize(text))
        if len(stats["examples"]) < 3:
            stats["examples"].append(
                (float(segment.get("start") or 0.0),
                 float(segment.get("end") or 0.0),
                 text.strip(), why)
            )

    if not stats["dropped"]:
        return segments, stats
    return kept, stats


def describe(stats: Dict[str, Any], before: int, after: int) -> Optional[str]:
    """One log line, or None when nothing was dropped."""
    if not stats.get("dropped"):
        return None
    return (
        f"  [LOOP] {before} -> {after} segments | dropped {stats['dropped']} "
        f"({stats['chars']} chars of repetition)"
    )
