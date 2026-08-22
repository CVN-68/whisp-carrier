"""
subtitle_format.py
Subtitle line formatting for whisp-carrier.

Implements the behaviour promised by --sentence, --max_line_width,
--max_line_count, --max_gap and the --standard / --standard_asia presets.

Everything operates on a word stream rather than on raw strings, so cues that
get split keep accurate timestamps. When Whisper ran without word timestamps,
pseudo words are synthesised from the segment text with linearly interpolated
timings, which keeps the rest of the pipeline uniform.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

# Languages written without spaces between words: wrapping happens per character.
CJK_LANGS = {"ja", "zh", "yue", "th", "lo", "my", "km"}

# A sentence is considered finished after one of these characters, optionally
# followed by closing punctuation.
SENTENCE_END = "。．.！!？?…‥"

# Closing punctuation ignored when looking backwards for a sentence terminator.
CLOSING = "」』｣）)】〉》〕］]｝}”’\"'"

# Characters that may not begin a line (light kinsoku shori).
NO_LINE_START = (
    "、。，．,.!！?？:;：；"
    "）)｝}】」』〉》］]〕｣"
    "”’\"'…‥ー々ゝゞ"
    "ぁぃぅぇぉっゃゅょゎゕゖ"
    "ァィゥェォッャュョヮヵヶ"
)

# Sentinel defaults from build_parser(); formatting is a no-op at these values.
_WIDTH_SENTINEL = 1000
_COUNT_SENTINEL = 1

_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def is_enabled(args: Any) -> bool:
    """True when the user asked for any kind of line formatting."""
    return bool(
        getattr(args, "sentence", False)
        or _int_opt(args, "max_line_width", _WIDTH_SENTINEL) < _WIDTH_SENTINEL
        or _int_opt(args, "max_line_count", _COUNT_SENTINEL) > _COUNT_SENTINEL
    )


def _int_opt(args: Any, name: str, fallback: int) -> int:
    value = getattr(args, name, fallback)
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _is_cjk(language: Optional[str], text: str) -> bool:
    """Decide whether to wrap per character or per word."""
    if language:
        return language.lower() in CJK_LANGS
    return bool(_CJK_RE.search(text))


def _render(words: List[Dict[str, Any]]) -> str:
    """Join a word list back into displayable text."""
    return "".join(str(w.get("word", "")) for w in words).strip()


def _pseudo_words(seg: Dict[str, Any], is_cjk: bool) -> List[Dict[str, Any]]:
    """Synthesise a word stream with interpolated timings from segment text."""
    text = str(seg.get("text", ""))
    if not text.strip():
        return []

    if is_cjk:
        units = [c for c in text if c.strip()]
    else:
        units = re.findall(r"\s*\S+", text)
    if not units:
        return []

    start = float(seg.get("start", 0.0))
    end = float(seg.get("end", start))
    span = max(0.0, end - start)
    total = sum(len(u) for u in units) or 1

    out: List[Dict[str, Any]] = []
    acc = 0
    for unit in units:
        w_start = start + span * acc / total
        acc += len(unit)
        w_end = start + span * acc / total
        out.append({"word": unit, "start": round(w_start, 3), "end": round(w_end, 3)})
    return out


def _words_of(seg: Dict[str, Any], is_cjk: bool) -> List[Dict[str, Any]]:
    """Return a usable word stream for a segment, real or synthesised."""
    words = seg.get("words") or []
    usable = [
        w for w in words
        if isinstance(w, dict)
        and w.get("start") is not None
        and w.get("end") is not None
        and str(w.get("word", "")).strip()
    ]
    if usable:
        return usable
    return _pseudo_words(seg, is_cjk)


def _ends_sentence(text: str) -> bool:
    stripped = text.rstrip().rstrip(CLOSING)
    return bool(stripped) and stripped[-1] in SENTENCE_END


def _split_sentences(
    words: List[Dict[str, Any]],
    max_gap: Optional[float],
) -> List[List[Dict[str, Any]]]:
    """Split a word stream at sentence terminators and at long pauses."""
    groups: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []

    for i, word in enumerate(words):
        current.append(word)
        boundary = _ends_sentence(_render(current))

        if not boundary and max_gap is not None and i + 1 < len(words):
            try:
                gap = float(words[i + 1]["start"]) - float(word["end"])
            except (TypeError, ValueError, KeyError):
                gap = 0.0
            if gap > max_gap:
                boundary = True

        if boundary:
            groups.append(current)
            current = []

    if current:
        groups.append(current)
    return groups


def _wrap(words: List[Dict[str, Any]], max_width: int) -> List[List[Dict[str, Any]]]:
    """Group words into lines whose rendered length stays within max_width."""
    lines: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []

    for word in words:
        if not current:
            current = [word]
            continue

        if len(_render(current + [word])) <= max_width:
            current.append(word)
            continue

        # Kinsoku shori: never let a line begin with punctuation that may not
        # lead, even if that pushes the previous line past max_width.
        lead = str(word.get("word", "")).strip()[:1]
        if lead and lead in NO_LINE_START:
            current.append(word)
            continue

        lines.append(current)
        current = [word]

    if current:
        lines.append(current)
    return lines


def _chunk(lines: List[List[Dict[str, Any]]], max_count: int) -> List[List[List[Dict[str, Any]]]]:
    return [lines[i:i + max_count] for i in range(0, len(lines), max_count)]


def _carry_words(words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep only the fields the JSON writer expects."""
    out = []
    for w in words:
        entry = {"word": w.get("word", ""), "start": w.get("start"), "end": w.get("end")}
        if w.get("probability") is not None:
            entry["probability"] = w["probability"]
        out.append(entry)
    return out


# ─────────────────────────────────────────────
# Always-on sanitisation
# ─────────────────────────────────────────────
#
# Whisper decodes in 30s windows, so a segment longer than this cannot have come
# out of the decoder. It comes from timestamp restoration: when the VAD removes
# silence from the waveform, a segment whose first and last words sit on
# opposite sides of a removed pause has its start and end mapped back
# separately, so the span grows by the length of the pause while the text stays
# short. Measured on nine episodes with large-v3, the built-in VAD produced 14
# such segments totalling 1581s, the longest running 370s. A cue displayed for
# 370s is broken output, so this is repaired regardless of the formatting
# options, which previously only fixed it when --sentence happened to be on.
IMPOSSIBLE_DURATION = 30.0


def _real_words(seg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Word stream with timings, or empty. Never synthesises.

    Pseudo words interpolate evenly across the segment span, so on exactly the
    segments this repairs they would invent pauses that are not there. Without
    real words the only safe repair is to bound the duration.
    """
    return [
        w for w in (seg.get("words") or [])
        if isinstance(w, dict)
        and w.get("start") is not None
        and w.get("end") is not None
        and str(w.get("word", "")).strip()
    ]


def _widest_gap_index(group: List[Dict[str, Any]]) -> Optional[int]:
    """Index to cut after, at the widest pause between consecutive words."""
    best_index: Optional[int] = None
    best_gap = -1.0
    for index in range(len(group) - 1):
        try:
            gap = float(group[index + 1]["start"]) - float(group[index]["end"])
        except (TypeError, ValueError, KeyError):
            continue
        if gap > best_gap:
            best_gap, best_index = gap, index
    return best_index


def _enforce_duration(
    groups: List[List[Dict[str, Any]]],
    max_duration: float,
) -> List[List[Dict[str, Any]]]:
    """Cut any group still longer than max_duration at its widest pause."""
    result: List[List[Dict[str, Any]]] = []
    pending = list(groups)
    guard = 0
    while pending:
        guard += 1
        if guard > 10000:  # pragma: no cover - defensive
            result.extend(pending)
            break
        group = pending.pop(0)
        if len(group) < 2:
            result.append(group)
            continue
        try:
            span = float(group[-1]["end"]) - float(group[0]["start"])
        except (TypeError, ValueError, KeyError):
            result.append(group)
            continue
        if span <= max_duration:
            result.append(group)
            continue
        cut = _widest_gap_index(group)
        if cut is None:
            result.append(group)
            continue
        pending.insert(0, group[cut + 1:])
        pending.insert(0, group[:cut + 1])
        # The two halves go back through the check, so a segment spanning
        # several removed pauses is cut at each of them.
        result.extend([])
    return [g for g in result if g]


def sanitize_segments(
    segments: List[Dict[str, Any]],
    max_gap: Optional[float] = 3.0,
    max_duration: float = IMPOSSIBLE_DURATION,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Split segments that straddle a pause removed from the waveform.

    Returns (segments, stats). The input list is returned unchanged, same
    object, when nothing needed repairing, so ids stay stable in the common
    case.
    """
    stats: Dict[str, Any] = {
        "split": 0, "clamped": 0, "capped": 0, "added": 0,
        "longest_before": 0.0, "longest_after": 0.0,
    }
    if not segments:
        return segments, stats

    try:
        gap_limit = float(max_gap) if max_gap is not None else None
    except (TypeError, ValueError):
        gap_limit = None
    if gap_limit is not None and gap_limit <= 0:
        gap_limit = None

    out: List[Dict[str, Any]] = []
    changed = False

    for seg in segments:
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        stats["longest_before"] = max(stats["longest_before"], end - start)

        words = _real_words(seg)

        if not words:
            # No word timings to split on. Bound the span so a restored segment
            # cannot hold a cue on screen for minutes.
            if end - start > max_duration:
                fixed = dict(seg)
                fixed["end"] = round(start + max_duration, 3)
                stats["capped"] += 1
                changed = True
                out.append(fixed)
            else:
                out.append(seg)
            continue

        # Clamp to the extent the words actually cover. The words are restored
        # individually and carry true times, so they bound the real speech.
        word_start = float(words[0]["start"])
        word_end = float(words[-1]["end"])
        if word_start > start + 0.05 or word_end < end - 0.05:
            stats["clamped"] += 1
            changed = True
        start = max(start, word_start)
        end = min(end, word_end) if word_end > start else end

        groups: List[List[Dict[str, Any]]] = [[]]
        for index, word in enumerate(words):
            groups[-1].append(word)
            if gap_limit is None or index + 1 >= len(words):
                continue
            try:
                gap = float(words[index + 1]["start"]) - float(word["end"])
            except (TypeError, ValueError, KeyError):
                gap = 0.0
            if gap > gap_limit:
                groups.append([])
        groups = [g for g in groups if g]

        # Whatever --max_gap is set to, no piece may come out longer than the
        # decoder's window: that length can only be restoration damage. Anything
        # still over is cut at its widest internal pause until it fits, so the
        # guarantee does not depend on the gap threshold being tuned.
        groups = _enforce_duration(groups, max_duration)

        if len(groups) > 1:
            stats["split"] += 1
            stats["added"] += len(groups) - 1
            changed = True
        elif not changed:
            out.append(seg)
            continue

        for group in groups:
            piece_start = float(group[0]["start"])
            piece_end = float(group[-1]["end"])
            if piece_end <= piece_start:
                piece_end = piece_start + 0.001
            piece = dict(seg)
            piece["start"] = round(piece_start, 3)
            piece["end"] = round(piece_end, 3)
            piece["text"] = _render(group)
            piece["words"] = _carry_words(group)
            out.append(piece)

    if not changed:
        return segments, stats

    for index, seg in enumerate(out):
        seg["id"] = index
        stats["longest_after"] = max(
            stats["longest_after"], float(seg["end"]) - float(seg["start"])
        )
    return out, stats


def describe_sanitize(stats: Dict[str, Any], before: int, after: int) -> Optional[str]:
    """One log line, or None when nothing was repaired."""
    if not (stats["split"] or stats["clamped"] or stats["capped"]):
        return None
    return (
        f"  [FIX] {before} -> {after} segments | split {stats['split']} spanning a "
        f"removed pause | clamped {stats['clamped']} | capped {stats['capped']} | "
        f"longest {stats['longest_before']:.0f}s -> {stats['longest_after']:.0f}s"
    )


def format_segments(
    segments: List[Dict[str, Any]],
    args: Any,
    language: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Re-split and wrap segments according to the formatting options.

    Returns a new segment list. Segment ids are renumbered because one input
    segment can turn into several cues.
    """
    if not is_enabled(args) or not segments:
        return segments

    max_width = max(1, _int_opt(args, "max_line_width", _WIDTH_SENTINEL))
    max_count = max(1, _int_opt(args, "max_line_count", _COUNT_SENTINEL))
    use_sentence = bool(getattr(args, "sentence", False))

    max_gap = getattr(args, "max_gap", None)
    if max_gap is not None:
        try:
            max_gap = float(max_gap)
        except (TypeError, ValueError):
            max_gap = None
        else:
            if max_gap <= 0:
                max_gap = None

    out: List[Dict[str, Any]] = []
    next_id = 0

    for seg in segments:
        is_cjk = _is_cjk(language, str(seg.get("text", "")))
        words = _words_of(seg, is_cjk)

        if not words:
            # Nothing to work with: keep the segment as-is, only renumbering it.
            kept = dict(seg)
            kept["id"] = next_id
            out.append(kept)
            next_id += 1
            continue

        groups = _split_sentences(words, max_gap) if use_sentence else [words]
        had_real_words = bool(seg.get("words"))

        for group in groups:
            if not group:
                continue
            for cue_lines in _chunk(_wrap(group, max_width), max_count):
                rendered = [_render(line) for line in cue_lines]
                rendered = [line for line in rendered if line]
                if not rendered:
                    continue

                flat = [w for line in cue_lines for w in line]
                start = float(flat[0]["start"])
                end = float(flat[-1]["end"])
                if end <= start:
                    end = start + 0.001

                entry: Dict[str, Any] = {
                    "id": next_id,
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "text": "\n".join(rendered),
                }
                if had_real_words:
                    entry["words"] = _carry_words(flat)

                out.append(entry)
                next_id += 1

    return out or segments
