#!/usr/bin/env python3
"""
filler_filter.py
Drop the stock closing phrases Whisper emits over non-speech audio.

Why this exists
    Whisper was trained on a lot of internet video, so when it is handed audio
    that is not speech it often emits the phrase that ends one:
    「ご視聴ありがとうございました」. On the evaluation set this happens 43 times
    across nine 24 minute recordings, and removing those segments ahead of time
    takes the total CER from 16.1% to 15.5% (MEASUREMENTS 測定結果 #20).

    That 0.6 points understates the problem, which is why the earlier decision
    to skip this filter was reversed. The phrases land on OP, ED, credits and
    trailers, where the ARIB reference is empty, so CER barely charges for them.
    A line that appears over the ED is invisible to the metric and completely
    visible to whoever is watching.

    loop_filter cannot see these: they are neither a repetition inside one
    segment nor over 30 seconds. faster-whisper's own guards do not fire either.
    `no_speech_threshold` only discards a segment when the no-speech probability
    is high *and* `avg_logprob` is below -1.0, and these phrases are high
    frequency training data, so Whisper emits them with ordinary confidence.

Two measured dead ends, so they are not retried here
    Confidence does not separate them. Mean word probability of matched
    segments sits at p5 0.720 against the corpus p5 0.582 -- the filler is
    *more* confident at the bottom of the distribution. A gate at 0.7 catches
    20 of 645 while putting 8,590 real segments below the same line.

    Loudness does not either. Matched segments sit at a median -28.7 dBFS
    against -28.2 for everything else, and the loudest is -19.6. These are not
    hallucinations over silence, they are hallucinations over audible
    non-speech: screams, sound effects, the attack of a music cue.

    Both measurements are in MEASUREMENTS 測定結果 #20. The remaining untouched
    signal is `no_speech_prob`, which is why this module reports it on the log
    line rather than acting on it.

What is deliberately conservative about the rule
    The phrase has to account for the whole segment. Matching a substring and
    then dropping the segment would take real dialogue with it: measured over
    93,646 segments, 645 matched a phrase and 3 of those carried extra text
    beyond it. Requiring the whole segment keeps 642 of 645 and removes that
    failure mode by construction rather than by luck.

    The list holds only phrases measured to catch something. A bare
    「チャンネル登録」 caught nothing in 93,646 segments while being the single
    most dangerous entry, because a character who streams can say it as real
    dialogue.

    None of the seven phrases originally tested appears in any of the 18
    references. The corpus does get close, though: 「ありがとうございました」 alone
    appears 15 times as real dialogue (「手伝ってくれてありがとうございました」), and
    「登録」 3 times (「納品登録を」). That is the reason entries stay at full-phrase
    length and the match is anchored to the whole segment. Shortening any entry
    towards those fragments would start deleting speech.

Scope and the residual risk
    Fifteen anime recordings produced no false positive, which is a statement
    about this corpus and not about the rule. The risk is genre dependent and
    peaks exactly where the phrase is really spoken: a variety show host closes
    with 「ご視聴ありがとうございました」, and Amatsukaze users record variety and
    drama too. Hence `--filler_filter false`, and hence the log line: detection
    always runs and always reports, so a false positive is discoverable in a
    log the user already sends with a bug report, whether or not dropping is on.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

# The phrase list was measured against eval/normalize.py's "plain" level, and
# loop_filter.normalize() folds the same character class (NFKC, punctuation and
# whitespace removed). Sharing it keeps the two from drifting apart, because
# both want the same thing: text with the decoration taken off.
from loop_filter import normalize

# Phrases that have to account for the whole segment. Ordered so that the entry
# doing almost all the work comes first; the reported reason names the entry
# that matched, so order shows up in logs.
#
# Counts are over 93,646 segments from 40 configurations (_eval/_filler_risk.txt):
#
#     635  ご視聴ありがとうございました
#       7  チャンネル登録をお願いいたします
#       2  ご視聴ありがとうございます
#
# Three entries from the original list were removed for catching nothing at all
# while carrying the real risk: 「チャンネル登録」 on its own, 「高評価とチャンネル登録」
# and 「ご覧いただきありがとうございました」.
#
# 「最後までご視聴」 was removed too, which goes one step past the table in
# MEASUREMENTS #20. Its single hit was a partial match
# (「最後までご視聴してくださって嬉しかったら高評価ボタンを押してね」), so anchoring to the
# whole segment already makes it catch nothing, and what is left is a sentence
# fragment -- the shape most likely to match a piece of real dialogue.
FILLER_JA = (
    "ご視聴ありがとうございました",
    "ご視聴ありがとうございます",
    "チャンネル登録をお願いいたします",
)

# The same failure class in Korean, found in production output rather than in
# the evaluation set: when language detection picks the wrong language on
# Japanese material, the stock phrases come out in that language instead.
# Observed in sample/errdata (ニワトリ・ファイター 第三羽, lang=ko 36.45%), where
# 「다음 영상에서 만나요」 ("see you in the next video") and the subscribe request
# below appear as whole segments over the OP and over non-speech.
#
# These cannot collide with Japanese dialogue: the scripts do not overlap. They
# are listed separately so that a future language brings its own group rather
# than growing one flat list.
FILLER_KO = (
    "영상편집및자막이도움이되셨다면구독좋아요댓글부탁드립니다",
    "다음영상에서만나요",
    "시청해주셔서감사합니다",
    "구독과좋아요부탁드립니다",
)

FILLER = FILLER_JA + FILLER_KO

# How much text may sit alongside the phrase and still count as "the whole
# segment". Two characters absorbs a stray quote or an interjection the fold
# does not remove, without letting a clause in. 642 of the 645 measured matches
# fall inside this; the 3 that do not are the ones this margin is meant to
# exclude.
WHOLE_SEGMENT_SLACK = 2


def _folded_phrases() -> Tuple[Tuple[str, str], ...]:
    """(folded, original) for each phrase, so reasons can quote the original."""
    return tuple((normalize(p), p) for p in FILLER)


_PHRASES = _folded_phrases()


def match(text: str) -> Optional[Tuple[str, bool]]:
    """(phrase, is_whole_segment) for the first phrase found, or None.

    Detection is separate from the decision to drop on purpose. A partial match
    is still returned, so it can be reported without being acted on: that is the
    only way the 3-in-645 case shows up in a log instead of being invisible.
    """
    folded = normalize(text)
    if not folded:
        return None
    for phrase, original in _PHRASES:
        if phrase in folded:
            whole = len(folded) <= len(phrase) + WHOLE_SEGMENT_SLACK
            return original, whole
    return None


def is_filler(text: str) -> bool:
    """True when the segment is nothing but a stock phrase."""
    found = match(text)
    return bool(found and found[1])


def filter_segments(
    segments: List[Dict[str, Any]],
    drop: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Detect stock phrases, and remove them when `drop` is set.

    Returns (segments, stats). Detection runs either way, so `stats` is
    populated even with drop=False and the caller can log what it would have
    removed. The same list object comes back when nothing was removed, so ids
    stay untouched in the common case -- matching loop_filter.filter_segments.
    """
    stats: Dict[str, Any] = {
        "detected": 0,
        "dropped": 0,
        "chars": 0,
        "partial": 0,
        "hits": [],
    }
    if not segments:
        return segments, stats

    kept: List[Dict[str, Any]] = []
    for segment in segments:
        text = str(segment.get("text") or "")
        found = match(text)
        if found is None:
            kept.append(segment)
            continue

        phrase, whole = found
        stats["detected"] += 1
        removing = bool(drop and whole)
        if not whole:
            stats["partial"] += 1
        if removing:
            stats["dropped"] += 1
            stats["chars"] += len(normalize(text))

        stats["hits"].append({
            "start": float(segment.get("start") or 0.0),
            "end": float(segment.get("end") or 0.0),
            "action": "drop" if removing else "keep",
            # "partial" is why it was kept despite matching; with drop off the
            # reason is still the match itself, and the action says what happened.
            "reason": "exact" if whole else "partial",
            "phrase": phrase,
            "text": text.strip(),
            "no_speech_prob": segment.get("no_speech_prob"),
        })

        if not removing:
            kept.append(segment)

    if not stats["dropped"]:
        return segments, stats
    return kept, stats


def describe(
    stats: Dict[str, Any],
    enabled: bool = True,
) -> List[str]:
    """The [FILLER] block, or an empty list when nothing was detected.

    Silent on material that has none, the way describe_sanitize() only speaks up
    for a file it actually repaired. Every hit is listed rather than a sample of
    three: the whole point is that a user can check the timestamps against the
    recording, and there are single digits of these per file.
    """
    if not stats.get("detected"):
        return []

    head = f"  [FILLER] {stats['detected']} detected, {stats['dropped']} dropped"
    if not enabled:
        head += " (--filler_filter false: detection only)"
    if stats.get("partial"):
        head += f", {stats['partial']} partial (kept)"
    lines = [head]

    for hit in stats["hits"]:
        stamp = time.strftime("%H:%M:%S", time.gmtime(max(hit["start"], 0.0)))
        span = hit["end"] - hit["start"]
        prob = hit.get("no_speech_prob")
        # Reported, never acted on. Confidence was measured not to separate
        # these from real speech (MEASUREMENTS #20), and no_speech_prob is the
        # one signal left unmeasured -- so production logs become the data.
        tail = "" if prob is None else f" ns={float(prob):.2f}"
        lines.append(
            f"           {stamp} {span:5.1f}s {hit['action']:4} "
            f"{hit['reason']}:{tail} {hit['text'][:48]}"
        )
    return lines
