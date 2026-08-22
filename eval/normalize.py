"""
eval/normalize.py
Text normalisation and character error rate for comparing ASR output against
ARIB captions.

Why there are three levels rather than one
------------------------------------------
ARIB captions are authored for reading, so a raw comparison measures the
difference in output conventions at least as much as it measures recognition.
Reporting one number would hide which of the two moved. The levels are additive,
so the drop from one to the next tells you what that class of difference was
worth:

    asis    whitespace collapsed only. Captions use inline spaces as phrase
            separators and no recogniser emits those, so without this much
            nothing is comparable at all.
    markup  + private use area characters (this is where the DRCS glyphs land),
            symbol ranges, speaker labels and parentheticals. All of these are
            caption notation that no recogniser produces.
    plain   + NFKC and punctuation removal. This is the headline number.

Punctuation is dropped at the last level on purpose. Japanese CER is dominated
by punctuation choices, and anime-whisper is documented to usually omit the
sentence-final 。, so keeping it would charge that model for a formatting habit.

What is deliberately not normalised
-----------------------------------
Long vowel marks stay. They are phonemic, so folding them would hide real
recognition errors.

Non-verbal utterances written as plain text stay, for instance 「えっ…」 and
「ぐっ…」. The captions in this set do contain them, and transcribing them is the
behaviour anime-whisper is being evaluated for, so removing them would erase the
thing under test. Only the parenthesised annotations go, because 「（すすり泣き）」
is a description of a sound rather than a transcription of one and no recogniser
writes it that way.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

# Private use area. The DRCS glyph definitions in these captions are mapped here
# (U+EC00 and friends), and they are never speech.
PUA_RE = re.compile(r"[\uE000-\uF8FF\U000F0000-\U000FFFFD\U00100000-\U0010FFFD]")

# Miscellaneous symbols, dingbats and arrows. Covers the music notes actually
# present (U+266C, not U+266A) plus U+27A1 and the U+269E/U+269F pair.
SYMBOL_RE = re.compile(r"[\u2190-\u21FF\u2600-\u27BF\u2B00-\u2BFF\u3004\u3012\u3013]")

# Music symbols specifically, for flagging song regions rather than removing.
MUSIC_RE = re.compile(r"[\u2669-\u266F]")

# A cue that carries nothing but a wave dash marks an instrumental stretch: the
# captions say "music plays here" without lyrics. 終末ツーリング #12 spends 330.7s
# of its 27:45 this way, in runs of 62 and 66 seconds. MUSIC_RE never sees these
# because there is no note codepoint to find, and the DRCS glyph that renders
# one is mapped into the private use area alongside the line-continuation arrow,
# so the codepoint cannot tell them apart (U+EC00 lands on 62 ordinary cues in
# クレバテス and 74 in LIAR GAME).
INSTRUMENTAL_RE = re.compile(r"^[\u301c\uff5e~\s\u3000]*$")

# Lyrics, when the captions do print them, are wrapped in corner brackets. This
# is checked against every reference in the set: it fires on 73 cues in
# おねがいアイプリ（はじめて）(23.4% of that reference), 40 in アイプリ（情熱）(12.8%),
# 26 in マジルミエ and 17 in ハクメイ (both the ED), and on nothing at all in
# 公女殿下 / LIAR GAME / クレバテス / Summer Pockets / 死亡遊戯 / ニンジャラ / ぷにる.
# Spoken quotation inside a longer line does not match, because the whole cue
# has to be bracketed.
LYRIC_RE = re.compile(r"^(?:\s*「[^「」]*」\s*)+$")

# 「（レロニラ）」 and 「（すすり泣き）」 alike. Non-greedy and length capped so a
# run-on line cannot swallow real text.
PAREN_RE = re.compile(r"[（(][^（()）]{0,40}[)）]")

# 「ぐみ：ミラーパクト」 style labels, only at the start of a line.
SPEAKER_COLON_RE = re.compile(r"^[^\s:：（）()]{1,10}[:：]")

# Narration brackets used by some shows around a whole block of text.
NARRATION_RE = re.compile(r"[＜＞<>]")

PUNCT_CHARS = (
    "、。，．・：；！？!?,.:;"
    "…‥"
    "「」『』〈〉《》【】〔〕［］[]｛｝{}"
    "\u201c\u201d\u2018\u2019\"'"
    "\u301c\uff5e~"
    "\u2010\u2011\u2012\u2013\u2014\u2015-"
    "\u3000 \t\r\n"
)
PUNCT_RE = re.compile("[" + re.escape(PUNCT_CHARS) + "]")

WHITESPACE_RE = re.compile(r"\s+")

LEVELS = ("asis", "markup", "plain")


def collapse_space(text: str) -> str:
    return WHITESPACE_RE.sub("", text)


def strip_markup(text: str) -> str:
    """Remove caption notation that no recogniser emits."""
    text = PUA_RE.sub("", text)
    lines = []
    for line in text.splitlines():
        line = SPEAKER_COLON_RE.sub("", line.strip())
        lines.append(line)
    text = "\n".join(lines)
    # Parentheticals after the colon pass, so 「（名）」 goes here either way.
    text = PAREN_RE.sub("", text)
    text = NARRATION_RE.sub("", text)
    text = SYMBOL_RE.sub("", text)
    return text


def to_plain(text: str) -> str:
    """NFKC and punctuation removal, on top of strip_markup."""
    text = unicodedata.normalize("NFKC", text)
    text = PUNCT_RE.sub("", text)
    return text


def normalize(text: str, level: str = "plain") -> str:
    if level not in LEVELS:
        raise ValueError(f"unknown level {level!r}; choose from {LEVELS}")
    if level == "asis":
        return collapse_space(text)
    text = strip_markup(text)
    if level == "markup":
        return collapse_space(text)
    return collapse_space(to_plain(text))


def has_music_mark(text: str) -> bool:
    """True when the text carries a music symbol, i.e. a sung passage."""
    return bool(MUSIC_RE.search(text))


def is_sung(text: str) -> bool:
    """True when the cue is music rather than speech: instrumental or lyrics.

    Separate from has_music_mark because the note codepoint it looks for is not
    present in this material at all. Three cases, all measured across the set:

        a music symbol            U+2669..U+266F, kept for other sources
        a wave dash on its own    an instrumental stretch, no lyrics printed
        a fully bracketed cue     printed lyrics

    The point of dropping these is that a subtitle for an episode does not need
    the songs, so charging a recogniser for either transcribing them or not is
    measuring the wrong thing. Both directions were observed: おねがいアイプリ has
    23.4% of its reference in lyrics, so a configuration that attempts singing
    scores better there, while 終末ツーリング prints no lyrics at all and the same
    configuration is charged for every invented line. Neither says anything
    about how well dialogue was recognised.

    build_blocks() drops the whole block, and score_cer_whole() takes its
    hypothesis text from the surviving blocks, so the exclusion is symmetric:
    the reference lyrics and whatever the recogniser produced there both leave
    the measurement together.
    """
    stripped = strip_markup(text).strip()
    if not stripped:
        return False          # empty cues are handled by Cue.has_text
    return bool(
        MUSIC_RE.search(text)
        or INSTRUMENTAL_RE.match(stripped)
        or LYRIC_RE.match(stripped)
    )


# ─────────────────────────────────────────────
# Character error rate
# ─────────────────────────────────────────────

def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance, with the shared prefix and suffix trimmed first.

    The trim matters in practice: most of a correctly recognised block matches,
    so it usually removes the bulk of the matrix before any work is done.
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    start = 0
    limit = min(len(a), len(b))
    while start < limit and a[start] == b[start]:
        start += 1
    end = 0
    while end < limit - start and a[len(a) - 1 - end] == b[len(b) - 1 - end]:
        end += 1
    a = a[start:len(a) - end]
    b = b[start:len(b) - end]

    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):
        a, b = b, a

    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        append = current.append
        for j, cb in enumerate(b, 1):
            append(min(previous[j] + 1, current[j - 1] + 1,
                       previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def _banded_distance(a: str, b: str, band: int) -> int:
    """Levenshtein restricted to a diagonal band of the given half width.

    Exact whenever the optimal path stays inside the band, which it does when
    the two strings are broadly aligned. Full DP on a whole episode is not
    viable in Python: 5000 by 5000 characters is 25M cells per pair, and the
    comparison runs over several files, configurations and normalisation levels.
    """
    n, m = len(a), len(b)
    infinity = n + m + 1
    previous = [infinity] * (m + 1)
    for j in range(0, min(m, band) + 1):
        previous[j] = j

    for i in range(1, n + 1):
        current = [infinity] * (m + 1)
        low = max(1, i - band)
        high = min(m, i + band)
        if low == 1:
            current[0] = i
        row_a = a[i - 1]
        for j in range(low, high + 1):
            cost = 0 if row_a == b[j - 1] else 1
            best = previous[j] + 1
            insertion = current[j - 1] + 1
            if insertion < best:
                best = insertion
            substitution = previous[j - 1] + cost
            if substitution < best:
                best = substitution
            current[j] = best
        previous = current
    return previous[m]


def edit_distance_large(
    a: str,
    b: str,
    band: int = 256,
    max_band: int = 32768,
) -> Tuple[int, bool]:
    """Edit distance for long strings, widening the band until it is not binding.

    Returns (distance, exact). A result below the band width proves the optimal
    path never touched the band's edge, so the value is exact. Otherwise the
    band doubles and the computation repeats, up to max_band, after which the
    distance is only an upper bound and exact is False.

    The cap is applied by clamping the width, not by testing the width after
    doubling. Testing afterwards made the cap depend on where the doubling
    happened to land: on a 5h22m recording two configurations differing by 282
    characters in length took different paths, one stopping at band 8416 with an
    upper bound and the other going on to 15704 and landing exact. Their CERs
    then differed by 1.9pt with no way to tell from the report that only one of
    them was a real number.
    """
    if a == b:
        return 0, True
    if not a:
        return len(b), True
    if not b:
        return len(a), True

    start = 0
    limit = min(len(a), len(b))
    while start < limit and a[start] == b[start]:
        start += 1
    end = 0
    while end < limit - start and a[len(a) - 1 - end] == b[len(b) - 1 - end]:
        end += 1
    a = a[start:len(a) - end]
    b = b[start:len(b) - end]
    if not a:
        return len(b), True
    if not b:
        return len(a), True

    # A band at least as wide as the longer string spans the whole matrix, so
    # widening past that buys nothing and capping below it is what makes a
    # result approximate.
    full = max(len(a), len(b))
    ceiling = min(max_band, full)
    width = min(max(band, abs(len(a) - len(b)) + 1), ceiling)
    while True:
        distance = _banded_distance(a, b, width)
        # Under the band width, the optimal path never reached the band's edge.
        # At or above full width there is no edge to reach. Either way exact.
        if distance < width or width >= full:
            return distance, True
        if width >= ceiling:
            return distance, False
        width = min(width * 2, ceiling)


# ─────────────────────────────────────────────
# Longest common subsequence
# ─────────────────────────────────────────────
#
# Edit distance alone cannot say whether a CER of 24% is the recogniser missing
# speech or getting it wrong, because a deletion and a substitution cost the
# same. Splitting it needs the number of reference characters actually recovered,
# which is the LCS:
#
#     coverage  = LCS / reference length   how much of the reference came back
#     precision = LCS / hypothesis length  how much of the output is in the reference
#
# Low coverage with high precision is missed speech; the reverse is invention.
# A full D/I/S breakdown would need a traceback over the whole matrix, which does
# not fit in memory at 38k by 38k, and the two ratios answer the question anyway.


def lcs(a: str, b: str) -> int:
    """Exact LCS length. For short strings and for checking the banded version."""
    a, b, fixed = _strip_common_affixes(a, b)
    if not a or not b:
        return fixed
    if len(a) < len(b):
        a, b = b, a

    previous = [0] * (len(b) + 1)
    for ca in a:
        current = [0]
        append = current.append
        for j, cb in enumerate(b, 1):
            if ca == cb:
                append(previous[j - 1] + 1)
            else:
                up, left = previous[j], current[j - 1]
                append(up if up >= left else left)
        previous = current
    return fixed + previous[-1]


def _strip_common_affixes(a: str, b: str) -> Tuple[str, str, int]:
    """Peel the shared prefix and suffix off, returning how many were peeled.

    Safe for both metrics: a shared leading character can always be matched in
    some optimal alignment, so LCS(a, b) == 1 + LCS(a[1:], b[1:]) when the first
    characters agree.
    """
    if not a or not b:
        return a, b, 0
    start = 0
    limit = min(len(a), len(b))
    while start < limit and a[start] == b[start]:
        start += 1
    end = 0
    while end < limit - start and a[len(a) - 1 - end] == b[len(b) - 1 - end]:
        end += 1
    return a[start:len(a) - end], b[start:len(b) - end], start + end


def _banded_lcs(a: str, b: str, band: int) -> int:
    """LCS restricted to a diagonal band, so a lower bound in general.

    Unreachable cells hold -1 rather than 0, otherwise a cell outside the band
    would look like "no matches yet" and let the path re-enter the band for free,
    which overstates the result instead of understating it.
    """
    n, m = len(a), len(b)
    unreachable = -1
    previous = [unreachable] * (m + 1)
    for j in range(0, min(m, band) + 1):
        previous[j] = 0

    for i in range(1, n + 1):
        current = [unreachable] * (m + 1)
        low = max(1, i - band)
        high = min(m, i + band)
        if low == 1:
            current[0] = 0
        row_a = a[i - 1]
        for j in range(low, high + 1):
            best = unreachable
            if row_a == b[j - 1]:
                diagonal = previous[j - 1]
                if diagonal >= 0:
                    best = diagonal + 1
            up = previous[j]
            if up > best:
                best = up
            left = current[j - 1]
            if left > best:
                best = left
            current[j] = best
        previous = current
    return previous[m]


def lcs_large(
    a: str,
    b: str,
    band: int = 256,
    max_band: int = 32768,
) -> Tuple[int, bool]:
    """LCS for long strings, widening the band until it is not binding.

    Returns (matches, exact). The exactness test mirrors edit_distance_large but
    uses the insertion/deletion distance implied by the result: an alignment with
    that many single-sided steps cannot stray further than that from the
    diagonal, so a value below the band width proves the band did not bind.
    """
    a, b, fixed = _strip_common_affixes(a, b)
    if not a or not b:
        return fixed, True

    full = max(len(a), len(b))
    ceiling = min(max_band, full)
    width = min(max(band, abs(len(a) - len(b)) + 1), ceiling)
    while True:
        matches = _banded_lcs(a, b, width)
        indel = len(a) + len(b) - 2 * matches
        if indel < width or width >= full:
            return fixed + matches, True
        if width >= ceiling:
            return fixed + matches, False
        width = min(width * 2, ceiling)


@dataclass
class CerResult:
    ref_chars: int = 0
    hyp_chars: int = 0
    edits: int = 0
    blocks: int = 0
    empty_ref_blocks: int = 0
    # Pairs whose banded distance hit the band cap. Their edit count is an upper
    # bound, so the CER is too. Non-zero means the number must not be compared
    # against a configuration that scored exactly.
    approximate: int = 0
    # Reference characters recovered, for the coverage/precision split. Only
    # filled when add() is called with with_lcs.
    matches: int = 0
    lcs_pairs: int = 0
    lcs_approximate: int = 0

    @property
    def cer(self) -> float:
        return (self.edits / self.ref_chars) if self.ref_chars else 0.0

    @property
    def exact(self) -> bool:
        return self.approximate == 0

    @property
    def has_lcs(self) -> bool:
        return self.lcs_pairs > 0

    @property
    def coverage(self) -> float:
        """Share of the reference recovered. Low means missed speech."""
        return (self.matches / self.ref_chars) if self.ref_chars else 0.0

    @property
    def precision(self) -> float:
        """Share of the output that is in the reference. Low means invention."""
        return (self.matches / self.hyp_chars) if self.hyp_chars else 0.0

    @property
    def lcs_exact(self) -> bool:
        return self.lcs_approximate == 0

    @property
    def length_ratio(self) -> float:
        """Hypothesis length over reference length.

        Above 1 means the recogniser is producing more text than the captions
        carry, which is the expected direction for a model that transcribes
        non-verbal speech against captions that leave it out.
        """
        return (self.hyp_chars / self.ref_chars) if self.ref_chars else 0.0

    def add(
        self,
        reference: str,
        hypothesis: str,
        large: bool = False,
        with_lcs: bool = False,
    ) -> None:
        self.blocks += 1
        if not reference:
            self.empty_ref_blocks += 1
            self.hyp_chars += len(hypothesis)
            return
        self.ref_chars += len(reference)
        self.hyp_chars += len(hypothesis)
        if large:
            distance, exact = edit_distance_large(reference, hypothesis)
            if not exact:
                self.approximate += 1
        else:
            distance = edit_distance(reference, hypothesis)
        self.edits += distance

        if with_lcs:
            self.lcs_pairs += 1
            if large:
                matches, lcs_ok = lcs_large(reference, hypothesis)
                if not lcs_ok:
                    self.lcs_approximate += 1
            else:
                matches = lcs(reference, hypothesis)
            self.matches += matches


def score_pairs(
    pairs: Iterable[Tuple[str, str]],
    level: str = "plain",
    large: bool = False,
    with_lcs: bool = False,
) -> CerResult:
    """CER over (reference, hypothesis) text pairs, normalised at one level.

    Set large for whole-episode strings, which switches to the banded distance.
    Set with_lcs to also get the coverage/precision split, which costs a second
    pass of the same order.
    """
    result = CerResult()
    for reference, hypothesis in pairs:
        result.add(
            normalize(reference, level),
            normalize(hypothesis, level),
            large=large,
            with_lcs=with_lcs,
        )
    return result


def score_all_levels(
    pairs: Sequence[Tuple[str, str]],
) -> Dict[str, CerResult]:
    """CER at every level, so the drop between levels is visible."""
    return {level: score_pairs(pairs, level) for level in LEVELS}
