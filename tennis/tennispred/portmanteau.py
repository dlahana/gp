"""Enumerate every way to glue part of one surname onto part of another.

Surnames are split into rough syllables ("alcaraz" -> al|ca|raz,
"federer" -> fe|de|rer). A candidate is a prefix of one name (whole
syllables) followed by a suffix of the other (whole syllables), in either
order, plus variants that splice at a shared letter. That covers the
canonical blends (sin+caraz) and the stupid ones (alca+rer, na+derer).

Which of these are *funny* is learned from people's votes, in ``reward.py``.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

VOWELS = set("aeiouy")


def clean(name: str) -> str:
    """'Auger-Aliassime' -> 'augeraliassime', 'Davidovich Fokina' -> 'fokina'."""
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    toks = [t for t in re.split(r"[\s]+", s) if t]
    particles = {"de", "del", "van", "von", "da", "di", "le", "la", "der", "den"}
    # Use the last word plus any particles right before it: "Alex de Minaur" ->
    # "deminaur", "Alejandro Davidovich Fokina" -> "fokina" (what fans say).
    i = len(toks) - 1
    while i > 0 and toks[i - 1] in particles:
        i -= 1
    toks = toks[i:]
    return re.sub(r"[^a-z]", "", "".join(toks))


def _is_vowel(word: str, i: int) -> bool:
    c = word[i]
    if c == "y":
        return 0 < i  # word-initial y is a consonant (Yastremska)
    return c in VOWELS


DIGRAPHS = ("tch", "sch", "ch", "sh", "th", "ph", "ck", "ts", "tz", "cz", "sz", "rz", "zh", "dj", "dz", "gn", "qu")


def _consonant_units(cluster: str) -> list[str]:
    """Split a consonant cluster into sounds: 'tch' and 'cz' are one sound each."""
    units, i = [], 0
    while i < len(cluster):
        for d in DIGRAPHS:
            if cluster.startswith(d, i):
                units.append(d)
                i += len(d)
                break
        else:
            units.append(cluster[i])
            i += 1
    return units


def _vowel_starts(word: str) -> list[int]:
    return [i for i in range(1, len(word)) if _is_vowel(word, i) and not _is_vowel(word, i - 1)]


def syllables(word: str) -> list[str]:
    """Approximate syllables by vowel groups.

    Consonants between vowels: one goes to the next syllable (ca.raz), two or
    more split after the first (al.ca, sin.ner).
    """
    if not word:
        return []
    is_v = [_is_vowel(word, i) for i in range(len(word))]
    groups = []  # (start, end) of vowel groups
    i = 0
    while i < len(word):
        if is_v[i]:
            j = i
            while j < len(word) and is_v[j]:
                j += 1
            groups.append((i, j))
            i = j
        else:
            i += 1
    if len(groups) <= 1:
        return [word]
    cuts = []
    for (_, e1), (s2, _) in zip(groups, groups[1:]):
        units = _consonant_units(word[e1:s2])
        if len(units) <= 1:
            cuts.append(e1)                   # ca.raz, mu.cho.va
        else:
            cuts.append(e1 + len(units[0]))   # al.ca, sin.ner
    parts, prev = [], 0
    for c in cuts:
        parts.append(word[prev:c])
        prev = c
    parts.append(word[prev:])
    return [p for p in parts if p]


@dataclass(frozen=True)
class Candidate:
    text: str
    first: str          # name contributing the start
    second: str         # name contributing the end
    head: str           # piece taken from `first`
    tail: str           # piece taken from `second`
    kind: str           # "syllable" | "letter" | "whole"


def _join(head: str, tail: str) -> list[str]:
    out = [head + tail]
    if head and tail and head[-1] == tail[0]:
        out.append(head + tail[1:])   # fed + dal -> fedal
    return out


def enumerate_candidates(name_a: str, name_b: str, min_len: int = 4, max_len: int = 14) -> list[Candidate]:
    a, b = clean(name_a), clean(name_b)
    seen: dict[str, Candidate] = {}

    def add(text: str, first: str, second: str, head: str, tail: str, kind: str) -> None:
        if not (min_len <= len(text) <= max_len) or text in (a, b) or text in seen:
            return
        seen[text] = Candidate(text, first, second, head, tail, kind)

    for x, y in ((a, b), (b, a)):
        sx, sy = syllables(x), syllables(y)
        heads = ["".join(sx[:k]) for k in range(1, len(sx) + 1)]       # incl. the whole name
        tails = ["".join(sy[j:]) for j in range(0, len(sy))]            # incl. the whole name
        # Also cut inside syllables: heads that keep the next consonant
        # (fe -> fed, ca -> car) and tails that start at a vowel (federer -> erer).
        heads += [x[: i + 1] for i in _vowel_starts(x)] + [x[:i] for i in _vowel_starts(x)]
        tails += [y[i:] for i in _vowel_starts(y)]
        heads = list(dict.fromkeys(h for h in heads if len(h) >= 2))
        tails = list(dict.fromkeys(t for t in tails if len(t) >= 2))
        for h in heads:
            for t in tails:
                if h == x and t == y:
                    continue  # just both names glued: allowed as "whole" below
                for text in _join(h, t):
                    kind = "whole" if (h == x or t == y) else "syllable"
                    add(text, x, y, h, t, kind)
        add(x + y, x, y, x, y, "whole")
        # Splice at a shared letter: x up to (and incl.) letter c, y after its c.
        for i in range(1, len(x)):
            for j in range(0, len(y) - 2):
                if x[i] == y[j]:
                    add(x[: i + 1] + y[j + 1:], x, y, x[: i + 1], y[j + 1:], "letter")
    return list(seen.values())
