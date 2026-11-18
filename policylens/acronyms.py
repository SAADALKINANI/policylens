"""Build an acronym glossary from definitions such as "Chief Data Officer (CDO)"."""

from __future__ import annotations

import re
from collections import Counter, defaultdict

_DEF = re.compile(r"((?:[A-Z][\w&'-]*\s+(?:(?:of|and|for|the|on|in|to)\s+)*){1,9})\(([A-Z][A-Za-z&]{1,9})\)")
_SMALL = {"of", "and", "for", "the", "on", "in", "to"}


def _initials(words: list[str], skip_small: bool) -> list[str]:
    out = []
    for w in words:
        if skip_small and w.lower() in _SMALL:
            continue
        # an all-caps word such as "AI" contributes every letter
        out.extend(list(w) if w.isupper() and len(w) <= 4 else [w[0].upper()])
    return out


def _matches(words: list[str], acronym: str) -> list[str] | None:
    """Shortest suffix of ``words`` whose initials spell ``acronym``."""
    letters = [c for c in acronym.upper() if c.isalpha()]
    for start in range(len(words) - 1, -1, -1):
        cand = words[start:]
        if _initials(cand, skip_small=True) == letters:
            while cand and cand[0].lower() in _SMALL:
                cand = cand[1:]
            return cand
        # allow small words to contribute ("Department of Defense" → DoD)
        if _initials(cand, skip_small=False) == letters:
            return cand
    return None


def extract_acronyms(texts: list[str]) -> dict[str, dict]:
    """Return ``{ACRONYM: {"expansion": str, "count": int, "variants": {...}}}``."""
    found: dict[str, Counter] = defaultdict(Counter)
    for text in texts:
        for m in _DEF.finditer(text):
            words = m.group(1).split()
            acr = m.group(2)
            hit = _matches(words, acr)
            if hit:
                found[acr][" ".join(hit)] += 1
    out = {}
    for acr, variants in sorted(found.items()):
        expansion, count = variants.most_common(1)[0]
        out[acr] = {"expansion": expansion, "count": sum(variants.values()), "variants": dict(variants)}
    return out
