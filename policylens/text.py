"""Small, dependency-free text utilities."""

from __future__ import annotations

import re

STOPWORDS = frozenset("""
a about above after again against all also am an and any are as at be because been before being below
between both but by can could did do does doing down during each either few for from further had has
have having he her here hers herself him himself his how i if in into is it its itself just may me might
more most must my myself no nor not of off on once only or other our ours ourselves out over own same
shall she should so some such than that the their theirs them themselves then there these they this those
through to too under until up upon very was we were what when where which while who whom why will with
within without would you your yours yourself yourselves per via shall herein thereof
""".split())

_TOKEN = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])")
_ABBREV = re.compile(r"(?:\b(?:U\.S\.C|U\.S|C\.F\.R|E\.O|Pub|e\.g|i\.e|No|Sec|Vol|Fig|Dr|Mr|Ms|[A-Z])|§)\.$")


_SUFFIXES = ("ization", "ations", "ation", "ments", "ment", "ness", "ings", "ing", "ies", "ied",
             "ers", "er", "ed", "es", "ly", "s")
_HEADING = re.compile(r"^\s*(?:[0-9]+(?:\.[0-9]+)*\.?\s*)?[A-Z][A-Z0-9 ,&/()-]{2,}$")


def stem(word: str) -> str:
    """Light suffix-stripping stemmer (report/reported/reporting → report)."""
    if len(word) <= 4 or any(c.isdigit() for c in word):
        return word
    for suf in _SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 4:
            base = word[: -len(suf)]
            if suf == "ies" or suf == "ied":
                base += "y"
            if len(base) > 4 and base[-1] == base[-2] and base[-1] not in "ls":
                base = base[:-1]  # planned → plan
            return base
    return word


def tokenize(text: str, remove_stopwords: bool = True, stemming: bool = True) -> list[str]:
    """Lower-case word tokens; keeps identifiers like ``8510.01`` and ``pl-117`` intact."""
    toks = _TOKEN.findall(text.lower())
    if remove_stopwords:
        toks = [t for t in toks if t not in STOPWORDS and len(t) > 1]
    if stemming:
        toks = [stem(t) for t in toks]
    return toks


def clean_text(text: str) -> str:
    """Drop front-matter (``ID:`` lines) and ALL-CAPS section headings."""
    keep = []
    for line in text.splitlines():
        if line.startswith("ID:") or (_HEADING.match(line) and len(line.split()) <= 6):
            keep.append("")
        else:
            keep.append(line)
    return "\n".join(keep)


def sentence_windows(text: str, size: int = 3, stride: int = 2) -> list[str]:
    """Overlapping windows of ``size`` sentences within each paragraph."""
    out = []
    for para in split_paragraphs(text, min_chars=1):
        sents = split_sentences(para)
        if len(sents) <= size:
            out.append(" ".join(sents))
            continue
        for i in range(0, len(sents) - size + stride, stride):
            out.append(" ".join(sents[i:i + size]))
    return [o for o in out if o]


def split_sentences(text: str) -> list[str]:
    """Rule-based sentence splitter that avoids breaking on common abbreviations."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts, buf = [], ""
    for piece in _SENT_SPLIT.split(text):
        buf = f"{buf} {piece}".strip() if buf else piece
        if not _ABBREV.search(buf):
            parts.append(buf)
            buf = ""
    if buf:
        parts.append(buf)
    return [p for p in parts if len(p) > 1]


def split_paragraphs(text: str, min_chars: int = 40, max_chars: int = 1200) -> list[str]:
    """Split on blank lines; merge tiny fragments and cut very long blocks by sentence."""
    blocks = [re.sub(r"\s+", " ", b).strip() for b in re.split(r"\n\s*\n", text)]
    blocks = [b for b in blocks if b]
    merged: list[str] = []
    for b in blocks:
        if merged and len(merged[-1]) < min_chars:
            merged[-1] = f"{merged[-1]} {b}"
        else:
            merged.append(b)
    out: list[str] = []
    for b in merged:
        if len(b) <= max_chars:
            out.append(b)
            continue
        chunk = ""
        for s in split_sentences(b):
            if chunk and len(chunk) + len(s) > max_chars:
                out.append(chunk)
                chunk = s
            else:
                chunk = f"{chunk} {s}".strip()
        if chunk:
            out.append(chunk)
    return out
