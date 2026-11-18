"""Find references between documents and build a citation graph."""

from __future__ import annotations

import re
from collections import Counter

import networkx as nx

# Default patterns cover common U.S. federal citation styles plus the
# identifiers used by the bundled sample corpus. Pass your own to
# ``ReferenceExtractor`` for other numbering schemes.
DEFAULT_PATTERNS: dict[str, str] = {
    "issuance": r"\b(?:[A-Z]{2,6}-)?(?:Directive|Instruction|Manual|Memorandum|Policy|Order|Notice)\s+\d{3,5}(?:\.\d{1,3}){1,2}\b",
    "us_code": r"\b\d{1,2}\s+U\.S\.C\.\s+(?:§+\s*)?\d+[a-z]?(?:\([a-z0-9]+\))*",
    "cfr": r"\b\d{1,2}\s+C\.?F\.?R\.?\s+(?:[Pp]art\s+)?\d+(?:\.\d+)?",
    "public_law": r"\b(?:Public Law|Pub\.\s?L\.)\s+(?:No\.\s+)?\d{2,3}-\d{1,3}\b",
    "omb_memo": r"\bOMB\s+(?:Memorandum\s+)?M-\d{2}-\d{2}\b",
    "executive_order": r"\b(?:Executive Order|E\.O\.|EO)\s+\d{5}\b",
}


def _normalize(ref: str) -> str:
    ref = re.sub(r"\s+", " ", ref.strip())
    ref = re.sub(r"§+\s*", "", ref)
    ref = re.sub(r"^(?:E\.O\.|EO\b)", "Executive Order", ref)
    ref = re.sub(r"^Pub\.\s?L\.", "Public Law", ref)
    ref = re.sub(r"\bOMB Memorandum ", "OMB ", ref)
    return ref


class ReferenceExtractor:
    def __init__(self, patterns: dict[str, str] | None = None):
        self.patterns = {k: re.compile(v) for k, v in (patterns or DEFAULT_PATTERNS).items()}

    def extract(self, text: str) -> Counter:
        """Count normalised references found in ``text``."""
        found: Counter = Counter()
        for rx in self.patterns.values():
            for m in rx.finditer(text):
                found[_normalize(m.group(0))] += 1
        return found


def build_citation_graph(docs, extractor: ReferenceExtractor | None = None) -> nx.DiGraph:
    """Directed graph ``citing → cited``.

    Nodes for documents in the corpus have ``in_corpus=True``; references to
    outside authorities (laws, executive orders) appear as external nodes so
    you can see which statutes the corpus leans on most.
    """
    extractor = extractor or ReferenceExtractor()
    g = nx.DiGraph()
    ids = {_normalize(d.doc_id): d for d in docs}
    for d in docs:
        g.add_node(_normalize(d.doc_id), title=d.title, in_corpus=True)
    for d in docs:
        src = _normalize(d.doc_id)
        for ref, count in extractor.extract(d.text).items():
            if ref == src:
                continue
            if ref not in g:
                g.add_node(ref, title=ids[ref].title if ref in ids else ref, in_corpus=ref in ids)
            g.add_edge(src, ref, weight=count)
    return g


def influence(g: nx.DiGraph, top_n: int = 10, in_corpus_only: bool = False) -> list[tuple[str, float, int]]:
    """Rank nodes by PageRank on the citation graph: ``(node, pagerank, times_cited)``."""
    pr = nx.pagerank(g, weight="weight") if len(g) else {}
    rows = [(n, pr[n], g.in_degree(n)) for n in g
            if not in_corpus_only or g.nodes[n].get("in_corpus")]
    rows.sort(key=lambda r: (-r[1], -r[2]))
    return rows[:top_n]
