"""``PolicyLens``: one object that indexes a corpus and answers questions about it."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np

from .acronyms import extract_acronyms
from .corpus import Document, load_directory
from .references import ReferenceExtractor, _normalize, build_citation_graph, influence
from .search import HybridIndex
from .summarize import textrank


@dataclass
class SearchHit:
    doc_id: str
    title: str
    passage: str
    score: float

    def __str__(self) -> str:
        snippet = self.passage if len(self.passage) < 300 else self.passage[:297] + "..."
        return f"[{self.doc_id}] {self.title}\n    {snippet}"


class PolicyLens:
    """Search, summarise and connect a collection of policy documents.

    >>> lens = PolicyLens.from_directory("data/sample_corpus")
    >>> lens.search("how fast must a security incident be reported?")[0].doc_id
    'Instruction 1000.03'
    """

    def __init__(self, docs: list[Document], n_components: int = 150,
                 extractor: ReferenceExtractor | None = None):
        if not docs:
            raise ValueError("Need at least one document")
        self.docs = {d.doc_id: d for d in docs}
        self.passages: list[tuple[str, str]] = [(d.doc_id, p) for d in docs for p in d.paragraphs]
        self.index = HybridIndex([f"{self.docs[i].title}. {p}" for i, p in self.passages], n_components)
        self.graph: nx.DiGraph = build_citation_graph(docs, extractor)
        self.glossary = extract_acronyms([d.text for d in docs])
        # document vectors = mean of passage embeddings
        emb = self.index.lsa.embeddings
        ids = np.array([i for i, _ in self.passages])
        self._doc_ids = list(self.docs)
        vecs = np.stack([emb[ids == d].mean(0) for d in self._doc_ids])
        self._doc_vecs = vecs / np.clip(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12, None)

    @classmethod
    def from_directory(cls, folder: str | Path, **kwargs) -> "PolicyLens":
        return cls(load_directory(folder), **kwargs)

    # ---------------------------------------------------------------- search
    def search(self, query: str, top_k: int = 5, mode: str = "hybrid", one_per_doc: bool = True) -> list[SearchHit]:
        hits, seen = [], set()
        for i, score in self.index.search(query, top_k=top_k * 10, mode=mode):
            doc_id, passage = self.passages[i]
            if one_per_doc and doc_id in seen:
                continue
            seen.add(doc_id)
            hits.append(SearchHit(doc_id, self.docs[doc_id].title, passage, score))
            if len(hits) == top_k:
                break
        return hits

    def answer(self, question: str, n_sentences: int = 2, top_k: int = 3) -> dict:
        """Extractive answer: query-focused summary of the best matching passages."""
        hits = self.search(question, top_k=top_k)
        context = "\n\n".join(h.passage for h in hits)
        return {"question": question,
                "answer": " ".join(textrank(context, n_sentences=n_sentences, query=question, query_weight=0.9)),
                "sources": [h.doc_id for h in hits]}

    # ------------------------------------------------------------- summarise
    def summarize(self, doc_id: str, n_sentences: int = 3, query: str | None = None) -> str:
        return " ".join(textrank(self._doc(doc_id).body, n_sentences=n_sentences, query=query))

    # ------------------------------------------------------------- connect
    def similar(self, doc_id: str, top_k: int = 5) -> list[tuple[str, float]]:
        i = self._doc_ids.index(self._doc(doc_id).doc_id)
        sims = self._doc_vecs @ self._doc_vecs[i]
        order = [j for j in np.argsort(-sims) if j != i][:top_k]
        return [(self._doc_ids[j], float(sims[j])) for j in order]

    def references(self, doc_id: str) -> list[str]:
        """Everything ``doc_id`` cites (internal and external)."""
        return sorted(self.graph.successors(_normalize(self._doc(doc_id).doc_id)))

    def cited_by(self, ref: str) -> list[str]:
        """Corpus documents that cite ``ref`` (a doc id or an external authority)."""
        ref = _normalize(ref)
        return sorted(self.graph.predecessors(ref)) if ref in self.graph else []

    def most_influential(self, top_n: int = 10, in_corpus_only: bool = False):
        return influence(self.graph, top_n=top_n, in_corpus_only=in_corpus_only)

    def acronym(self, acr: str) -> str | None:
        entry = self.glossary.get(acr)
        return entry["expansion"] if entry else None

    # ---------------------------------------------------------- persistence
    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> "PolicyLens":
        """Load an index you created yourself. Only unpickle files you trust."""
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, PolicyLens):
            raise TypeError("Not a PolicyLens index")
        return obj

    def _doc(self, doc_id: str) -> Document:
        if doc_id in self.docs:
            return self.docs[doc_id]
        norm = {_normalize(k): v for k, v in self.docs.items()}
        if _normalize(doc_id) in norm:
            return norm[_normalize(doc_id)]
        raise KeyError(f"Unknown document '{doc_id}'")
