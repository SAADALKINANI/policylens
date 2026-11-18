"""Hybrid passage retrieval: BM25 (keywords) + LSA (concepts), fused with RRF.

* **BM25** rewards exact term matches, which matters for policy numbers and
  jargon ("8510.01", "records schedule").
* **LSA** (TF-IDF → truncated SVD) maps passages into a concept space so a
  query like "remote work" also finds passages about "telework".
* **Reciprocal Rank Fusion** combines the two ranked lists without having to
  calibrate their very different score scales.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .text import tokenize


class BM25:
    def __init__(self, corpus_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_len = np.array([len(d) for d in corpus_tokens], dtype=float)
        self.avgdl = float(self.doc_len.mean()) if len(self.doc_len) else 0.0
        self.tf = [Counter(d) for d in corpus_tokens]
        df = Counter(t for d in corpus_tokens for t in set(d))
        n = len(corpus_tokens)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        # inverted index for speed
        self.postings: dict[str, list[int]] = {}
        for i, d in enumerate(self.tf):
            for t in d:
                self.postings.setdefault(t, []).append(i)

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        out = np.zeros(len(self.tf))
        for t in set(query_tokens):
            if t not in self.idf:
                continue
            idf = self.idf[t]
            for i in self.postings[t]:
                f = self.tf[i][t]
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                out[i] += idf * f * (self.k1 + 1) / denom
        return out


class LSA:
    def __init__(self, texts: list[str], n_components: int = 150, seed: int = 0):
        self.vectorizer = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None,
                                          ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        X = self.vectorizer.fit_transform(texts)
        k = max(1, min(n_components, X.shape[0] - 1, X.shape[1] - 1))
        self.svd = TruncatedSVD(n_components=k, random_state=seed)
        self.embeddings = normalize(self.svd.fit_transform(X))

    def embed(self, texts: list[str]) -> np.ndarray:
        return normalize(self.svd.transform(self.vectorizer.transform(texts)))

    def scores(self, query: str) -> np.ndarray:
        return self.embeddings @ self.embed([query])[0]


def reciprocal_rank_fusion(score_lists: list[np.ndarray], k: int = 60,
                           weights: list[float] | None = None) -> np.ndarray:
    """RRF (Cormack et al., 2009): ``sum_i w_i / (k + rank_i)``."""
    weights = weights or [1.0] * len(score_lists)
    fused = np.zeros(len(score_lists[0]))
    for w, s in zip(weights, score_lists):
        ranks = np.empty(len(s))
        ranks[np.argsort(-s)] = np.arange(1, len(s) + 1)
        fused += w / (k + ranks)
    return fused


class HybridIndex:
    """Index of passages supporting ``bm25``, ``semantic`` and ``hybrid`` search."""

    def __init__(self, passages: list[str], n_components: int = 150):
        self.passages = passages
        self.bm25 = BM25([tokenize(p) for p in passages])
        self.lsa = LSA(passages, n_components=n_components)

    def search(self, query: str, top_k: int = 10, mode: str = "hybrid") -> list[tuple[int, float]]:
        q_tokens = tokenize(query)
        if mode == "bm25":
            scores = self.bm25.scores(q_tokens)
        elif mode == "semantic":
            scores = self.lsa.scores(query)
        elif mode == "hybrid":
            bm = self.bm25.scores(q_tokens)
            scores = reciprocal_rank_fusion([bm, self.lsa.scores(query)])
            scores[bm == 0] *= 0.5  # favour passages with at least one literal match
        else:
            raise ValueError("mode must be 'bm25', 'semantic' or 'hybrid'")
        order = np.argsort(-scores)[:top_k]
        return [(int(i), float(scores[i])) for i in order if scores[i] > 0]
