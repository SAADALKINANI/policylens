"""Extractive summarisation with TextRank (Mihalcea & Tarau, 2004).

Sentences are nodes; edges are TF-IDF cosine similarities; PageRank picks
the most central sentences. Extractive output never invents text, which is
the right default for policy and legal documents.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .text import split_sentences, tokenize


def textrank(text: str, n_sentences: int = 3, query: str | None = None, query_weight: float = 0.5,
             min_words: int = 6) -> list[str]:
    """Return the ``n_sentences`` most central sentences, in document order.

    With ``query`` the ranking is personalised (biased PageRank) towards
    sentences similar to the query, which gives a query-focused summary.
    """
    sents = list(dict.fromkeys(s for s in split_sentences(text) if len(s.split()) >= min_words))
    if len(sents) <= n_sentences:
        return sents
    vec = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None)
    try:
        X = vec.fit_transform(sents + ([query] if query else []))
    except ValueError:  # empty vocabulary
        return sents[:n_sentences]
    S = X[: len(sents)]
    sim = cosine_similarity(S)
    np.fill_diagonal(sim, 0.0)
    g = nx.from_numpy_array(sim)
    personalization = None
    if query:
        q_sim = cosine_similarity(S, X[len(sents):]).ravel()
        base = np.full(len(sents), (1 - query_weight) / len(sents))
        if q_sim.sum() > 0:
            base += query_weight * q_sim / q_sim.sum()
        personalization = {i: float(v) for i, v in enumerate(base)}
    try:
        scores = nx.pagerank(g, personalization=personalization, max_iter=200)
    except nx.PowerIterationFailedConvergence:  # pragma: no cover
        scores = {i: float(sim[i].sum()) for i in range(len(sents))}
    top = sorted(sorted(scores, key=scores.get, reverse=True)[:n_sentences])
    return [sents[i] for i in top]
