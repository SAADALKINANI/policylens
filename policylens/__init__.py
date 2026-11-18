"""PolicyLens: search, summarise and connect government policy documents."""

__version__ = "0.1.0"

from .acronyms import extract_acronyms  # noqa: E402
from .corpus import Document, load_directory, load_file  # noqa: E402
from .engine import PolicyLens, SearchHit  # noqa: E402
from .references import ReferenceExtractor, build_citation_graph, influence  # noqa: E402
from .summarize import textrank  # noqa: E402
from .synthetic import generate_policy_corpus, write_policy_corpus  # noqa: E402

__all__ = ["PolicyLens", "SearchHit", "Document", "load_directory", "load_file", "textrank",
           "ReferenceExtractor", "build_citation_graph", "influence", "extract_acronyms",
           "generate_policy_corpus", "write_policy_corpus"]
