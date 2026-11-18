"""Documents and loaders (plain text, Markdown, PDF)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .text import clean_text, sentence_windows


@dataclass
class Document:
    doc_id: str
    title: str
    text: str
    metadata: dict = field(default_factory=dict)

    @property
    def body(self) -> str:
        """Text without front-matter and section headings."""
        lines = [ln for ln in self.text.splitlines() if ln.strip().lstrip("#").strip() != self.title]
        return clean_text("\n".join(lines))

    @property
    def paragraphs(self) -> list[str]:
        """Retrieval passages: overlapping 3-sentence windows."""
        return sentence_windows(self.body)


def _title_from_text(text: str, fallback: str) -> str:
    for line in text.splitlines():
        line = line.strip().lstrip("#").strip()
        if line and not line.startswith("ID:"):
            return line[:200]
    return fallback


def read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as e:  # pragma: no cover
        raise ImportError("Reading PDFs needs pypdf: pip install 'policylens[pdf]'") from e
    reader = PdfReader(str(path))
    return "\n\n".join((page.extract_text() or "") for page in reader.pages)


def load_file(path: str | Path, doc_id: str | None = None) -> Document:
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        text = read_pdf(path)
    else:
        text = path.read_text(encoding="utf-8", errors="replace")
    meta = {"path": str(path)}
    # Optional front-matter lines such as "ID: XYZ-1" at the top of the file
    m = re.search(r"^ID:\s*(.+)$", text, flags=re.M)
    did = doc_id or (m.group(1).strip() if m else path.stem)
    return Document(doc_id=did, title=_title_from_text(text, path.stem), text=text, metadata=meta)


def load_directory(folder: str | Path, patterns: tuple[str, ...] = ("*.txt", "*.md", "*.pdf")) -> list[Document]:
    """Load every matching file below ``folder`` (recursively), sorted by path."""
    folder = Path(folder)
    files = sorted({p for pat in patterns for p in folder.rglob(pat)})
    if not files:
        raise FileNotFoundError(f"No {', '.join(patterns)} files found in {folder}")
    docs = [load_file(p) for p in files]
    seen: set[str] = set()
    for d in docs:
        if d.doc_id in seen:
            raise ValueError(f"Duplicate document id '{d.doc_id}'")
        seen.add(d.doc_id)
    return docs
