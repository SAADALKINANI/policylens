"""``policylens`` command-line interface."""

from __future__ import annotations

import argparse
import json

from . import __version__
from .engine import PolicyLens
from .synthetic import write_policy_corpus


def _lens(a) -> PolicyLens:
    return PolicyLens.load(a.index) if a.index else PolicyLens.from_directory(a.corpus)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="policylens", description="Search and connect policy documents")
    p.add_argument("--version", action="version", version=f"policylens {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sample-corpus", help="write the fictional NPA policy corpus")
    s.add_argument("--out", default="data/sample_corpus")

    b = sub.add_parser("index", help="build and save an index from a folder of .txt/.md/.pdf files")
    b.add_argument("corpus")
    b.add_argument("--out", default="policylens.idx")

    def add_src(sp):
        sp.add_argument("--corpus", default="data/sample_corpus", help="folder of documents")
        sp.add_argument("--index", help="saved index (faster than --corpus)")

    q = sub.add_parser("search", help="hybrid search over passages")
    q.add_argument("query")
    q.add_argument("--top", type=int, default=5)
    q.add_argument("--mode", default="hybrid", choices=["hybrid", "bm25", "semantic"])
    add_src(q)

    a = sub.add_parser("ask", help="extractive answer with sources")
    a.add_argument("question")
    add_src(a)

    m = sub.add_parser("summarize", help="TextRank summary of one document")
    m.add_argument("doc_id")
    m.add_argument("--sentences", type=int, default=3)
    add_src(m)

    g = sub.add_parser("graph", help="most influential documents and authorities")
    g.add_argument("--top", type=int, default=10)
    g.add_argument("--doc", help="show what this document cites and who cites it")
    g.add_argument("--export", help="write the citation graph as GraphML")
    add_src(g)

    c = sub.add_parser("acronyms", help="acronym glossary")
    add_src(c)

    args = p.parse_args(argv)
    if args.cmd == "sample-corpus":
        paths = write_policy_corpus(args.out)
        print(f"wrote {len(paths)} documents → {args.out}")
    elif args.cmd == "index":
        lens = PolicyLens.from_directory(args.corpus)
        lens.save(args.out)
        print(f"indexed {len(lens.docs)} documents / {len(lens.passages)} passages → {args.out}")
    elif args.cmd == "search":
        for i, h in enumerate(_lens(args).search(args.query, top_k=args.top, mode=args.mode), 1):
            print(f"{i}. {h}\n")
    elif args.cmd == "ask":
        print(json.dumps(_lens(args).answer(args.question), indent=2))
    elif args.cmd == "summarize":
        print(_lens(args).summarize(args.doc_id, n_sentences=args.sentences))
    elif args.cmd == "graph":
        import networkx as nx
        lens = _lens(args)
        if args.doc:
            print("cites:   ", ", ".join(lens.references(args.doc)))
            print("cited by:", ", ".join(lens.cited_by(args.doc)) or "-")
        else:
            print(f"{'node':32s} {'pagerank':>9s} {'cited':>6s}")
            for node, pr, deg in lens.most_influential(args.top):
                print(f"{node:32s} {pr:9.4f} {deg:6d}")
        if args.export:
            nx.write_graphml(lens.graph, args.export)
            print(f"graph → {args.export}")
    elif args.cmd == "acronyms":
        for acr, e in _lens(args).glossary.items():
            print(f"{acr:8s} {e['expansion']}  ({e['count']}×)")


if __name__ == "__main__":
    main()
