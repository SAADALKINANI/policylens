"""PolicyLens tour on the bundled fictional NPA policy corpus.

Run:  python examples/demo.py
Writes output/citation_graph.png and output/citation_graph.graphml.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402

from policylens import PolicyLens, write_policy_corpus  # noqa: E402

corpus = Path("data/sample_corpus")
if not corpus.exists():
    write_policy_corpus(corpus)
lens = PolicyLens.from_directory(corpus)
print(f"Indexed {len(lens.docs)} documents, {len(lens.passages)} passages\n")

print("=== Search ===")
for q in ["how quickly must a security incident be reported",
          "can staff paste personal information into generative AI tools",
          "when can records be destroyed"]:
    a = lens.answer(q)
    print(f"Q: {q}\nA: {a['answer']}\n   sources: {', '.join(a['sources'])}\n")

print("=== Summary of Directive 3000.01 ===")
print(lens.summarize("Directive 3000.01", n_sentences=2), "\n")

print("=== Most influential nodes in the citation graph ===")
for node, pr, cited in lens.most_influential(8):
    print(f"  {node:28s} pagerank={pr:.3f}  cited by {cited}")

print("\n=== Who relies on the Federal Records Act (44 U.S.C. 3101)? ===")
print("  " + ", ".join(lens.cited_by("44 U.S.C. 3101")))

print("\n=== Documents most similar to Instruction 3000.03 ===")
for d, s in lens.similar("Instruction 3000.03", top_k=3):
    print(f"  {d:22s} {s:.2f}")

print("\n=== Acronym glossary (first 8) ===")
for acr, e in list(lens.glossary.items())[:8]:
    print(f"  {acr:6s} {e['expansion']}")

out = Path("output")
out.mkdir(exist_ok=True)
nx.write_graphml(lens.graph, out / "citation_graph.graphml")
g = lens.graph
colors = ["#2f6db3" if g.nodes[n].get("in_corpus") else "#e07b39" for n in g]
sizes = [80 + 120 * g.in_degree(n) for n in g]
plt.figure(figsize=(13, 10))
pos = nx.spring_layout(g, k=0.45, seed=3)
nx.draw_networkx(g, pos, node_color=colors, node_size=sizes, font_size=6, edge_color="#bbbbbb",
                 arrowsize=6, width=0.6)
plt.title("NPA citation graph (blue = agency issuances, orange = external authorities)")
plt.axis("off")
plt.tight_layout()
plt.savefig(out / "citation_graph.png", dpi=130)
print(f"\nSaved {out / 'citation_graph.png'} and {out / 'citation_graph.graphml'}")
