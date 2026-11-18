import pytest

from policylens import (PolicyLens, ReferenceExtractor, extract_acronyms, generate_policy_corpus,
                        textrank, write_policy_corpus)
from policylens.cli import main as cli
from policylens.text import split_sentences, stem, tokenize


@pytest.fixture(scope="module")
def corpus_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("corpus")
    write_policy_corpus(d)
    return d


@pytest.fixture(scope="module")
def lens(corpus_dir):
    return PolicyLens.from_directory(corpus_dir)


def test_text_utils():
    assert split_sentences("See 5 U.S.C. 552a for details. Then report it.") == [
        "See 5 U.S.C. 552a for details.", "Then report it."]
    assert stem("reported") == stem("reporting") == "report"
    assert "8510.01" in tokenize("Per Instruction 8510.01, the owner shall act")


def test_reference_extraction():
    refs = ReferenceExtractor().extract(
        "Under 44 U.S.C. § 3101, E.O. 13960, Pub. L. 117-263, OMB Memorandum M-25-21 and Directive 1000.01.")
    assert set(refs) == {"44 U.S.C. 3101", "Executive Order 13960", "Public Law 117-263", "OMB M-25-21",
                         "Directive 1000.01"}


def test_acronyms():
    g = extract_acronyms(["The Chief AI Officer (CAIO) and the Department of Defense (DoD) and the "
                          "Office of the Chief Data Officer (OCDO) met."])
    assert g["CAIO"]["expansion"] == "Chief AI Officer"
    assert g["DoD"]["expansion"] == "Department of Defense"


def test_corpus_shape():
    docs = generate_policy_corpus()
    assert len(docs) == 45 and len({d["doc_id"] for d in docs}) == 45


@pytest.mark.parametrize("query,expected_series", [
    ("how quickly must a security incident be reported", "1000"),
    ("telework agreement", "6000"),
    ("when can records be destroyed", "2000"),
    ("documenting AI training data", "8000"),
])
def test_search_finds_right_topic(lens, query, expected_series):
    hits = lens.search(query, top_k=3)
    assert all(h.doc_id.split()[1].startswith(expected_series) for h in hits)


def test_answer_extracts_fact(lens):
    a = lens.answer("how quickly must a security incident be reported")
    assert "within one hour" in a["answer"] and a["sources"]


def test_graph(lens):
    assert "Directive 1000.01" in lens.references("Instruction 1000.03")
    assert "Instruction 1000.03" in lens.cited_by("Directive 1000.01")
    top = lens.most_influential(5, in_corpus_only=True)
    assert all(n.startswith("Directive") for n, _, _ in top)


def test_summary_and_similarity(lens):
    s = lens.summarize("Directive 3000.01", n_sentences=2)
    assert 0 < len(s) < 600
    assert all(d.split()[1].startswith("3000") for d, _ in lens.similar("Instruction 3000.02", top_k=3))


def test_textrank_query_focus():
    text = ("Cats are small animals. Dogs are loyal animals that like walks. "
            "Budgets must be approved by the finance office. The finance office reviews every budget each year.")
    assert "budget" in " ".join(textrank(text, 1, query="budget approval")).lower()


def test_save_load_and_cli(lens, corpus_dir, tmp_path, capsys):
    idx = tmp_path / "x.idx"
    lens.save(idx)
    assert PolicyLens.load(idx).search("telework")[0].doc_id.startswith(("Directive 6000", "Instruction 6000"))
    cli(["search", "records retention", "--index", str(idx), "--top", "2"])
    cli(["graph", "--corpus", str(corpus_dir), "--top", "3"])
    cli(["acronyms", "--index", str(idx)])
    out = capsys.readouterr().out
    assert "1." in out and "pagerank" in out and "CAIO" in out
