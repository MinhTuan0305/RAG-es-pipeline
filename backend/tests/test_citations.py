import numpy as np

from app.services.citations import clean_citations, strip_citations


# ---------------------------------------------------------------- cleaning the answer

def test_clean_keeps_valid_and_reports_order_of_first_use():
    text, cited, invalid = clean_citations("A [2]. B [1][2]. C [3].", {1, 2, 3})
    assert text == "A [2]. B [1][2]. C [3]."
    assert cited == [2, 1, 3] and invalid == []


def test_clean_normalizes_lists_and_drops_made_up_numbers():
    text, cited, invalid = clean_citations("A [1, 3]. B [2;9] x. C [7].", {1, 2, 3})
    assert text == "A [1][3]. B [2] x. C."
    assert cited == [1, 3, 2] and invalid == [9, 7]


def test_clean_leaves_links_years_and_plain_brackets_alone():
    text = "Xem [tài liệu](https://example.com), năm [2024], mảng a[i], bảng | [1](#x) |"
    assert clean_citations(text, set()) == (text, [], [])


def test_strip_for_history():
    assert strip_citations("Gatsby giàu có [1][2] và bí ẩn [3].") == "Gatsby giàu có và bí ẩn."


# ---------------------------------------------------------------- numbering in the agent's tool

DOCS = {
    0.0: ["c1", "c2", "c3"],  # what the first query finds
    1.0: ["c2", "c4"],        # the second query re-finds c2 and adds c4
}


class FakeEmbed:
    def encode(self, texts, **kwargs):
        code = 0.0 if texts[0] == "first query" else 1.0
        return {"dense_vecs": np.array([[code, 0.0]]), "lexical_weights": [{"5": 0.5}]}

    def convert_id_to_token(self, lexical_weights):
        return {"tok": 0.5}


class FakeSearchES:
    def search(self, index, knn=None, query=None, size=None, source=None):
        code = knn["query_vector"][0] if knn else self.last_code
        self.last_code = code
        hits = [{"_id": c, "_score": 1.0, "_source": {
            "chunk_id": c, "book_title": "Doc", "chapter_title": "I", "section_title": None,
            "type": "paragraph", "text": f"text of {c}",
        }} for c in DOCS[code]]
        return {"hits": {"hits": hits}}


class FakeReranker:
    def compute_score(self, pairs, **kwargs):
        return [1.0 - i * 0.1 for i in range(len(pairs))]


def test_citation_numbers_run_across_searches(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")  # the LLM is never called here
    from app.core.langgraph_agent import DocumentQAAgent, RunContext

    agent = DocumentQAAgent(FakeSearchES(), FakeEmbed(), FakeReranker())
    tool = agent.graph.nodes["tools"].bound.tools_by_name["search_documents"]
    ctx = RunContext()
    config = {"configurable": {"run_context": ctx}}

    first = tool.invoke({"query": "first query"}, config=config)
    second = tool.invoke({"query": "second query"}, config=config)

    assert [(c["citation"], c["hit"]["_source"]["chunk_id"]) for c in ctx.sources] == [
        (1, "c1"), (2, "c2"), (3, "c3"), (4, "c4"),  # c2 found again: not renumbered
    ]
    assert first.startswith("[1] Tài liệu: Doc - Chapter I\ntext of c1")
    assert "[2] Tài liệu" in first and "[3] Tài liệu" in first
    assert second.startswith("[4] Tài liệu: Doc - Chapter I\ntext of c4")
    assert "c2" not in second  # already sent to the model, citable as [2]
