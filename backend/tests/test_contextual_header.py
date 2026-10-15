import sys
from pathlib import Path

import numpy as np

from app.core import rag_pipeline
from app.core.rag_pipeline import EMBED_FORMAT_CONTEXTUAL, EMBED_FORMAT_PLAIN, embed_chunks_batch, passage_with_context, rerank

CHUNK = {"chunk_id": "doc-c1", "book_title": "Sổ tay nhân sự", "chapter_title": "3. Phúc lợi",
         "section_title": "3.2. Chính sách nghỉ phép", "text": "Nhân viên được 12 ngày mỗi năm."}


class RecordingEmbed:
    def __init__(self):
        self.texts = []

    def encode(self, texts, **kwargs):
        self.texts.extend(texts)
        return {"dense_vecs": np.ones((len(texts), 2)), "lexical_weights": [{"5": 0.5} for _ in texts]}

    def convert_id_to_token(self, lexical_weights):
        out = [{"tok": 0.5} for _ in lexical_weights]
        return out[0] if len(out) == 1 else out


class RecordingReranker:
    def __init__(self):
        self.pairs = []

    def compute_score(self, pairs, **kwargs):
        self.pairs.extend(pairs)
        return [0.9 for _ in pairs]


def test_passage_gets_document_and_location_line():
    assert passage_with_context(CHUNK) == (
        "Sổ tay nhân sự — Chapter 3. Phúc lợi > 3.2. Chính sách nghỉ phép\nNhân viên được 12 ngày mỗi năm."
    )


def test_embedding_and_rerank_use_the_header_but_text_is_untouched():
    model = RecordingEmbed()
    chunks = [dict(CHUNK)]
    embed_chunks_batch(model, chunks)
    assert model.texts == [passage_with_context(CHUNK)]
    assert chunks[0]["text"] == CHUNK["text"]  # what the UI shows / the LLM reads is unchanged
    assert chunks[0]["embed_format"] == EMBED_FORMAT_CONTEXTUAL

    reranker = RecordingReranker()
    rerank(reranker, "nghỉ phép bao nhiêu ngày?", [{"hit": {"_source": dict(CHUNK)}, "rrf_score": 0.1}])
    assert reranker.pairs == [["nghỉ phép bao nhiêu ngày?", passage_with_context(CHUNK)]]


def test_header_can_be_turned_off(monkeypatch):
    monkeypatch.setattr(rag_pipeline, "settings", rag_pipeline.settings.__class__(
        **{**rag_pipeline.settings.__dict__, "embed_context_header": False},
    ))
    assert passage_with_context(CHUNK) == CHUNK["text"]
    model = RecordingEmbed()
    chunks = [dict(CHUNK)]
    embed_chunks_batch(model, chunks)
    assert model.texts == [CHUNK["text"]] and chunks[0]["embed_format"] == EMBED_FORMAT_PLAIN


# ---------------------------------------------------------------- reembed script

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from reembed import reembed  # noqa: E402


class PagingES:
    """search(): match_all sorted by chunk_id with search_after, like the script asks."""

    def __init__(self, docs):
        self.docs = sorted(docs, key=lambda d: d["chunk_id"])
        self.requests = []

    def search(self, index, size, query, sort, search_after, source):
        self.requests.append(search_after)
        start = 0 if search_after is None else next(
            i + 1 for i, d in enumerate(self.docs) if d["chunk_id"] == search_after[0]
        )
        page = self.docs[start:start + size]
        return {"hits": {"hits": [{"_source": dict(d), "sort": [d["chunk_id"]]} for d in page]}}


def test_reembed_pages_through_everything_and_writes_new_vectors():
    docs = [{**CHUNK, "chunk_id": f"doc-c{i:02d}", "text": f"đoạn {i}"} for i in range(8)]
    es = PagingES(docs)
    written, progress = [], []
    model = RecordingEmbed()

    done = reembed(es, model, "document-chunks-v1", written.extend, on_progress=progress.append, page_size=3)

    assert done == 8 and progress == [3, 6, 8]
    assert [c["chunk_id"] for c in written] == [d["chunk_id"] for d in es.docs]
    assert all(c["embed_format"] == EMBED_FORMAT_CONTEXTUAL and c["embedding"] == [1.0, 1.0] for c in written)
    assert model.texts[0].startswith("Sổ tay nhân sự — Chapter 3. Phúc lợi")
    assert es.requests[0] is None and es.requests[1] == ["doc-c02"]  # search_after paging
