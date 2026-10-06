from dataclasses import dataclass

import torch
from elasticsearch import Elasticsearch
from FlagEmbedding import BGEM3FlagModel, FlagReranker

from config import settings
from es_index import ES_HOST, INDEX_MAPPING, INDEX_NAME, ensure_index_exists, versioned_index  # noqa: F401 (re-exported)

EMBED_MODEL_NAME = settings.embed_model
RERANKER_MODEL_NAME = settings.reranker_model
GEMINI_MODEL = settings.llm_model

SOURCE_EXCLUDES = {"excludes": ["embedding", "sparse_embedding"]}

# kNN looks at k * this many candidates per shard before picking the top k.
KNN_CANDIDATES_FACTOR = 5


def get_device():
    if settings.model_device != "auto":
        return settings.model_device
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_es_client():
    return Elasticsearch(ES_HOST)


def load_embed_model():
    device = get_device()
    return BGEM3FlagModel(EMBED_MODEL_NAME, use_fp16=(device == "cuda"), device=device)


def load_reranker():
    device = get_device()
    return FlagReranker(RERANKER_MODEL_NAME, use_fp16=(device == "cuda"), device=device)


def embed_query(embed_model, query_text):
    output = embed_model.encode(
        [query_text],
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    dense_vec = [float(x) for x in output["dense_vecs"][0]]

    # convert_id_to_token collapses its result to a bare dict (not a 1-item list)
    # whenever the input batch has exactly one entry -- always true here since we
    # always encode a single query string.
    raw_sparse = embed_model.convert_id_to_token(output["lexical_weights"])
    if isinstance(raw_sparse, list):
        raw_sparse = raw_sparse[0]
    sparse_weights = {tok.replace(".", "·"): float(w) for tok, w in raw_sparse.items()}

    return dense_vec, sparse_weights


def list_documents(es):
    """All documents currently in the index: [{book_id, book_title, num_chunks}, ...]."""
    resp = es.search(
        index=INDEX_NAME,
        size=0,
        aggs={
            "docs": {
                "terms": {"field": "book_id", "size": 1000},
                "aggs": {"title": {"terms": {"field": "book_title.keyword", "size": 1}}},
            }
        },
    )
    docs = []
    for bucket in resp["aggregations"]["docs"]["buckets"]:
        title_buckets = bucket["title"]["buckets"]
        docs.append({
            "book_id": bucket["key"],
            "book_title": title_buckets[0]["key"] if title_buckets else bucket["key"],
            "num_chunks": bucket["doc_count"],
        })
    return sorted(docs, key=lambda d: d["book_title"].lower())


def knn_search(es, query_vector, k=settings.dense_k, num_candidates=None, book_id=None):
    num_candidates = num_candidates or k * KNN_CANDIDATES_FACTOR
    knn = {"field": "embedding", "query_vector": query_vector, "k": k, "num_candidates": num_candidates}
    if book_id:
        # Filter inside the knn clause (pre-filtering) so we still get k hits from
        # that document, instead of taking the global top-k and discarding most of it.
        knn["filter"] = {"term": {"book_id": book_id}}
    resp = es.search(index=INDEX_NAME, knn=knn, source=SOURCE_EXCLUDES)
    return resp["hits"]["hits"]


def sparse_search(es, sparse_weights, size=settings.sparse_k, top_n_tokens=settings.sparse_top_tokens, book_id=None):
    top_tokens = sorted(sparse_weights.items(), key=lambda kv: -kv[1])[:top_n_tokens]
    should_clauses = [
        {"rank_feature": {"field": f"sparse_embedding.{token}", "boost": weight}}
        for token, weight in top_tokens
    ]
    bool_query = {"should": should_clauses}
    if book_id:
        bool_query["filter"] = [{"term": {"book_id": book_id}}]
        # Once a bool has a filter clause, should clauses become optional by default,
        # which would match every chunk of the document with score 0. Require at
        # least one token to actually match.
        bool_query["minimum_should_match"] = 1
    resp = es.search(
        index=INDEX_NAME,
        query={"bool": bool_query},
        size=size,
        source=SOURCE_EXCLUDES,
    )
    return resp["hits"]["hits"]


def reciprocal_rank_fusion(result_lists, k=settings.rrf_k):
    fused = {}
    for hits in result_lists:
        for rank, hit in enumerate(hits, start=1):
            entry = fused.setdefault(hit["_id"], {"rrf_score": 0.0, "hit": hit})
            entry["rrf_score"] += 1.0 / (k + rank)
    return sorted(fused.values(), key=lambda e: -e["rrf_score"])


def rerank(reranker, query_text, candidates, top_n=settings.final_top_n):
    if not candidates:
        return []
    pairs = [[query_text, c["hit"]["_source"]["text"]] for c in candidates]
    scores = reranker.compute_score(pairs, normalize=True, max_length=settings.rerank_max_length)
    if not isinstance(scores, list):
        scores = [scores]  # compute_score returns a bare float for a single pair

    for c, score in zip(candidates, scores):
        c["rerank_score"] = score

    return sorted(candidates, key=lambda c: -c["rerank_score"])[:top_n]


# ---------------------------------------------------------------- retrieval (single entry point)

@dataclass(frozen=True)
class RetrievalParams:
    """Knobs for one retrieve() call. Defaults come from .env (config.py); pass a
    different instance to compare settings, e.g. in an evaluation run."""

    dense_k: int = settings.dense_k
    sparse_k: int = settings.sparse_k
    sparse_top_tokens: int = settings.sparse_top_tokens
    num_candidates_factor: int = KNN_CANDIDATES_FACTOR
    rrf_k: int = settings.rrf_k
    fused_top_n: int = settings.fused_top_n
    final_top_n: int = settings.final_top_n

    @property
    def num_candidates(self) -> int:
        return self.dense_k * self.num_candidates_factor

    def describe(self) -> str:
        return (
            f"dense_k={self.dense_k} sparse_k={self.sparse_k} fused_top_n={self.fused_top_n} "
            f"final_top_n={self.final_top_n} rrf_k={self.rrf_k}"
        )


@dataclass
class RetrievalResult:
    """Everything one retrieve() call produced, intermediate lists included (handy for
    evaluation: e.g. recall of dense vs sparse without querying ES again)."""

    query: str
    dense_hits: list   # raw ES hits from kNN
    sparse_hits: list  # raw ES hits from the rank_feature query
    fused: list        # RRF entries {"hit", "rrf_score"}, best fused_top_n
    ranked: list       # fused entries + "rerank_score", best final_top_n -- the final result


def plain_step(name, inputs, fn, summarize):
    """Default `run_step` for retrieve(): just run the step. The agent passes
    tracing.traced_step instead, which also records each step as a trace span."""
    return fn()


def chunk_location(src) -> str:
    location = f"Chapter {src['chapter_title']}"
    if src.get("section_title"):
        location += f" > {src['section_title']}"
    return location


def _hit_row(rank, hit, **scores):
    """One readable line of a ranked result list in a trace (no passage text)."""
    src = hit["_source"]
    return {"rank": rank, "chunk_id": src["chunk_id"], "document": src["book_title"],
            "location": chunk_location(src), **scores}


def _model_label(model_name: str) -> str:
    return model_name.split("/")[-1]


def retrieve(es, embed_model, reranker, query, *, book_id=None, params=RetrievalParams(),
             run_step=plain_step, scope_label=None) -> RetrievalResult:
    """The whole retrieval pipeline for one query:
    embed (dense + sparse) -> kNN + sparse search -> RRF fusion -> rerank.

    book_id: restrict to one document (None = whole library).
    run_step(name, inputs, fn, summarize): wraps every step. Default just calls fn();
        pass tracing.traced_step to get one trace span per step, where `inputs` and
        `summarize(result)` are the small readable dicts shown in the trace.
    scope_label: human-readable scope for the trace (e.g. the document title).
    """
    scope = scope_label or book_id or "all documents"

    dense_vec, sparse_weights = run_step(
        f"1. embed_query ({_model_label(EMBED_MODEL_NAME)})", {"query": query},
        lambda: embed_query(embed_model, query),
        lambda r: {
            "dense_dims": len(r[0]),
            "top_sparse_tokens": {
                tok: round(w, 3) for tok, w in sorted(r[1].items(), key=lambda kv: -kv[1])[:10]
            },
        },
    )
    dense_hits = run_step(
        "2a. dense_knn_search",
        {"k": params.dense_k, "num_candidates": params.num_candidates, "scope": scope},
        lambda: knn_search(es, dense_vec, k=params.dense_k, num_candidates=params.num_candidates, book_id=book_id),
        lambda hits: [_hit_row(i, h, knn_score=round(h["_score"], 4)) for i, h in enumerate(hits, 1)],
    )
    sparse_hits = run_step(
        "2b. sparse_search (rank_feature)",
        {"size": params.sparse_k, "top_tokens": params.sparse_top_tokens, "scope": scope},
        lambda: sparse_search(
            es, sparse_weights, size=params.sparse_k, top_n_tokens=params.sparse_top_tokens, book_id=book_id,
        ),
        lambda hits: [_hit_row(i, h, sparse_score=round(h["_score"], 4)) for i, h in enumerate(hits, 1)],
    )

    dense_rank = {h["_id"]: i for i, h in enumerate(dense_hits, 1)}
    sparse_rank = {h["_id"]: i for i, h in enumerate(sparse_hits, 1)}
    fused = run_step(
        "3. rrf_fusion",
        {"rrf_k": params.rrf_k, "dense_hits": len(dense_hits), "sparse_hits": len(sparse_hits),
         "keep_top": params.fused_top_n},
        lambda: reciprocal_rank_fusion([dense_hits, sparse_hits], k=params.rrf_k)[:params.fused_top_n],
        lambda entries: [
            _hit_row(
                i, e["hit"], rrf_score=round(e["rrf_score"], 5),
                dense_rank=dense_rank.get(e["hit"]["_id"]), sparse_rank=sparse_rank.get(e["hit"]["_id"]),
            )
            for i, e in enumerate(entries, 1)
        ],
    )
    ranked = run_step(
        f"4. rerank ({_model_label(RERANKER_MODEL_NAME)})",
        {"query": query, "candidates": len(fused), "keep_top": params.final_top_n},
        lambda: rerank(reranker, query, fused, top_n=params.final_top_n),
        lambda entries: [
            _hit_row(i, e["hit"], rerank_score=round(float(e["rerank_score"]), 4), rrf_score=round(e["rrf_score"], 5))
            for i, e in enumerate(entries, 1)
        ],
    )
    return RetrievalResult(query=query, dense_hits=dense_hits, sparse_hits=sparse_hits, fused=fused, ranked=ranked)


def build_context(results):
    blocks = []
    for i, c in enumerate(results, start=1):
        src = c["hit"]["_source"]
        kind = "Bảng" if src.get("type") == "table" else "Đoạn"
        blocks.append(
            f"[{kind} {i} - Tài liệu: {src['book_title']} - {chunk_location(src)}, chunk_id={src['chunk_id']}]\n{src['text']}"
        )
    return "\n\n".join(blocks)


def embed_chunks_batch(embed_model, chunks, batch_size=settings.embed_batch_size, max_length=settings.embed_max_length):
    """Embed a list of chunk dicts (each with a "text" key) in place -- adds
    "embedding" (dense) and "sparse_embedding" (lexical weights) to every chunk.
    Used when indexing a new document, as opposed to embed_query() which embeds a
    single query string at search time."""
    texts = [c["text"] for c in chunks]
    output = embed_model.encode(
        texts,
        batch_size=batch_size,
        max_length=max_length,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    dense_vecs = output["dense_vecs"]

    raw_sparse_list = embed_model.convert_id_to_token(output["lexical_weights"])
    if isinstance(raw_sparse_list, dict):
        raw_sparse_list = [raw_sparse_list]  # convert_id_to_token unwraps 1-item batches

    for chunk, dense_vec, sparse in zip(chunks, dense_vecs, raw_sparse_list):
        chunk["embedding"] = [float(x) for x in dense_vec]
        chunk["sparse_embedding"] = {k.replace(".", "·"): float(v) for k, v in sparse.items()}

    return chunks


def index_chunks(es, chunks):
    """Bulk-index chunk dicts, using chunk_id as the document _id (idempotent re-runs)."""
    from elasticsearch.helpers import bulk

    def actions():
        for c in chunks:
            yield {"_index": INDEX_NAME, "_id": c["chunk_id"], "_source": c}

    success_count, errors = bulk(es, actions())
    return success_count, errors
