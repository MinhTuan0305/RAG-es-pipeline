import os

import torch
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from FlagEmbedding import BGEM3FlagModel, FlagReranker
from google import genai
from google.genai import types

load_dotenv()  # reads .env in the working directory into os.environ, if present

ES_HOST = "http://localhost:9200"
INDEX_NAME = "gatsby-chunks"

EMBED_MODEL_NAME = "BAAI/bge-m3"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
GEMINI_MODEL = "gemini-2.5-flash"

SOURCE_EXCLUDES = {"excludes": ["embedding", "sparse_embedding"]}

INDEX_MAPPING = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
    },
    "mappings": {
        "properties": {
            "chunk_id": {"type": "keyword"},
            "book_id": {"type": "keyword"},
            "book_title": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
            "author": {"type": "keyword"},
            "chapter_number": {"type": "integer"},
            "chapter_title": {"type": "keyword"},
            "section_index": {"type": "integer"},
            "section_title": {"type": "keyword"},
            "chunk_index": {"type": "integer"},
            "global_index": {"type": "integer"},
            "type": {"type": "keyword"},
            "paragraph_ids": {"type": "integer"},
            "new_paragraph_ids": {"type": "integer"},
            "token_count": {"type": "integer"},
            "prev_chunk_id": {"type": "keyword"},
            "next_chunk_id": {"type": "keyword"},
            "text": {"type": "text", "analyzer": "english"},
            "embedding": {
                "type": "dense_vector",
                "dims": 1024,
                "index": True,
                "similarity": "cosine",
                "index_options": {"type": "hnsw", "m": 16, "ef_construction": 100},
            },
            "sparse_embedding": {"type": "rank_features"},
        }
    },
}

SYSTEM_PROMPT = """Bạn là trợ lý trả lời câu hỏi dựa trên kho tài liệu đã được index trong hệ thống -- kho này có thể chứa nhiều tài liệu khác nhau (sách, báo cáo, tài liệu,...), không chỉ riêng một cuốn.
Chỉ được trả lời dựa trên các đoạn trích (context) được cung cấp bên dưới -- KHÔNG dùng kiến thức bên ngoài, KHÔNG suy đoán hay bịa thêm chi tiết không có trong context.
Mỗi đoạn trích đều ghi rõ tên tài liệu nguồn -- nếu context có đoạn từ nhiều tài liệu khác nhau, hãy phân biệt rõ thông tin nào đến từ tài liệu nào, không gộp lẫn.
Nếu context không đủ thông tin để trả lời, hãy nói rõ là không tìm thấy thông tin liên quan trong các đoạn được cung cấp, đừng cố trả lời.
Khi trả lời, trích dẫn tên tài liệu và chương/mục liên quan (ví dụ: "(The Great Gatsby, Chapter III)") cho các chi tiết quan trọng."""


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_es_client():
    return Elasticsearch(ES_HOST)


def load_embed_model():
    device = get_device()
    return BGEM3FlagModel(EMBED_MODEL_NAME, use_fp16=(device == "cuda"), device=device)


def load_reranker():
    device = get_device()
    return FlagReranker(RERANKER_MODEL_NAME, use_fp16=(device == "cuda"), device=device)


def load_gemini_client():
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Chưa có GOOGLE_API_KEY trong biến môi trường. "
            "Lấy API key tại https://aistudio.google.com/apikey rồi set biến môi trường trước khi chạy app."
        )
    return genai.Client(api_key=api_key)


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


def knn_search(es, query_vector, k=20, num_candidates=100, book_id=None):
    knn = {"field": "embedding", "query_vector": query_vector, "k": k, "num_candidates": num_candidates}
    if book_id:
        # Filter inside the knn clause (pre-filtering) so we still get k hits from
        # that document, instead of taking the global top-k and discarding most of it.
        knn["filter"] = {"term": {"book_id": book_id}}
    resp = es.search(index=INDEX_NAME, knn=knn, source=SOURCE_EXCLUDES)
    return resp["hits"]["hits"]


def sparse_search(es, sparse_weights, size=20, top_n_tokens=32, book_id=None):
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


def reciprocal_rank_fusion(result_lists, k=60):
    fused = {}
    for hits in result_lists:
        for rank, hit in enumerate(hits, start=1):
            entry = fused.setdefault(hit["_id"], {"rrf_score": 0.0, "hit": hit})
            entry["rrf_score"] += 1.0 / (k + rank)
    return sorted(fused.values(), key=lambda e: -e["rrf_score"])


def hybrid_retrieve(es, embed_model, query_text, dense_k=20, sparse_k=20, fused_top_n=20, book_id=None):
    dense_vec, sparse_weights = embed_query(embed_model, query_text)

    dense_hits = knn_search(es, dense_vec, k=dense_k, num_candidates=dense_k * 5, book_id=book_id)
    sparse_hits = sparse_search(es, sparse_weights, size=sparse_k, book_id=book_id)

    fused = reciprocal_rank_fusion([dense_hits, sparse_hits])
    return fused[:fused_top_n]


def rerank(reranker, query_text, candidates, top_n=5):
    if not candidates:
        return []
    pairs = [[query_text, c["hit"]["_source"]["text"]] for c in candidates]
    scores = reranker.compute_score(pairs, normalize=True, max_length=1024)
    if not isinstance(scores, list):
        scores = [scores]  # compute_score returns a bare float for a single pair

    for c, score in zip(candidates, scores):
        c["rerank_score"] = score

    return sorted(candidates, key=lambda c: -c["rerank_score"])[:top_n]


def build_context(results):
    blocks = []
    for i, c in enumerate(results, start=1):
        src = c["hit"]["_source"]
        location = f"Chapter {src['chapter_title']}"
        if src.get("section_title"):
            location += f" > {src['section_title']}"
        blocks.append(
            f"[Đoạn {i} - Tài liệu: {src['book_title']} - {location}, chunk_id={src['chunk_id']}]\n{src['text']}"
        )
    return "\n\n".join(blocks)


def generate_answer(gemini_client, query_text, results, model=GEMINI_MODEL):
    context = build_context(results)
    user_prompt = f"""Context:
{context}

Câu hỏi: {query_text}

Trả lời:"""

    response = gemini_client.models.generate_content(
        model=model,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.2,
        ),
    )
    return response.text


def search(es, embed_model, reranker, query_text, dense_k=20, sparse_k=20, fused_top_n=20, final_top_n=5, book_id=None):
    fused = hybrid_retrieve(
        es, embed_model, query_text,
        dense_k=dense_k, sparse_k=sparse_k, fused_top_n=fused_top_n, book_id=book_id,
    )
    return rerank(reranker, query_text, fused, top_n=final_top_n)


def ask(es, embed_model, reranker, gemini_client, query_text, final_top_n=5):
    results = search(es, embed_model, reranker, query_text, final_top_n=final_top_n)
    answer = generate_answer(gemini_client, query_text, results)
    return answer, results


def ensure_index_exists(es):
    """Create the index with INDEX_MAPPING if missing. If it already exists (e.g. from
    before a new field like section_title was added), additively PUT any new field
    mappings -- this is safe: adding a new field to an existing ES mapping never
    requires reindexing, only *changing* an existing field's type would."""
    if not es.indices.exists(index=INDEX_NAME):
        es.indices.create(index=INDEX_NAME, body=INDEX_MAPPING)
    else:
        es.indices.put_mapping(index=INDEX_NAME, properties=INDEX_MAPPING["mappings"]["properties"])


def embed_chunks_batch(embed_model, chunks, batch_size=32, max_length=8192):
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
