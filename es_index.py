"""
Elasticsearch index definition: name, mapping, and creation.

Kept separate from rag_pipeline.py so index tooling (index_admin.py) can use it
without importing torch/FlagEmbedding. rag_pipeline re-exports these names, so
existing `from rag_pipeline import INDEX_NAME, ...` imports keep working.
"""

import logging

from config import settings

logger = logging.getLogger(__name__)

ES_HOST = settings.es_host
# INDEX_NAME is an ALIAS, not a concrete index. Every search/count/bulk goes through
# it, while the data lives in a versioned index behind it (document-chunks-v1, -v2,
# ...). Changing the mapping = build the next version and swap the alias atomically
# (see index_admin.py) -- no code change, no downtime, old version kept for rollback.
INDEX_NAME = settings.es_index


def versioned_index(version: int) -> str:
    return f"{INDEX_NAME}-v{version}"


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
                "dims": settings.embed_dims,
                "index": True,
                "similarity": "cosine",
                "index_options": {"type": "hnsw", "m": 16, "ef_construction": 100},
            },
            "sparse_embedding": {"type": "rank_features"},
        }
    },
}


def ensure_index_exists(es):
    """Make sure INDEX_NAME is usable and has every field in INDEX_MAPPING.

    - Alias exists: additively PUT any new field mappings (adding a field never needs a
      reindex -- only *changing* an existing field does, which is index_admin.py's job).
    - Legacy concrete index named INDEX_NAME (from before aliases): still works the same
      way, but logs a hint to run `python index_admin.py migrate` once.
    - Nothing yet: create document-chunks-v1 with INDEX_NAME as its write alias.
    """
    properties = INDEX_MAPPING["mappings"]["properties"]
    if es.indices.exists_alias(name=INDEX_NAME):
        es.indices.put_mapping(index=INDEX_NAME, properties=properties)
    elif es.indices.exists(index=INDEX_NAME):
        logger.warning(
            "'%s' is a concrete index, not an alias -- run `python index_admin.py migrate` "
            "once to move it behind a versioned index.", INDEX_NAME,
        )
        es.indices.put_mapping(index=INDEX_NAME, properties=properties)
    else:
        es.indices.create(
            index=versioned_index(1),
            settings=INDEX_MAPPING["settings"],
            mappings=INDEX_MAPPING["mappings"],
            aliases={INDEX_NAME: {"is_write_index": True}},
        )
