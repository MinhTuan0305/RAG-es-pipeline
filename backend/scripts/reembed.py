"""
Re-embed every chunk of the library into a new index version, then switch the alias.

Needed after changing HOW chunks are embedded -- e.g. turning on the contextual chunk
header (EMBED_CONTEXT_HEADER), or switching the embedding model. `index_admin.py
reindex` can't do this: it copies the old vectors as they are.

    python backend/scripts/reembed.py                # shows the plan, asks to confirm
    python backend/scripts/reembed.py -y --delete-old

Stop the backend first: this script loads the embedding model itself (RAM/VRAM), and a
document uploaded during the run would be missing from the new version (the doc-count
check catches that and aborts before switching). The old version is kept for rollback
(`python backend/scripts/index_admin.py switch <old>`) unless --delete-old.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # package "app" (backend/app)
sys.path.insert(0, str(Path(__file__).resolve().parent))  # index_admin, next to this file

from elasticsearch.helpers import bulk  # noqa: E402

from app.core.es_index import INDEX_NAME  # noqa: E402
from app.core.rag_pipeline import (  # noqa: E402
    SOURCE_EXCLUDES,
    current_embed_format,
    embed_chunks_batch,
    load_embed_model,
    load_es_client,
)
from index_admin import alias_targets, confirm, count, create_version, next_version_name  # noqa: E402

PAGE_SIZE = 64


def iter_chunk_pages(es, index, page_size=PAGE_SIZE):
    """Every chunk of `index` without its vectors, page by page, in chunk_id order
    (search_after paging: no scroll context to keep alive between slow embed batches)."""
    after = None
    while True:
        resp = es.search(
            index=index, size=page_size, query={"match_all": {}}, sort=[{"chunk_id": "asc"}],
            search_after=after, source=SOURCE_EXCLUDES,
        )
        hits = resp["hits"]["hits"]
        if not hits:
            return
        yield [h["_source"] for h in hits]
        after = hits[-1]["sort"]


def bulk_writer(es, dest):
    def write(chunks):
        _, errors = bulk(es, ({"_index": dest, "_id": c["chunk_id"], "_source": c} for c in chunks), raise_on_error=False)
        if errors:
            raise RuntimeError(f"{len(errors)} chunk lỗi khi ghi vào {dest}; lỗi đầu tiên: {str(errors[0])[:300]}")
    return write


def reembed(es, embed_model, source, write, on_progress=None, on_retry=None, page_size=PAGE_SIZE) -> int:
    """Embed every chunk of `source` again (current settings) and hand each page to
    write(chunks). Returns how many chunks were done."""
    done = 0
    for page in iter_chunk_pages(es, source, page_size):
        embed_chunks_batch(embed_model, page, on_retry=on_retry)
        write(page)
        done += len(page)
        if on_progress:
            on_progress(done)
    return done


def embed_format_counts(es, index) -> dict:
    """{embed_format: chunk count}; chunks from before the field existed count as "(chưa ghi)"."""
    try:
        resp = es.search(index=index, size=0, aggs={"f": {"terms": {"field": "embed_format", "missing": "(chưa ghi)"}}})
    except Exception:  # noqa: BLE001 -- informational only
        return {}
    return {b["key"]: b["doc_count"] for b in resp["aggregations"]["f"]["buckets"]}


def main():
    parser = argparse.ArgumentParser(description="Re-embed the whole library into a new index version.")
    parser.add_argument("-y", "--yes", action="store_true", help="skip the confirmation prompt")
    parser.add_argument("--delete-old", action="store_true", help="delete the previous version after the switch")
    args = parser.parse_args()

    es = load_es_client()
    targets = alias_targets(es)
    if len(targets) != 1:
        sys.exit(f"'{INDEX_NAME}' must be an alias of exactly one index (now: {targets or 'none'}); "
                 "see `python backend/scripts/index_admin.py status`.")
    current = targets[0]
    total = count(es, current)
    dest = next_version_name(es)

    print(f"Index hiện tại: {current} ({total} chunk), cách embed hiện có: {embed_format_counts(es, current)}")
    print(f"Sẽ embed lại toàn bộ với cách embed '{current_embed_format()}' vào {dest}, rồi chuyển alias '{INDEX_NAME}' sang đó.")
    print(f"{current} sẽ {'bị xoá' if args.delete_old else 'được giữ lại để rollback'}. Hãy tắt backend trước khi chạy.")
    if not confirm("Tiếp tục?", args.yes):
        print("Đã huỷ.")
        return

    print("Đang load model embedding...")
    embed_model = load_embed_model()
    create_version(es, dest)
    done = reembed(
        es, embed_model, current, bulk_writer(es, dest),
        on_progress=lambda n: print(f"  đã embed {n}/{total} chunk", flush=True),
        on_retry=lambda size, err: print(f"  thiếu bộ nhớ, thử lại với {size} chunk mỗi lượt", flush=True),
    )

    new_total = count(es, dest)
    if new_total != total:
        sys.exit(f"Số chunk lệch ({current}: {total}, {dest}: {new_total}); alias giữ nguyên. "
                 f"Xoá {dest} rồi chạy lại: python backend/scripts/index_admin.py delete {dest}")

    es.indices.update_aliases(actions=[
        {"remove": {"index": current, "alias": INDEX_NAME}},
        {"add": {"index": dest, "alias": INDEX_NAME, "is_write_index": True}},
    ])
    print(f"Xong: đã embed lại {done} chunk; alias '{INDEX_NAME}' -> '{dest}'.")
    if args.delete_old:
        es.indices.delete(index=current)
        print(f"Đã xoá {current}.")
    else:
        print(f"Rollback nếu cần: python backend/scripts/index_admin.py switch {current}")


if __name__ == "__main__":
    main()
