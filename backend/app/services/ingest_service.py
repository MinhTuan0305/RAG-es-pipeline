"""
Adding a document, in two steps:

1. parse_upload(): duplicate check on the file bytes, Docling parse, duplicate check
   on the parsed text. The parsed tree is kept in a short-lived in-memory store under
   a parse_id while the user reviews it.
2. start_index_job(): background job -- chunk, embed (with progress and automatic
   batch shrinking on out-of-memory), index, and when replacing an older copy delete
   it only after the new one is fully indexed.

Everything here reuses the pipeline functions; nothing about parsing/chunking/
embedding is reimplemented.
"""

import logging
import os
import tempfile
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.core.chunking import build_chunk_records
from app.core.document_ingest import (
    SUPPORTED_EXTENSIONS,
    drop_tables,
    file_sha256,
    ingest_document,
    tree_stats,
    tree_tables,
    tree_to_book,
)
from app.core.errors import ApiError, AppError
from app.core.rag_pipeline import (
    delete_document,
    embed_chunks_batch,
    ensure_index_exists,
    find_document_by_hash,
    get_document,
    index_chunks,
)
from app.schemas import JobDone, ParseResult, ParseReview, Progress, Retry
from app.services.jobs import JobManager

logger = logging.getLogger(__name__)

PARSE_TTL = timedelta(hours=1)


# ---------------------------------------------------------------- parse store

class ParseStore:
    """Parsed-but-not-yet-indexed documents, by parse_id, for PARSE_TTL."""

    def __init__(self):
        self._items: dict[str, dict] = {}
        self._lock = threading.Lock()

    def put(self, record: dict) -> str:
        parse_id = uuid.uuid4().hex
        with self._lock:
            self._cleanup_locked()
            self._items[parse_id] = {**record, "created_at": datetime.now(timezone.utc)}
        return parse_id

    def get(self, parse_id: str) -> dict:
        with self._lock:
            self._cleanup_locked()
            record = self._items.get(parse_id)
        if record is None:
            raise ApiError(
                404, "Không tìm thấy bản phân tích tài liệu (đã hết hạn hoặc đã được thêm).",
                "Chọn lại file và bấm xử lý lại.",
            )
        return record

    def discard(self, parse_id: str):
        with self._lock:
            self._items.pop(parse_id, None)

    def _cleanup_locked(self):
        cutoff = datetime.now(timezone.utc) - PARSE_TTL
        for key in [k for k, v in self._items.items() if v["created_at"] < cutoff]:
            del self._items[key]


# ---------------------------------------------------------------- review payload

def _slim_tree(node) -> dict:
    """The heading tree without paragraph texts (the review UI only needs counts),
    which keeps the response small even for long documents."""
    return {
        "title": node["title"],
        "depth": node["depth"],
        "n_paragraphs": sum(b["type"] == "paragraph" for b in node["blocks"]),
        "table_ids": [b["table_id"] for b in node["blocks"] if b["type"] == "table"],
        "children": [_slim_tree(c) for c in node["children"]],
    }


def build_review(record: dict) -> ParseReview:
    tree = record["tree"]
    return ParseReview(
        metadata=record["metadata"],
        stats=tree_stats(tree),
        tree=_slim_tree(tree),
        tables=[
            {
                "table_id": t["table_id"], "caption": t["caption"], "path": path,
                "n_rows": len(t["rows"]), "n_cols": len(t["header"]), "markdown": t["text"],
            }
            for path, t in tree_tables(tree)
        ],
    )


# ---------------------------------------------------------------- step 1: parse

def _resolve_replace(es, replace_book_id):
    if not replace_book_id:
        return None
    doc = get_document(es, replace_book_id)
    if doc is None:
        raise ApiError(404, "Không tìm thấy tài liệu cần thay thế (có thể đã bị xoá).")
    return doc


def parse_upload(es, store: ParseStore, filename: str, data: bytes, replace_book_id: str | None = None) -> ParseResult:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ApiError(
            400, f"Không hỗ trợ định dạng '{suffix or filename}'.",
            "Hỗ trợ: " + ", ".join(sorted(SUPPORTED_EXTENSIONS)),
        )
    if not data:
        raise ApiError(400, "File rỗng.")

    replace = _resolve_replace(es, replace_book_id)

    def is_the_replaced_one(dup):
        return replace is not None and dup["book_id"] == replace["book_id"]

    # Same bytes already indexed: stop before the (slow) Docling parse.
    file_hash = file_sha256(data)
    dup = find_document_by_hash(es, file_hash=file_hash)
    if dup and not is_the_replaced_one(dup):
        return ParseResult(status="duplicate", match="file", existing=dup)

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        result = ingest_document(tmp_path, filename)
    finally:
        try:
            os.remove(tmp_path)  # Docling only needs the file during parsing
        except OSError:
            pass

    tree = result["tree"]
    if not tree["children"] and not tree["blocks"]:
        raise ApiError(422, "Không tìm thấy nội dung văn bản nào trong tài liệu này.")

    metadata = dict(result["metadata"])
    if replace:
        metadata["book_title"] = replace["book_title"]  # a replacement keeps the document's name by default
    record = {
        "tree": tree,
        "metadata": metadata,
        "file_hash": file_hash,
        "content_hash": result["content_hash"],
        "source_file": filename,
    }
    parse_id = store.put(record)
    review = build_review(record)

    # Same text already indexed (a re-saved/re-exported copy): parsed anyway, so the
    # user can go straight to review if they choose to replace the old copy.
    dup = find_document_by_hash(es, content_hash=result["content_hash"])
    if dup and not is_the_replaced_one(dup):
        return ParseResult(status="duplicate", match="content", existing=dup, parse_id=parse_id, review=review)
    return ParseResult(status="ready", parse_id=parse_id, review=review)


# ---------------------------------------------------------------- step 2: index (background job)

def start_index_job(
    jobs: JobManager, store: ParseStore, es, embed_model,
    parse_id: str, title: str, author: str, excluded_table_ids: list[str], replace_book_id: str | None,
):
    record = store.get(parse_id)  # 404 if expired/unknown -- before starting a job
    replace = _resolve_replace(es, replace_book_id)
    metadata = {**record["metadata"], "book_title": title.strip(), "author": author.strip()}

    def work(emit) -> JobDone:
        def progress(stage, value, text, done=None, total=None):
            emit(Progress(stage=stage, value=round(min(value, 1.0), 4), text=text, done=done, total=total))

        progress("prepare", 0.03, "Đang dựng cấu trúc chương/mục...")
        book = tree_to_book(
            drop_tables(record["tree"], set(excluded_table_ids)), **metadata,
            source_file=record["source_file"],
            uploaded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            file_hash=record["file_hash"],
            content_hash=record["content_hash"],
        )

        progress("chunk", 0.06, "Đang chia nhỏ văn bản...")
        chunks = build_chunk_records(book)
        n_table_chunks = sum(c["type"] == "table" for c in chunks)

        # Embedding is by far the slowest step: it gets 10% -> 90% of the bar.
        progress("embed", 0.10, f"Đang embedding: 0/{len(chunks)} chunk", 0, len(chunks))
        embed_chunks_batch(
            embed_model, chunks,
            on_progress=lambda done, total: progress(
                "embed", 0.10 + 0.80 * done / total, f"Đang embedding: {done}/{total} chunk", done, total,
            ),
            on_retry=lambda size, err: emit(Retry(
                batch_size=size, text=f"Thiếu bộ nhớ, đã giải phóng bộ nhớ và thử lại với {size} chunk mỗi lượt...",
            )),
        )

        progress("index", 0.92, "Đang index vào Elasticsearch...")
        ensure_index_exists(es)
        success_count, errors = index_chunks(es, chunks)
        if errors:
            # Don't leave half a document searchable; the old copy (if any) is untouched.
            delete_document(es, metadata["book_id"])
            raise AppError(
                f"{len(errors)} chunk lỗi khi index vào Elasticsearch, đã huỷ thêm tài liệu.",
                ("Tài liệu cũ được giữ nguyên. " if replace else "") + f"Lỗi đầu tiên: {str(errors[0])[:300]}",
            )

        replaced = None
        if replace:
            progress("replace", 0.97, "Đang xoá bản cũ...")
            replaced = delete_document(es, replace["book_id"])

        store.discard(parse_id)
        progress("done", 1.0, f"Hoàn tất: {success_count} chunk")
        return JobDone(
            book_id=metadata["book_id"], book_title=metadata["book_title"],
            chunks=success_count, table_chunks=n_table_chunks, replaced_chunks=replaced,
        )

    return jobs.start("index", work)
