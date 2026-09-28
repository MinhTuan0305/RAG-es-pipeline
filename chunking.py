"""
Generic paragraph -> token-budgeted, overlapping chunk logic.

Operates on any "book" dict shaped like:
    {
        "book_id": str, "book_title": str, "author": str,
        "front_matter": [{"type": "paragraph"|"verse", "text": str}, ...],
        "chapters": [
            {
                "chapter_number": int, "chapter_title": str,
                "sections": [
                    {"section_index": int, "paragraphs": [{"paragraph_index": int, "type": str, "text": str}, ...]},
                    ...
                ],
            },
            ...
        ],
    }

This shape is source-agnostic -- the same chunker is used whether the book dict came
from the hand-written Gutenberg-text extraction (ingestion_pipeline.ipynb) or from the
`unstructured`-based document upload feature (document_ingest.py).
"""

import re

from transformers import AutoTokenizer

TOKENIZER_NAME = "BAAI/bge-m3"
TARGET_CHUNK_TOKENS = 450   # soft target: stop accumulating paragraphs once exceeded
MAX_CHUNK_TOKENS = 600      # hard cap: only a single oversized paragraph is force-split above this
OVERLAP_RATIO = 0.15        # ~15% of TARGET_CHUNK_TOKENS carried forward into the next chunk

SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"‘’“”])')

_tokenizer = None


def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    return _tokenizer


def count_tokens(text: str) -> int:
    return len(get_tokenizer().encode(text, add_special_tokens=False))


def split_oversized_atom(atom, target_tokens=TARGET_CHUNK_TOKENS):
    """Force-split a single paragraph that exceeds MAX_CHUNK_TOKENS, at sentence boundaries."""
    sentences = SENTENCE_SPLIT_RE.split(atom["text"]) or [atom["text"]]
    groups = []
    current_sents, current_tokens = [], 0
    for sent in sentences:
        n = count_tokens(sent)
        if current_sents and current_tokens + n > target_tokens:
            groups.append((current_sents, current_tokens))
            current_sents, current_tokens = [], 0
        current_sents.append(sent)
        current_tokens += n
    if current_sents:
        groups.append((current_sents, current_tokens))

    total_parts = len(groups)
    base = {k: v for k, v in atom.items() if k not in ("text", "token_count")}
    return [
        {**base, "text": " ".join(sents), "token_count": tok_count,
         "is_overlap": False, "paragraph_part": f"{i}/{total_parts}"}
        for i, (sents, tok_count) in enumerate(groups, start=1)
    ]


def take_overlap_tail(atom_list, overlap_ratio=OVERLAP_RATIO, target_tokens=TARGET_CHUNK_TOKENS):
    """Pull trailing paragraphs from the previous chunk to seed the next chunk's overlap."""
    overlap_budget = max(1, round(target_tokens * overlap_ratio))
    tail, running = [], 0
    for atom in reversed(atom_list):
        if running >= overlap_budget and tail:
            break
        tail.insert(0, {**atom, "is_overlap": True})
        running += atom["token_count"]
    return tail


def chunk_section(paragraphs, target_tokens=TARGET_CHUNK_TOKENS, max_tokens=MAX_CHUNK_TOKENS, overlap_ratio=OVERLAP_RATIO):
    """Greedily group a section's paragraphs into overlapping, token-budgeted chunks."""
    atoms = [{**p, "token_count": count_tokens(p["text"])} for p in paragraphs]
    chunks, current, current_tokens = [], [], 0

    def close_chunk():
        nonlocal current, current_tokens
        if current:
            chunks.append(current)
        current, current_tokens = [], 0

    for atom in atoms:
        if atom["token_count"] > max_tokens:
            close_chunk()
            for sub_atom in split_oversized_atom(atom, target_tokens):
                chunks.append([sub_atom])
            continue

        if current and current_tokens + atom["token_count"] > target_tokens:
            close_chunk()
            current = take_overlap_tail(chunks[-1], overlap_ratio, target_tokens)
            current_tokens = sum(a["token_count"] for a in current)

        current.append({**atom, "is_overlap": False})
        current_tokens += atom["token_count"]

    close_chunk()
    return chunks


def build_chunk_records(book, target_tokens=TARGET_CHUNK_TOKENS, max_tokens=MAX_CHUNK_TOKENS, overlap_ratio=OVERLAP_RATIO):
    records = []

    # Front matter: each block is already tiny and self-contained -- one chunk each, no splitting/overlap.
    for i, block in enumerate(book["front_matter"]):
        records.append({
            "chunk_id": f"{book['book_id']}-front{i:02d}",
            "book_id": book["book_id"], "book_title": book["book_title"], "author": book["author"],
            "chapter_number": 0, "chapter_title": "Front Matter", "section_index": 0, "section_title": None,
            "chunk_index": i, "type": block["type"], "paragraph_ids": [i], "new_paragraph_ids": [i],
            "text": block["text"], "token_count": count_tokens(block["text"]),
        })

    for ch in book["chapters"]:
        for sec in ch["sections"]:
            atom_chunks = chunk_section(sec["paragraphs"], target_tokens, max_tokens, overlap_ratio)
            for chunk_idx, atoms in enumerate(atom_chunks):
                text = " ".join(a["text"] for a in atoms)
                chunk_type = "verse" if all(a["type"] == "verse" for a in atoms) else "paragraph"
                records.append({
                    "chunk_id": f"{book['book_id']}-ch{ch['chapter_number']:02d}-s{sec['section_index']:02d}-c{chunk_idx:03d}",
                    "book_id": book["book_id"], "book_title": book["book_title"], "author": book["author"],
                    "chapter_number": ch["chapter_number"], "chapter_title": ch["chapter_title"],
                    "section_index": sec["section_index"], "section_title": sec.get("section_title"),
                    "chunk_index": chunk_idx, "type": chunk_type,
                    "paragraph_ids": [a["paragraph_index"] for a in atoms],
                    "new_paragraph_ids": [a["paragraph_index"] for a in atoms if not a["is_overlap"]],
                    "text": text, "token_count": sum(a["token_count"] for a in atoms),
                })

    # Global reading-order stitching, across the whole book.
    for i, rec in enumerate(records):
        rec["global_index"] = i
        rec["prev_chunk_id"] = records[i - 1]["chunk_id"] if i > 0 else None
        rec["next_chunk_id"] = records[i + 1]["chunk_id"] if i < len(records) - 1 else None

    return records
