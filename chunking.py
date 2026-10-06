import re

from transformers import AutoTokenizer

from config import settings
from document_ingest import markdown_row, table_markdown

# Token counts must use the embedding model's own tokenizer, so it follows EMBED_MODEL.
TOKENIZER_NAME = settings.embed_model
TARGET_CHUNK_TOKENS = settings.chunk_target_tokens  # soft target: stop accumulating paragraphs once exceeded
MAX_CHUNK_TOKENS = settings.chunk_max_tokens        # hard cap: only a single oversized paragraph is force-split above this
OVERLAP_RATIO = settings.chunk_overlap_ratio        # share of TARGET_CHUNK_TOKENS carried forward into the next chunk

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


def split_table_atom(atom, target_tokens=TARGET_CHUNK_TOKENS, max_tokens=MAX_CHUNK_TOKENS, context_title=None):
    """A table becomes one chunk, or -- if longer than max_tokens -- several, split
    between rows, each repeating the caption and header row so every part is still a
    readable, self-describing table. A table with no caption gets its heading as a
    caption line instead: a bare grid of numbers says little to the embedding model."""
    caption = atom.get("caption") or (f"Bảng trong mục: {context_title}" if context_title else None)
    header, rows = atom["header"], atom["rows"]
    base = {k: v for k, v in atom.items() if k not in ("text", "token_count", "header", "rows")}

    full_text = table_markdown(header, rows, caption)
    full_tokens = count_tokens(full_text)
    if full_tokens <= max_tokens:
        return [{**base, "text": full_text, "token_count": full_tokens, "is_overlap": False}]

    fixed_tokens = count_tokens(table_markdown(header, [], caption)) + 8  # + the " (phần i/n)" suffix
    groups, current, current_tokens = [], [], fixed_tokens
    for row in rows:
        n = count_tokens(markdown_row(row))
        if current and current_tokens + n > target_tokens:
            groups.append(current)
            current, current_tokens = [], fixed_tokens
        current.append(row)
        current_tokens += n
    if current:
        groups.append(current)

    parts = []
    for i, group in enumerate(groups, start=1):
        part_caption = f"{caption} (phần {i}/{len(groups)})" if caption else f"(Bảng, phần {i}/{len(groups)})"
        text = table_markdown(header, group, part_caption)
        parts.append({**base, "text": text, "token_count": count_tokens(text),
                      "is_overlap": False, "paragraph_part": f"{i}/{len(groups)}"})
    return parts


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


def chunk_section(paragraphs, target_tokens=TARGET_CHUNK_TOKENS, max_tokens=MAX_CHUNK_TOKENS,
                  overlap_ratio=OVERLAP_RATIO, context_title=None):
    """Greedily group a section's paragraphs into overlapping, token-budgeted chunks.
    Tables are emitted as chunks of their own (see split_table_atom) and never used as
    overlap; prose after a table starts a fresh chunk."""
    atoms = [p if p["type"] == "table" else {**p, "token_count": count_tokens(p["text"])} for p in paragraphs]
    chunks, current, current_tokens = [], [], 0

    def close_chunk():
        nonlocal current, current_tokens
        if current:
            chunks.append(current)
        current, current_tokens = [], 0

    for atom in atoms:
        if atom["type"] == "table":
            close_chunk()
            for part in split_table_atom(atom, target_tokens, max_tokens, context_title):
                chunks.append([part])
            continue

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
            context_title = " > ".join(t for t in (ch["chapter_title"], sec.get("section_title")) if t)
            atom_chunks = chunk_section(sec["paragraphs"], target_tokens, max_tokens, overlap_ratio, context_title)
            for chunk_idx, atoms in enumerate(atom_chunks):
                text = " ".join(a["text"] for a in atoms)
                if atoms[0]["type"] == "table":
                    chunk_type = "table"
                elif all(a["type"] == "verse" for a in atoms):
                    chunk_type = "verse"
                else:
                    chunk_type = "paragraph"
                record = {
                    "chunk_id": f"{book['book_id']}-ch{ch['chapter_number']:02d}-s{sec['section_index']:02d}-c{chunk_idx:03d}",
                    "book_id": book["book_id"], "book_title": book["book_title"], "author": book["author"],
                    "chapter_number": ch["chapter_number"], "chapter_title": ch["chapter_title"],
                    "section_index": sec["section_index"], "section_title": sec.get("section_title"),
                    "chunk_index": chunk_idx, "type": chunk_type,
                    "paragraph_ids": [a["paragraph_index"] for a in atoms],
                    "new_paragraph_ids": [a["paragraph_index"] for a in atoms if not a["is_overlap"]],
                    "text": text, "token_count": sum(a["token_count"] for a in atoms),
                }
                if chunk_type == "table":
                    # The original table (merged cells intact) for display; the
                    # Markdown in `text` is what gets embedded and sent to the LLM.
                    record["table_html"] = atoms[0]["html"]
                records.append(record)

    # Global reading-order stitching, across the whole book.
    for i, rec in enumerate(records):
        rec["global_index"] = i
        rec["prev_chunk_id"] = records[i - 1]["chunk_id"] if i > 0 else None
        rec["next_chunk_id"] = records[i + 1]["chunk_id"] if i < len(records) - 1 else None

    return records
