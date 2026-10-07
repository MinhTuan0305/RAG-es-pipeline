"""
Turn an uploaded document (PDF, DOCX, HTML, Markdown) into a nested heading tree,
then flatten that tree into the Book -> Chapter -> Section -> Paragraph dict shape
used by the rest of the pipeline (chunking.py), using Docling to parse it.

Docling's heading levels: a Title item is the top level (Word "Title" style, or an
HTML <h1> / Markdown "#"), and section headers count down from it (Heading 1 /
<h2> / "##" = level 1, ...). Page headers/footers are left out (Docling puts them on
a separate "furniture" layer). Pictures are skipped for now.

Tree shape (what the review UI shows):
    {"title", "depth", "blocks": [block, ...], "children": [node, ...]}
where a block is one of
    {"type": "paragraph", "text": str}
    {"type": "table", "table_id": str, "caption": str|None, "header": [str],
     "rows": [[str]], "text": str (Markdown, caption included), "html": str}
"""

import hashlib
import html
import re
import threading
import unicodedata
import uuid
from pathlib import Path

from config import settings

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".html", ".htm", ".md"}

# A paragraph like "Bảng 2: Doanh thu theo quý" / "Table 3 - ..." right before a table
# is that table's caption even when the author didn't use a real caption style.
_CAPTION_RE = re.compile(r"^\s*(bảng|bang|table|tbl\.?)\s*[\dIVXLC]+\b", re.IGNORECASE)
_CAPTION_MAX_CHARS = 200

_converter = None
_converter_lock = threading.Lock()


def get_converter():
    """The shared Docling DocumentConverter, built on first use.

    Importing Docling is slow (tens of seconds cold), so it's imported here rather
    than at module import -- the app starts fast and only pays this on the first
    upload. Built once and reused: converters cache their loaded PDF models."""
    global _converter
    with _converter_lock:
        if _converter is None:
            from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import DocumentConverter, PdfFormatOption

            pdf_options = PdfPipelineOptions(
                do_table_structure=True,  # TableFormer: real rows/columns, not loose text lines
                do_ocr=True,              # only actually used where a page has no text layer
                generate_picture_images=False,
                accelerator_options=AcceleratorOptions(device=AcceleratorDevice(settings.model_device)),
            )
            _converter = DocumentConverter(
                allowed_formats=[InputFormat.PDF, InputFormat.DOCX, InputFormat.HTML, InputFormat.MD],
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options)},
            )
    return _converter


# ---------------------------------------------------------------- tables

def _clean_cell(text: str) -> str:
    """One Markdown-safe line: no newlines, and literal pipes escaped."""
    return " ".join((text or "").split()).replace("|", "\\|")


def markdown_row(cells) -> str:
    return "| " + " | ".join(cells) + " |"


def table_markdown(header, rows, caption=None) -> str:
    """Compact Markdown table (no column padding -- padding only costs tokens)."""
    lines = []
    if caption:
        lines += [caption, ""]
    lines.append(markdown_row(header))
    lines.append(markdown_row(["---"] * len(header)))
    lines += [markdown_row(r) for r in rows]
    return "\n".join(lines)


def _table_block(item, doc, caption, table_id):
    """TableItem -> table block, or None for an empty table.

    Header = the leading rows Docling marked as column headers; a multi-row header is
    merged per column ("2024 / Q1"). Without any marked header, the first row is used
    (Markdown requires one). Merged cells are repeated across the cells they span, so
    every row stays complete and readable on its own -- which matters once a long
    table is split into several chunks by row."""
    grid = [[_clean_cell(cell.text) for cell in row] for row in item.data.grid]
    flags = [any(cell.column_header for cell in row) for row in item.data.grid]
    grid = [row for row in grid if any(row)]
    if not grid:
        return None

    n_header = 0
    while n_header < len(flags) and flags[n_header]:
        n_header += 1
    n_header = max(1, min(n_header, len(grid)))

    header = []
    for col in range(len(grid[0])):
        parts = []
        for row in grid[:n_header]:
            if row[col] and row[col] not in parts:
                parts.append(row[col])
        header.append(" / ".join(parts))
    rows = grid[n_header:]

    return {
        "type": "table",
        "table_id": table_id,
        "caption": caption or None,
        "header": header,
        "rows": rows,
        "text": table_markdown(header, rows, caption),
        "html": item.export_to_html(doc=doc),
    }


# ---------------------------------------------------------------- tree

def _new_node(title, depth):
    return {"title": title, "depth": depth, "blocks": [], "children": []}


def docling_to_tree(doc, fallback_title: str) -> dict:
    """Build the heading tree from a DoclingDocument, in reading order.

    Standard "stack of open ancestors": a new heading pops every open heading at the
    same or deeper level (those are its siblings/descendants, not ancestors), then
    attaches under what's left on top. Paragraphs and tables attach to whatever
    heading is currently open, so each keeps its place in the hierarchy."""
    from docling_core.types.doc import (
        DocItemLabel, ListItem, PictureItem, SectionHeaderItem, TableItem, TextItem, TitleItem,
    )

    # Caption items Docling already linked to a table become that table's caption,
    # not a separate paragraph (otherwise the caption text would appear twice).
    table_caption_refs = {
        cap.cref for item, _ in doc.iterate_items() if isinstance(item, TableItem) for cap in item.captions
    }

    root = _new_node(fallback_title, -1)
    stack = [root]
    n_tables = 0

    for item, _level in doc.iterate_items():
        if isinstance(item, (TitleItem, SectionHeaderItem)):
            title = " ".join(item.text.split())
            if not title:
                continue
            depth = 0 if isinstance(item, TitleItem) else item.level
            while len(stack) > 1 and stack[-1]["depth"] >= depth:
                stack.pop()
            node = _new_node(title, depth)
            stack[-1]["children"].append(node)
            stack.append(node)

        elif isinstance(item, TableItem):
            blocks = stack[-1]["blocks"]
            caption = " ".join(item.caption_text(doc).split())
            if (
                not caption and blocks and blocks[-1]["type"] == "paragraph"
                and _CAPTION_RE.match(blocks[-1]["text"]) and len(blocks[-1]["text"]) <= _CAPTION_MAX_CHARS
            ):
                caption = blocks.pop()["text"]
            block = _table_block(item, doc, caption, table_id=f"t{n_tables + 1}")
            if block:
                n_tables += 1
                blocks.append(block)

        elif isinstance(item, PictureItem):
            continue  # images: a later step

        elif isinstance(item, TextItem):
            if item.self_ref in table_caption_refs or item.label == DocItemLabel.PAGE_HEADER:
                continue
            text = " ".join(item.text.split())
            if not text:
                continue
            if isinstance(item, ListItem):
                text = f"{item.marker or '-'} {text}"
            stack[-1]["blocks"].append({"type": "paragraph", "text": text})

    # A document wrapped in a single top heading (Word "Title" / one <h1> / one "#"
    # over everything) would otherwise become ONE chapter holding the whole document.
    # That heading is really the document's title: lift its contents up a level.
    if not root["blocks"] and len(root["children"]) == 1:
        only = root["children"][0]
        root.update(title=only["title"], blocks=only["blocks"], children=only["children"])

    return root


def tree_to_book(tree: dict, book_id: str, book_title: str, author: str, **doc_fields) -> dict:
    """Flatten the nested heading tree into Book -> Chapter -> Section -> Paragraph.

    doc_fields (e.g. source_file, uploaded_at, file_hash, content_hash) are copied onto
    every chunk by chunking.build_chunk_records.

    Each direct child of the root becomes one Chapter. Within a chapter's subtree,
    every node that has its own directly-attached blocks (at any depth) becomes one
    Section carrying that node's own heading as `section_title` -- so a deeply nested
    heading (e.g. "1.2.1.") still gets its own titled, searchable section, even though
    the chunk/index schema itself only has two levels (chapter/section). Content
    before the very first heading becomes an implicit first chapter ("Mở đầu").
    Table blocks keep all their fields; chunking.py turns them into table chunks.
    """
    def as_paragraphs(blocks):
        return [{"paragraph_index": i, **block} for i, block in enumerate(blocks)]

    chapters = []

    for top_node in tree["children"]:
        sections = []

        def collect(node, is_top):
            if node["blocks"]:
                sections.append({
                    "section_index": len(sections),
                    "section_title": None if is_top else node["title"],
                    "paragraphs": as_paragraphs(node["blocks"]),
                })
            for child in node["children"]:
                collect(child, is_top=False)

        collect(top_node, is_top=True)

        if not sections:
            sections = [{"section_index": 0, "section_title": None, "paragraphs": []}]

        chapters.append({
            "chapter_number": len(chapters) + 1,
            "chapter_title": top_node["title"],
            "sections": sections,
        })

    if tree["blocks"]:
        intro_chapter = {
            "chapter_number": 0,
            "chapter_title": "Mở đầu",
            "sections": [{"section_index": 0, "section_title": None, "paragraphs": as_paragraphs(tree["blocks"])}],
        }
        chapters.insert(0, intro_chapter)
        for i, ch in enumerate(chapters, start=1):
            ch["chapter_number"] = i

    return {
        "book_id": book_id,
        "book_title": book_title,
        "author": author,
        "front_matter": [],
        "chapters": chapters,
        "doc_fields": {k: v for k, v in doc_fields.items() if v is not None},
    }


# ---------------------------------------------------------------- fingerprints (duplicate detection)

def file_sha256(data: bytes) -> str:
    """Fingerprint of the exact file bytes: same file = same hash, whatever its name."""
    return hashlib.sha256(data).hexdigest()


def _normalize_for_hash(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).lower().split())


def tree_content_hash(tree) -> str:
    """Fingerprint of the document's text as parsed (headings, paragraphs and tables
    in reading order, normalized), so a re-saved or re-exported copy of the same
    document -- different bytes, same content -- is still recognized.

    Computed on the full parsed tree, before the user unticks any table, so the same
    file always gets the same hash. The root title is left out: when the document has
    no title of its own it's just the file name, which must not affect the hash."""
    digest = hashlib.sha256()
    for node in iter_nodes(tree):
        if node is not tree:
            digest.update(_normalize_for_hash(node["title"]).encode("utf-8") + b"\n")
        for block in node["blocks"]:
            digest.update(_normalize_for_hash(block["text"]).encode("utf-8") + b"\n")
    return digest.hexdigest()


def make_metadata(title: str) -> dict:
    """A fresh unique book_id from the title. Author is left blank -- there's no
    reliable generic way to guess it, so the review UI lets the user fill it in."""
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") or "document"
    book_id = f"{slug}-{uuid.uuid4().hex[:8]}"  # suffix guarantees uniqueness across uploads
    return {"book_id": book_id, "book_title": title, "author": ""}


def ingest_document(file_path, filename: str) -> dict:
    """End-to-end: Docling conversion -> heading tree -> guessed metadata.

    Returns {"tree": ..., "metadata": {"book_id", "book_title", "author"}, "content_hash"}.
    The tree (not yet a flattened book dict) is what the review UI displays, so the
    real parent/child heading structure stays visible; call tree_to_book() only after
    the user has reviewed the metadata/tables and confirmed.
    """
    result = get_converter().convert(str(file_path))
    tree = docling_to_tree(result.document, fallback_title=Path(filename).stem)
    metadata = make_metadata(tree["title"])
    return {"tree": tree, "metadata": metadata, "content_hash": tree_content_hash(tree)}


# ---------------------------------------------------------------- tree helpers (review UI)

def iter_nodes(node):
    yield node
    for child in node["children"]:
        yield from iter_nodes(child)


def tree_stats(tree) -> dict:
    nodes = list(iter_nodes(tree))
    blocks = [b for n in nodes for b in n["blocks"]]
    return {
        "headings": len(nodes) - 1,
        "paragraphs": sum(b["type"] == "paragraph" for b in blocks),
        "tables": sum(b["type"] == "table" for b in blocks),
    }


def tree_tables(tree):
    """[(path, table_block), ...] in reading order; path = heading titles to the table."""
    found = []

    def walk(node, path):
        for b in node["blocks"]:
            if b["type"] == "table":
                found.append((path, b))
        for child in node["children"]:
            walk(child, path + [child["title"]])

    walk(tree, [])
    return found


def drop_tables(tree, table_ids: set) -> dict:
    """Copy of the tree without the given tables (the ones unticked in the review UI)."""
    return {
        **tree,
        "blocks": [b for b in tree["blocks"] if not (b["type"] == "table" and b["table_id"] in table_ids)],
        "children": [drop_tables(c, table_ids) for c in tree["children"]],
    }


def tree_html(tree) -> str:
    """The heading tree as nested, collapsible <details> HTML (Streamlit can't nest
    expanders). Top-level sections start open, deeper ones collapsed; every heading
    shows how many paragraphs/tables sit directly under it, and tables are listed at
    their actual position. Colors inherit from the page so it works in dark mode."""
    def badge(node):
        n_par = sum(b["type"] == "paragraph" for b in node["blocks"])
        n_tab = sum(b["type"] == "table" for b in node["blocks"])
        parts = []
        if n_par:
            parts.append(f"{n_par} đoạn")
        if n_tab:
            parts.append(f"{n_tab} bảng")
        if node["children"]:
            parts.append(f"{len(node['children'])} mục con")
        return f"<span class='dt-meta'>{' · '.join(parts) or 'trống'}</span>"

    def table_lines(node):
        out = []
        for b in node["blocks"]:
            if b["type"] == "table":
                name = html.escape(b["caption"] or f"Bảng {b['table_id'][1:]}")
                dims = f"{len(b['rows'])} hàng × {len(b['header'])} cột"
                out.append(f"<div class='dt-table'>Bảng: {name} <span class='dt-meta'>{dims}</span></div>")
        return "".join(out)

    def render(node, level):
        title = html.escape(node["title"])
        body = table_lines(node) + "".join(render(c, level + 1) for c in node["children"])
        if not body:
            return f"<div class='dt-leaf'>{title} {badge(node)}</div>"
        open_attr = " open" if level == 0 else ""
        return (
            f"<details{open_attr}><summary>{title} {badge(node)}</summary>"
            f"<div class='dt-children'>{body}</div></details>"
        )

    intro = ""
    if tree["blocks"]:
        intro = (
            f"<div class='dt-leaf'><i>(Phần mở đầu, trước tiêu đề đầu tiên)</i> {badge(tree)}</div>"
            + table_lines(tree)
        )
    style = """<style>
.dt-tree { font-size: 0.92rem; line-height: 1.7; }
.dt-tree summary { cursor: pointer; }
.dt-tree summary::marker { color: currentColor; opacity: 0.5; }
.dt-children { margin-left: 0.6rem; padding-left: 0.9rem; border-left: 1px solid rgba(128,128,128,0.35); }
.dt-leaf { padding-left: 1.05rem; }
.dt-table { padding-left: 1.05rem; font-style: italic; }
.dt-meta { opacity: 0.6; font-size: 0.82rem; font-style: normal; margin-left: 0.35rem; }
</style>"""
    return style + "<div class='dt-tree'>" + intro + "".join(render(c, 0) for c in tree["children"]) + "</div>"
