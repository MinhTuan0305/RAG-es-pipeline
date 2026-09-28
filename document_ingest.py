"""
Turn an arbitrary uploaded document (PDF, DOCX, HTML, Markdown) into a nested
heading tree, then flatten that tree into the Book -> Chapter -> Section ->
Paragraph dict shape used by the rest of the pipeline (chunking.py), using the
`unstructured` library to detect headings/paragraphs.

Per-format detection mechanism (what `unstructured` actually looks at):
- DOCX: reads the Word paragraph style ("Heading 1", "Heading 2", ...) directly from
  the file's XML -- reliable IF the author used Word's real heading styles. The
  heading level number becomes `metadata.category_depth` (0 = Heading 1, 1 = Heading
  2, ...), which is exactly what lets us reconstruct real nesting (e.g. "1.2." as a
  child of "1." and "1.2.1"/"1.2.2" as children of "1.2.").
- HTML/Markdown: reads markup tags directly (<h1>-<h6> -> depth 0-5, <p>, <li>).
- PDF: "fast" strategy reads the embedded text layer + font size/boldness/position
  to guess headings (works for normal, non-scanned PDFs, no extra system deps); a
  scanned/image PDF falls back to a layout model + OCR (Tesseract) -- needs
  Tesseract + Poppler installed as system binaries on Windows. PDF heading *depth*
  detection is less reliable than DOCX/HTML; when depth isn't available every Title
  is treated as depth 0 (flat chapters -- same behavior as before this fix).
- Plain TXT: no formatting signal at all -- headings are only weak heuristic
  guesses, which is why the Gatsby book itself uses a hand-written parser instead.
"""

import re
import uuid
from pathlib import Path

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".html", ".htm", ".md"}

# Element categories from `unstructured` that should never become paragraph content.
_SKIP_CATEGORIES = {"Header", "Footer", "PageBreak", "Image", "Table", "FigureCaption"}


def partition_any(file_path):
    """Partition a document into a list of `unstructured` Element objects.

    Requires: pip install "unstructured[pdf,docx,md]"
    For scanned/image PDFs specifically (no embedded text layer), also requires the
    Tesseract OCR engine and Poppler installed as system binaries (Windows: download
    installers separately, e.g. https://github.com/UB-Mannheim/tesseract/wiki and
    https://github.com/oschwartz10612/poppler-windows, then add both to PATH) --
    text-based PDFs, DOCX, HTML, and Markdown do NOT need either of those.
    """
    from unstructured.partition.auto import partition

    return partition(filename=str(file_path))


def guess_metadata(elements, filename: str) -> dict:
    """Best-effort title guess (first Title element, else the filename) and a fresh
    unique book_id. Author is left blank -- there's no reliable generic way to guess
    it, so the review UI lets the user fill it in by hand before indexing."""
    title = Path(filename).stem
    for el in elements:
        text = (getattr(el, "text", "") or "").strip()
        if getattr(el, "category", None) == "Title" and text:
            title = text
            break

    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") or "document"
    book_id = f"{slug}-{uuid.uuid4().hex[:8]}"  # suffix guarantees uniqueness across uploads

    return {"book_id": book_id, "book_title": title, "author": ""}


def _get_depth(el) -> int:
    """0 = top-level heading, 1 = sub-heading, etc. Falls back to 0 (flat) when the
    source format/strategy doesn't expose category_depth."""
    metadata = getattr(el, "metadata", None)
    depth = getattr(metadata, "category_depth", None)
    return depth if isinstance(depth, int) else 0


def elements_to_tree(elements, book_title: str) -> dict:
    """Build a nested tree {"title", "depth", "paragraphs": [str, ...], "children": [...]}
    from the flat `unstructured` element list, using each Title's heading depth.

    Uses the standard "stack of open ancestors" algorithm: when a new heading arrives,
    pop any currently-open heading whose depth is >= the new one's (those are siblings
    or deeper, not ancestors), then attach the new heading under whatever remains on
    top of the stack -- that's its nearest actual ancestor.
    """
    root = {"title": book_title, "depth": -1, "paragraphs": [], "children": []}
    stack = [root]

    for el in elements:
        text = (getattr(el, "text", "") or "").strip()
        if not text:
            continue
        category = getattr(el, "category", None)
        if category in _SKIP_CATEGORIES:
            continue

        if category == "Title":
            depth = _get_depth(el)
            node = {"title": text, "depth": depth, "paragraphs": [], "children": []}
            while len(stack) > 1 and stack[-1]["depth"] >= depth:
                stack.pop()
            stack[-1]["children"].append(node)
            stack.append(node)
            continue

        stack[-1]["paragraphs"].append(text)

    return root


def tree_to_book(tree: dict, book_id: str, book_title: str, author: str) -> dict:
    """Flatten the nested heading tree into Book -> Chapter -> Section -> Paragraph.

    Each direct child of the root becomes one Chapter. Within a chapter's subtree,
    every node that has its own directly-attached paragraphs (at any depth) becomes
    one Section carrying that node's own heading as `section_title` -- so a deeply
    nested heading (e.g. "1.2.1.") still gets its own titled, searchable section,
    even though the chunk/index schema itself only has two levels (chapter/section).
    Any text that appears before the very first heading becomes an implicit
    chapter 0 ("Mở đầu" / front text).
    """
    chapters = []

    for top_node in tree["children"]:
        sections = []

        def collect(node, is_top):
            if node["paragraphs"]:
                sections.append({
                    "section_index": len(sections),
                    "section_title": None if is_top else node["title"],
                    "paragraphs": [
                        {"paragraph_index": i, "type": "paragraph", "text": t}
                        for i, t in enumerate(node["paragraphs"])
                    ],
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

    if tree["paragraphs"]:
        intro_chapter = {
            "chapter_number": 0,
            "chapter_title": "Mở đầu",
            "sections": [{
                "section_index": 0,
                "section_title": None,
                "paragraphs": [
                    {"paragraph_index": i, "type": "paragraph", "text": t}
                    for i, t in enumerate(tree["paragraphs"])
                ],
            }],
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
    }


def ingest_document(file_path, filename: str) -> dict:
    """End-to-end: partition -> guess metadata -> build the heading tree.

    Returns {"tree": ..., "metadata": {"book_id", "book_title", "author"}}. The tree
    (not yet a flattened book dict) is what the review UI should display, so the
    real parent/child heading structure stays visible; call tree_to_book() only after
    the user has reviewed/edited the metadata and confirmed.
    """
    elements = partition_any(file_path)
    metadata = guess_metadata(elements, filename)
    tree = elements_to_tree(elements, metadata["book_title"])
    return {"tree": tree, "metadata": metadata}
