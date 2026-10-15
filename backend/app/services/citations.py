"""
Inline citations: the answer cites passages as [n], where n is the passage's number
in this question's search results (numbered 1, 2, 3, ... across all searches; see
search_documents in app/core/langgraph_agent.py).

The model doesn't always follow the format exactly, so its answer is cleaned up:
"[1, 3]" / "[1;3]" become "[1][3]", and numbers that match no passage (made up) are
removed. Markdown links "[text](url)" and 4-digit brackets like "[2024]" are left alone.
"""

import re

# Optional leading whitespace, then [n] or [n, m, ...] not followed by "(" (a link).
_CITATION_RE = re.compile(r"([ \t]*)\[(\d{1,3}(?:\s*[,;]\s*\d{1,3})*)\](?!\()")
_SPLIT_RE = re.compile(r"\s*[,;]\s*")


def clean_citations(text: str, valid: set[int]) -> tuple[str, list[int], list[int]]:
    """Normalize the citations in `text` and drop the invalid ones.
    Returns (cleaned text, cited numbers in order of first use, invalid numbers)."""
    cited: list[int] = []
    invalid: list[int] = []

    def replace(match: re.Match) -> str:
        keep = []
        for n in (int(x) for x in _SPLIT_RE.split(match.group(2))):
            if n in valid:
                if n not in keep:
                    keep.append(n)
                if n not in cited:
                    cited.append(n)
            else:
                invalid.append(n)
        # Dropping a citation drops the space before it too ("foo [7]." -> "foo.").
        return match.group(1) + "".join(f"[{n}]" for n in keep) if keep else ""

    return _CITATION_RE.sub(replace, text), cited, invalid


def strip_citations(text: str) -> str:
    """The text without any citation markers -- for an earlier answer sent back to
    the model as history, where [2] meant a passage of THAT question's searches."""
    return _CITATION_RE.sub("", text)
