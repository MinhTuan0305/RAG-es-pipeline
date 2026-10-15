import type { Components } from "react-markdown";

import type { SourceRecord } from "../api/types";

const CITE_PREFIX = "#cite-";
// [n] or [n, m] (1-3 digits each), not followed by "(" -- that would be a Markdown
// link. Lists only reach the screen while streaming: the final answer from the
// backend is already normalized to [n][m].
const CITATION_RE = /\[(\d{1,3}(?:\s*[,;]\s*\d{1,3})*)\](?!\()/g;
// Code is left untouched: fenced blocks and inline code spans.
const CODE_RE = /(```[\s\S]*?```|`[^`\n]*`)/g;

/**
 * Turn the answer's [n] citations into Markdown links (#cite-n) so the Markdown
 * renderer hands them to <CitationLink> -- but only numbers that match a known source,
 * so a citation whose source hasn't arrived yet (or never will) stays plain text.
 */
export function linkCitations(markdown: string, known: Set<number>): string {
  return markdown
    .split(CODE_RE)
    .map((part, i) =>
      i % 2 === 1
        ? part
        : part.replace(CITATION_RE, (match, list: string) => {
            const numbers = list.split(/\s*[,;]\s*/).map(Number);
            if (!numbers.every((n) => known.has(n))) return match;
            return numbers.map((n) => `[${n}](${CITE_PREFIX}${n})`).join("");
          }),
    )
    .join("");
}

const PREVIEW_CHARS = 220;

/** Markdown `a` renderer: citation links become badges with a hover card; other
 * links open in a new tab. */
export function citationComponents(sources: Map<number, SourceRecord>, onOpen: (n: number) => void): Components {
  return {
    a({ href, children }) {
      if (href?.startsWith(CITE_PREFIX)) {
        const n = Number(href.slice(CITE_PREFIX.length));
        const source = sources.get(n);
        return (
          <sup className="cite">
            <button type="button" className="cite-badge" onClick={() => onOpen(n)} aria-label={`Nguồn ${n}`}>
              {n}
            </button>
            {source && (
              <span className="cite-card" role="tooltip">
                <strong>{source.book_title}</strong>
                <span className="muted"> — {source.location}{source.type === "table" ? " · bảng" : ""}</span>
                <span className="cite-card-text">
                  {source.text.length > PREVIEW_CHARS ? `${source.text.slice(0, PREVIEW_CHARS)}...` : source.text}
                </span>
                <span className="muted small">Bấm để xem trong danh sách nguồn</span>
              </span>
            )}
          </sup>
        );
      }
      return (
        <a href={href} target="_blank" rel="noreferrer">
          {children}
        </a>
      );
    },
  };
}
