import { useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";

import { api, toApiError, type ApiError } from "../api/client";
import { watchJob } from "../api/jobEvents";
import type { DocumentSummary, JobDone, ParseResult, ParseReview, TableInfo, TreeNode } from "../api/types";
import { useAppData } from "../appData";
import { Collapsible, ConfirmButton, ErrorNotice, Markdown, ProgressBar } from "../components/ui";
import { documentMetaLine } from "../format";

const ACCEPT = ".pdf,.docx,.html,.htm,.md";

export function DocumentsPage() {
  const [params, setParams] = useSearchParams();
  const tab = params.get("tab") === "library" ? "library" : "add";
  const { documents } = useAppData();

  const selectTab = (next: "add" | "library") =>
    setParams((p) => {
      const copy = new URLSearchParams(p);
      if (next === "add") copy.delete("tab");
      else copy.set("tab", next);
      return copy;
    });

  return (
    <div className="page">
      <h2>Quản lý tài liệu</h2>
      <div className="tabs" role="tablist">
        <button role="tab" aria-selected={tab === "add"} onClick={() => selectTab("add")}>
          Thêm tài liệu
        </button>
        <button role="tab" aria-selected={tab === "library"} onClick={() => selectTab("library")}>
          Tài liệu trong kho ({documents.length})
        </button>
      </div>
      {tab === "add" ? <AddDocument /> : <Library />}
    </div>
  );
}

// ---------------------------------------------------------------- add a document
//
//   choose file -> [parse] -> duplicate? -> review -> [index] -> job progress (SSE)
//
// The running job's id is kept in the URL (?job=...), so a page reload picks the
// progress up again instead of losing it.

type Stage =
  | { kind: "pick" }
  | { kind: "parsing" }
  | { kind: "duplicate"; result: ParseResult; file: File }
  | { kind: "review"; parseId: string; review: ParseReview; replace: DocumentSummary | null };

function AddDocument() {
  const [params, setParams] = useSearchParams();
  const jobId = params.get("job");
  const [stage, setStage] = useState<Stage>({ kind: "pick" });
  const [file, setFile] = useState<File | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [fileInputKey, setFileInputKey] = useState(0);

  const setJob = (id: string | null) =>
    setParams((p) => {
      const copy = new URLSearchParams(p);
      if (id) copy.set("job", id);
      else copy.delete("job");
      return copy;
    });

  const reset = () => {
    setStage({ kind: "pick" });
    setFile(null);
    setError(null);
    setFileInputKey((k) => k + 1);
    setJob(null);
  };

  const parse = async (target: File, replace: DocumentSummary | null = null) => {
    setError(null);
    setStage({ kind: "parsing" });
    try {
      const result = await api.parseDocument(target, replace?.book_id);
      if (result.status === "duplicate") {
        setStage({ kind: "duplicate", result, file: target });
      } else {
        setStage({ kind: "review", parseId: result.parse_id!, review: result.review!, replace });
      }
    } catch (e) {
      setError(toApiError(e));
      setStage({ kind: "pick" });
    }
  };

  const replaceExisting = (dup: Extract<Stage, { kind: "duplicate" }>) => {
    const existing = dup.result.existing!;
    if (dup.result.match === "content" && dup.result.parse_id && dup.result.review) {
      // Already parsed (same text, different file): go straight to review.
      setStage({ kind: "review", parseId: dup.result.parse_id, review: dup.result.review, replace: existing });
    } else {
      void parse(dup.file, existing); // same file bytes: parse it now, as a replacement
    }
  };

  if (jobId) return <JobProgress jobId={jobId} onFinished={reset} />;

  return (
    <div className="stack">
      {error && <ErrorNotice error={error} />}

      {(stage.kind === "pick" || stage.kind === "parsing") && (
        <div className="card stack">
          <span className="muted small">
            Hỗ trợ PDF (cả bản scan), DOCX, HTML, Markdown. Bảng được nhận diện tự động; tài liệu trùng với tài liệu đã
            có sẽ được phát hiện.
          </span>
          <div className="row">
            <input
              key={fileInputKey}
              type="file"
              accept={ACCEPT}
              disabled={stage.kind === "parsing"}
              onChange={(e) => {
                setFile(e.target.files?.[0] ?? null);
                setError(null);
              }}
            />
            <button
              className="btn primary"
              disabled={!file || stage.kind === "parsing"}
              onClick={() => file && parse(file)}
            >
              {stage.kind === "parsing" ? "Đang phân tích..." : "Xử lý tài liệu"}
            </button>
          </div>
          {stage.kind === "parsing" && (
            <span className="muted small">
              Đang đọc cấu trúc tài liệu. Lần đầu có thể mất khoảng 1 phút; PDF dài có thể lâu hơn.
            </span>
          )}
        </div>
      )}

      {stage.kind === "duplicate" && (
        <DuplicateNotice stage={stage} onCancel={reset} onReplace={() => replaceExisting(stage)} />
      )}

      {stage.kind === "review" && (
        <Review
          key={stage.parseId}
          parseId={stage.parseId}
          review={stage.review}
          replace={stage.replace}
          onCancel={reset}
          onStarted={setJob}
        />
      )}
    </div>
  );
}

function DuplicateNotice({
  stage,
  onCancel,
  onReplace,
}: {
  stage: Extract<Stage, { kind: "duplicate" }>;
  onCancel: () => void;
  onReplace: () => void;
}) {
  const existing = stage.result.existing!;
  return (
    <div className="notice warn stack">
      <strong>
        {stage.result.match === "file"
          ? "File này đã được thêm vào kho trước đây."
          : "Nội dung tài liệu này trùng với một tài liệu đã có trong kho (có thể là cùng tài liệu được lưu lại hoặc xuất lại)."}
      </strong>
      <div>
        Tài liệu đã có: <strong>{existing.book_title}</strong>
        <div className="muted small">{documentMetaLine(existing)}</div>
      </div>
      <div className="row">
        <button className="btn" onClick={onReplace}>
          Thay thế bản cũ
        </button>
        <button className="btn" onClick={onCancel}>
          Huỷ
        </button>
      </div>
      <span className="muted small">Thay thế: bản cũ chỉ bị xoá sau khi bản mới đã được thêm thành công.</span>
    </div>
  );
}

function Review({
  parseId,
  review,
  replace,
  onCancel,
  onStarted,
}: {
  parseId: string;
  review: ParseReview;
  replace: DocumentSummary | null;
  onCancel: () => void;
  onStarted: (jobId: string) => void;
}) {
  const [title, setTitle] = useState(review.metadata.book_title);
  const [author, setAuthor] = useState(review.metadata.author ?? "");
  const [excluded, setExcluded] = useState<Set<string>>(new Set());
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);

  const toggleTable = (id: string) =>
    setExcluded((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  const submit = async () => {
    setSubmitting(true);
    setError(null);
    try {
      const { job_id } = await api.indexDocument({
        parse_id: parseId,
        title: title.trim(),
        author: author.trim(),
        excluded_table_ids: [...excluded],
        replace_book_id: replace?.book_id ?? null,
      });
      onStarted(job_id);
    } catch (e) {
      setError(toApiError(e));
      setSubmitting(false);
    }
  };

  return (
    <div className="stack">
      <h3 style={{ margin: 0 }}>Xem lại trước khi embed</h3>
      {replace && (
        <div className="notice info small">
          Bản mới sẽ <strong>thay thế</strong> tài liệu <strong>{replace.book_title}</strong> ({replace.num_chunks}{" "}
          chunk). Bản cũ chỉ bị xoá sau khi bản mới đã được thêm thành công.
        </div>
      )}
      {error && <ErrorNotice error={error} />}

      <div className="field">
        <label htmlFor="doc-title">Tên tài liệu</label>
        <input id="doc-title" className="input" value={title} onChange={(e) => setTitle(e.target.value)} />
      </div>
      <div className="field">
        <label htmlFor="doc-author">Tác giả (tuỳ chọn)</label>
        <input id="doc-author" className="input" value={author} onChange={(e) => setAuthor(e.target.value)} />
      </div>

      <div className="stats">
        <Stat label="Mục / tiêu đề" value={review.stats.headings} />
        <Stat label="Đoạn văn" value={review.stats.paragraphs} />
        <Stat label="Bảng" value={review.stats.tables} />
      </div>

      <div className="stack" style={{ gap: 6 }}>
        <strong>Cấu trúc phân cấp</strong>
        <span className="muted small">Bấm vào một mục để mở/đóng. Số bên cạnh là nội dung nằm trực tiếp dưới mục đó.</span>
        <HeadingTree root={review.tree} tables={review.tables} />
      </div>

      {review.tables.length > 0 && (
        <div className="stack" style={{ gap: 6 }}>
          <strong>Bảng phát hiện được ({review.tables.length})</strong>
          <span className="muted small">
            Mỗi bảng được embed thành chunk riêng (bảng dài được chia theo hàng, giữ lại hàng tiêu đề). Bỏ chọn những
            bảng không muốn đưa vào hệ thống, ví dụ bảng dùng để dàn trang.
          </span>
          {review.tables.map((t) => (
            <TableReview key={t.table_id} table={t} included={!excluded.has(t.table_id)} onToggle={() => toggleTable(t.table_id)} />
          ))}
        </div>
      )}

      <div className="row">
        <button className="btn primary" disabled={submitting || !title.trim()} onClick={submit}>
          {submitting ? "Đang bắt đầu..." : replace ? "Xác nhận, embed và thay thế bản cũ" : "Xác nhận và embed vào hệ thống"}
        </button>
        <button className="btn" disabled={submitting} onClick={onCancel}>
          Huỷ
        </button>
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: number }) {
  return (
    <div className="stat">
      <div className="muted small">{label}</div>
      <div className="value">{value}</div>
    </div>
  );
}

function nodeMeta(node: TreeNode): string {
  const parts: string[] = [];
  if (node.n_paragraphs) parts.push(`${node.n_paragraphs} đoạn`);
  if (node.table_ids?.length) parts.push(`${node.table_ids.length} bảng`);
  if (node.children?.length) parts.push(`${node.children.length} mục con`);
  return parts.join(" · ") || "trống";
}

/** The parsed heading tree, collapsible at every level; tables shown where they sit. */
function HeadingTree({ root, tables }: { root: TreeNode; tables: TableInfo[] }) {
  const byId = useMemo(() => new Map(tables.map((t) => [t.table_id, t])), [tables]);
  const tableLines = (node: TreeNode) =>
    (node.table_ids ?? []).map((id) => {
      const t = byId.get(id);
      return (
        <div key={id} className="tree-table">
          {t?.caption ?? `Bảng ${id.slice(1)}`}
          <span className="tree-meta">bảng{t && ` · ${t.n_rows} hàng × ${t.n_cols} cột`}</span>
        </div>
      );
    });

  const renderNode = (node: TreeNode, level: number) => {
    const children = node.children ?? [];
    const hasBody = children.length > 0 || (node.table_ids ?? []).length > 0;
    if (!hasBody) {
      return (
        <div key={`${level}-${node.title}`} className="tree-leaf">
          {node.title}
          <span className="tree-meta">{nodeMeta(node)}</span>
        </div>
      );
    }
    return (
      <details key={`${level}-${node.title}`} open={level === 0}>
        <summary>
          {node.title}
          <span className="tree-meta">{nodeMeta(node)}</span>
        </summary>
        <div className="tree-children">
          {tableLines(node)}
          {children.map((c) => renderNode(c, level + 1))}
        </div>
      </details>
    );
  };

  return (
    <div className="tree">
      {(root.n_paragraphs > 0 || (root.table_ids ?? []).length > 0) && (
        <>
          <div className="tree-leaf">
            <em>(Phần mở đầu, trước tiêu đề đầu tiên)</em>
            <span className="tree-meta">{nodeMeta({ ...root, children: [] })}</span>
          </div>
          {tableLines(root)}
        </>
      )}
      {(root.children ?? []).map((c) => renderNode(c, 0))}
    </div>
  );
}

function TableReview({ table, included, onToggle }: { table: TableInfo; included: boolean; onToggle: () => void }) {
  const [view, setView] = useState<"table" | "markdown">("table");
  const name = table.caption ?? `Bảng ${table.table_id.slice(1)}`;
  // The caption is the summary already; show the grid without repeating it.
  const grid = table.caption ? table.markdown.split("\n").slice(2).join("\n") : table.markdown;
  return (
    <Collapsible
      summary={
        <span>
          {name} <span className="muted small">· {table.n_rows} hàng × {table.n_cols} cột</span>
          {!included && <span className="muted small"> · sẽ bỏ qua</span>}
        </span>
      }
    >
      <div className="stack" style={{ gap: 8 }}>
        <span className="muted small">Vị trí: {table.path.length ? table.path.join(" › ") : "phần mở đầu"}</span>
        <label className="row small">
          <input type="checkbox" checked={included} onChange={onToggle} /> Đưa bảng này vào hệ thống
        </label>
        <div className="segmented">
          <button aria-pressed={view === "table"} onClick={() => setView("table")}>
            Bảng
          </button>
          <button aria-pressed={view === "markdown"} onClick={() => setView("markdown")}>
            Nội dung sẽ embed
          </button>
        </div>
        {view === "table" ? <Markdown>{grid}</Markdown> : <pre className="code-block">{table.markdown}</pre>}
      </div>
    </Collapsible>
  );
}

// ---------------------------------------------------------------- job progress

function JobProgress({ jobId, onFinished }: { jobId: string; onFinished: () => void }) {
  const { refreshDocuments } = useAppData();
  const [value, setValue] = useState(0);
  const [label, setLabel] = useState("Đang kết nối...");
  const [notes, setNotes] = useState<string[]>([]);
  const [done, setDone] = useState<JobDone | null>(null);
  const [error, setError] = useState<{ message: string; hint?: string | null } | null>(null);

  useEffect(() => {
    // Replays from the first event, so the bar is right after a reload too.
    setValue(0);
    setNotes([]);
    setDone(null);
    setError(null);
    return watchJob(
      jobId,
      (event) => {
        switch (event.type) {
          case "progress":
            setValue(event.value);
            setLabel(event.text);
            break;
          case "retry":
            setNotes((n) => [...n, event.text]);
            break;
          case "done":
            setValue(1);
            setLabel("Hoàn tất");
            setDone(event);
            void refreshDocuments();
            break;
          case "error":
            setLabel("Dừng lại do lỗi");
            setError(event);
            break;
        }
      },
      (lost) => setError(lost),
    );
  }, [jobId, refreshDocuments]);

  return (
    <div className="card stack">
      <ProgressBar value={value} label={label} />
      {notes.map((n, i) => (
        <span key={i} className="muted small">
          {n}
        </span>
      ))}
      {done && (
        <div className="notice success">
          Đã thêm tài liệu <strong>{done.book_title}</strong> ({done.chunks} chunk, trong đó {done.table_chunks} chunk
          bảng)
          {done.replaced_chunks != null && <>, đã xoá bản cũ ({done.replaced_chunks} chunk)</>}.
        </div>
      )}
      {error && <ErrorNotice error={error} />}
      {(done || error) && (
        <div className="row">
          <button className="btn primary" onClick={onFinished}>
            Thêm tài liệu khác
          </button>
        </div>
      )}
      {!done && !error && (
        <span className="muted small">Có thể chuyển trang hoặc tải lại trang, tiến độ vẫn được theo dõi tiếp.</span>
      )}
    </div>
  );
}

// ---------------------------------------------------------------- library

function Library() {
  const { documents, documentsError, refreshDocuments } = useAppData();
  const [query, setQuery] = useState("");
  const [notice, setNotice] = useState<{ kind: "success" | "error"; text: string; hint?: string | null } | null>(null);

  useEffect(() => {
    void refreshDocuments();
  }, [refreshDocuments]);

  const shown = useMemo(() => {
    const q = query.trim().toLowerCase();
    return documents
      .filter((d) => !q || d.book_title.toLowerCase().includes(q) || (d.source_file ?? "").toLowerCase().includes(q))
      .sort((a, b) => (b.uploaded_at ?? "").localeCompare(a.uploaded_at ?? ""));
  }, [documents, query]);

  const remove = async (doc: DocumentSummary) => {
    try {
      const { deleted_chunks } = await api.deleteDocument(doc.book_id);
      setNotice({ kind: "success", text: `Đã xoá "${doc.book_title}" (${deleted_chunks} chunk).` });
    } catch (e) {
      const err = toApiError(e);
      setNotice({ kind: "error", text: `Không xoá được "${doc.book_title}". ${err.message}`, hint: err.hint });
    }
    await refreshDocuments();
  };

  if (documentsError) return <ErrorNotice error={documentsError} />;

  return (
    <div className="stack">
      {notice &&
        (notice.kind === "success" ? (
          <div className="notice success">{notice.text}</div>
        ) : (
          <ErrorNotice error={{ message: notice.text, hint: notice.hint }} />
        ))}

      {documents.length === 0 ? (
        <div className="notice info">Kho chưa có tài liệu nào. Thêm tài liệu ở tab bên cạnh.</div>
      ) : (
        <>
          <div className="row spread">
            <span className="muted small">
              {documents.length} tài liệu · {documents.reduce((sum, d) => sum + d.num_chunks, 0)} chunk
            </span>
            {documents.length > 5 && (
              <input
                className="input"
                style={{ maxWidth: 320 }}
                placeholder="Lọc theo tên tài liệu hoặc tên file..."
                value={query}
                onChange={(e) => setQuery(e.target.value)}
              />
            )}
          </div>
          {shown.length === 0 && <span className="muted">Không có tài liệu nào khớp.</span>}
          {shown.map((doc) => (
            <div key={doc.book_id} className="card doc-card">
              <div style={{ minWidth: 0 }}>
                <div className="title">{doc.book_title}</div>
                <div className="muted small">{documentMetaLine(doc)}</div>
              </div>
              <ConfirmButton
                label="Xoá"
                confirmText={`Xoá ${doc.num_chunks} chunk?`}
                onConfirm={() => remove(doc)}
              />
            </div>
          ))}
        </>
      )}
    </div>
  );
}
