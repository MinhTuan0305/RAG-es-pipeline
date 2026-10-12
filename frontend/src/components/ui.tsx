import { useState, type ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

export function Markdown({ children, className = "" }: { children: string; className?: string }) {
  return (
    <div className={`markdown ${className}`}>
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{children}</ReactMarkdown>
    </div>
  );
}

/** An error from the backend (or the network): what went wrong + what to do. */
export function ErrorNotice({ error }: { error: { message: string; hint?: string | null } }) {
  return (
    <div className="notice error" role="alert">
      <strong>{error.message}</strong>
      {error.hint && <Markdown className="small">{error.hint}</Markdown>}
    </div>
  );
}

export function Collapsible({
  summary,
  children,
  defaultOpen = false,
}: {
  summary: ReactNode;
  children: ReactNode;
  defaultOpen?: boolean;
}) {
  return (
    <details className="collapsible" open={defaultOpen}>
      <summary>{summary}</summary>
      <div className="collapsible-body">{children}</div>
    </details>
  );
}

/** A destructive action that asks for confirmation inline, without a modal. */
export function ConfirmButton({
  label,
  confirmText,
  onConfirm,
  busyLabel = "Đang xoá...",
}: {
  label: string;
  confirmText: string;
  onConfirm: () => Promise<void>;
  busyLabel?: string;
}) {
  const [state, setState] = useState<"idle" | "confirm" | "busy">("idle");
  if (state === "idle") {
    return (
      <button className="btn danger small" onClick={() => setState("confirm")}>
        {label}
      </button>
    );
  }
  return (
    <div className="row">
      <span className="small">{confirmText}</span>
      <button
        className="btn danger solid small"
        disabled={state === "busy"}
        onClick={async () => {
          setState("busy");
          try {
            await onConfirm();
          } finally {
            setState("idle");
          }
        }}
      >
        {state === "busy" ? busyLabel : "Xác nhận"}
      </button>
      <button className="btn small" disabled={state === "busy"} onClick={() => setState("idle")}>
        Huỷ
      </button>
    </div>
  );
}

export function ProgressBar({ value, label }: { value: number; label: string }) {
  const percent = Math.round(Math.max(0, Math.min(1, value)) * 100);
  return (
    <div className="stack" style={{ gap: 6 }}>
      <div className="row spread small">
        <span>{label}</span>
        <span className="muted">{percent}%</span>
      </div>
      <div className="progress" role="progressbar" aria-valuenow={percent} aria-valuemin={0} aria-valuemax={100}>
        <div style={{ width: `${percent}%` }} />
      </div>
    </div>
  );
}
