import type { DocumentSummary } from "./api/types";

/** Backend timestamp (UTC) -> local "dd/mm/yyyy HH:MM"; null if missing/invalid. */
export function formatDateTime(value: string | null | undefined): string | null {
  if (!value) return null;
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return null;
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${pad(date.getDate())}/${pad(date.getMonth() + 1)}/${date.getFullYear()} ${pad(date.getHours())}:${pad(date.getMinutes())}`;
}

export function documentMetaLine(doc: DocumentSummary): string {
  const parts = [`${doc.num_chunks} chunk`];
  if (doc.num_tables) parts.push(`${doc.num_tables} bảng`);
  if (doc.source_file) parts.push(doc.source_file);
  const uploaded = formatDateTime(doc.uploaded_at);
  parts.push(uploaded ? `thêm lúc ${uploaded}` : "chưa có thông tin file và ngày thêm");
  return parts.join(" · ");
}
