import type {
  ConversationDetail,
  ConversationSummary,
  DocumentSummary,
  ErrorBody,
  Health,
  IndexRequest,
  JobInfo,
  ParseResult,
} from "./types";

/** Every API failure, with the backend's user-facing message and hint. */
export class ApiError extends Error {
  readonly hint: string | null;
  readonly status: number;

  constructor(status: number, body: ErrorBody) {
    super(body.message);
    this.status = status;
    this.hint = body.hint ?? null;
  }
}

/** Turn a failed response into an ApiError (FastAPI validation errors included). */
export async function errorFromResponse(response: Response): Promise<ApiError> {
  let body: ErrorBody = { message: `Lỗi ${response.status} từ máy chủ.` };
  try {
    const data = await response.json();
    if (typeof data?.message === "string") {
      body = { message: data.message, hint: data.hint ?? null };
    } else if (Array.isArray(data?.detail)) {
      body = { message: "Dữ liệu gửi lên không hợp lệ.", hint: data.detail.map((d: { msg: string }) => d.msg).join("; ") };
    }
  } catch {
    // not JSON (e.g. the backend is down and the dev proxy answered): keep the generic message
  }
  if (response.status === 502 || response.status === 504) {
    body = { message: "Không kết nối được tới backend.", hint: "Kiểm tra backend đã chạy chưa (cổng 8000)." };
  }
  return new ApiError(response.status, body);
}

/** Network failures (backend down, offline) as an ApiError too. */
export function toApiError(error: unknown): ApiError {
  if (error instanceof ApiError) return error;
  return new ApiError(0, {
    message: "Không kết nối được tới backend.",
    hint: error instanceof Error ? error.message : null,
  });
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`/api${path}`, init);
  } catch (error) {
    throw toApiError(error);
  }
  if (!response.ok) throw await errorFromResponse(response);
  return (response.status === 204 ? undefined : await response.json()) as T;
}

const json = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  health: () => request<Health>("/health"),

  conversations: () => request<ConversationSummary[]>("/conversations?limit=100"),
  conversation: (id: string) => request<ConversationDetail>(`/conversations/${encodeURIComponent(id)}`),
  deleteConversation: (id: string) =>
    request<void>(`/conversations/${encodeURIComponent(id)}`, { method: "DELETE" }),

  documents: () => request<DocumentSummary[]>("/documents"),
  deleteDocument: (bookId: string) =>
    request<{ deleted_chunks: number }>(`/documents/${encodeURIComponent(bookId)}`, { method: "DELETE" }),

  parseDocument: (file: File, replaceBookId?: string) => {
    const form = new FormData();
    form.append("file", file);
    if (replaceBookId) form.append("replace_book_id", replaceBookId);
    return request<ParseResult>("/documents/parse", { method: "POST", body: form });
  },
  indexDocument: (body: IndexRequest) => request<{ job_id: string }>("/documents/index", json(body)),
  job: (jobId: string) => request<JobInfo>(`/jobs/${encodeURIComponent(jobId)}`),
};
