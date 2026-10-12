import { useCallback, useEffect, useLayoutEffect, useRef, useState, type KeyboardEvent, type RefObject } from "react";
import { useNavigate, useParams } from "react-router-dom";

import { api, toApiError, type ApiError } from "../api/client";
import { streamChat } from "../api/chatStream";
import type { ChatEvent, Message, SourceRecord } from "../api/types";
import { useAppData } from "../appData";
import { Collapsible, ConfirmButton, ErrorNotice, Markdown } from "../components/ui";

interface Step {
  query: string;
  numHits: number | null; // null while that search is still running
}

/** A message as displayed: a saved Message, or the answer currently streaming in. */
interface ViewMessage {
  key: string;
  role: "user" | "assistant";
  content: string;
  steps: Step[];
  sources: SourceRecord[];
  scope?: string | null;
  errorMarkdown?: string | null;
  stopped?: boolean;
  streaming?: boolean;
}

const SCOPE_STORAGE_KEY = "chat-scope";

function fromMessage(m: Message, index: number): ViewMessage {
  return {
    key: `saved-${index}`,
    role: m.role,
    content: m.error ? "" : m.content,
    steps: (m.tool_calls ?? []).map((c) => ({ query: c.query, numHits: c.num_hits })),
    sources: m.sources ?? [],
    scope: m.scope,
    errorMarkdown: m.error ? m.content : null,
    stopped: m.stopped,
  };
}

function readSavedScope(): string {
  try {
    return localStorage.getItem(SCOPE_STORAGE_KEY) ?? "";
  } catch {
    return "";
  }
}

export function ChatPage() {
  const { conversationId } = useParams();
  const navigate = useNavigate();
  const { conversations, refreshConversations, documents } = useAppData();

  const [messages, setMessages] = useState<ViewMessage[]>([]);
  const [loadError, setLoadError] = useState<ApiError | null>(null);
  const [loading, setLoading] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [scope, setScope] = useState<string>(readSavedScope);

  const abortRef = useRef<AbortController | null>(null);
  // Identifies the run allowed to update the message list. Opening another
  // conversation clears it, so a stream being torn down can't write into the
  // newly loaded messages.
  const runRef = useRef<symbol | null>(null);
  // The conversation the running stream belongs to. When a question starts a new
  // conversation, the URL switches to it mid-stream -- that must neither reload the
  // messages from the server nor abort the stream.
  const streamingConversationRef = useRef<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const stick = useStickToBottom(scrollRef, messages);

  const scopeDoc = documents.find((d) => d.book_id === scope) ?? null;
  const title = conversations.find((c) => c.id === conversationId)?.title ?? null;

  useEffect(() => {
    if (conversationId && conversationId === streamingConversationRef.current) return;
    // Opening another conversation stops the answer streaming in this one (the
    // backend saves what was received so far, marked as stopped).
    runRef.current = null;
    abortRef.current?.abort();
    setLoadError(null);
    if (!conversationId) {
      setMessages([]);
      return;
    }
    let alive = true;
    setLoading(true);
    api
      .conversation(conversationId)
      .then((detail) => alive && setMessages(detail.messages.map(fromMessage)))
      .catch((error: ApiError) => alive && (setMessages([]), setLoadError(error)))
      .finally(() => alive && setLoading(false));
    return () => {
      alive = false;
    };
  }, [conversationId]);

  const changeScope = (value: string) => {
    setScope(value);
    try {
      localStorage.setItem(SCOPE_STORAGE_KEY, value);
    } catch {
      // storage unavailable: the choice just won't be remembered
    }
  };

  const send = useCallback(
    async (text: string) => {
      const question = text.trim();
      if (!question || streaming) return;

      const runId = Symbol("run");
      runRef.current = runId;
      const updateAnswer = (fn: (m: ViewMessage) => ViewMessage) => {
        if (runRef.current !== runId) return;
        setMessages((ms) => [...ms.slice(0, -1), fn(ms[ms.length - 1])]);
      };

      setMessages((ms) => [
        ...ms,
        { key: `u-${Date.now()}`, role: "user", content: question, steps: [], sources: [] },
        { key: `a-${Date.now()}`, role: "assistant", content: "", steps: [], sources: [], streaming: true,
          scope: scopeDoc?.book_title ?? null },
      ]);
      stick.scrollToBottom();

      const controller = new AbortController();
      abortRef.current = controller;
      streamingConversationRef.current = conversationId ?? null;
      setStreaming(true);

      const onEvent = (event: ChatEvent) => {
        switch (event.type) {
          case "run_started":
            if (!conversationId) {
              streamingConversationRef.current = event.conversation_id;
              navigate(`/chat/${event.conversation_id}`, { replace: true });
              void refreshConversations();
            }
            break;
          case "tool_call":
            updateAnswer((m) => ({ ...m, steps: [...m.steps, { query: event.query, numHits: null }] }));
            break;
          case "tool_result":
            updateAnswer((m) => {
              const i = m.steps.findIndex((s) => s.query === event.query && s.numHits === null);
              if (i < 0) return m;
              const steps = [...m.steps];
              steps[i] = { ...steps[i], numHits: event.num_hits };
              return { ...m, steps };
            });
            break;
          case "answer_reset":
            updateAnswer((m) => ({ ...m, content: "" }));
            break;
          case "token":
            updateAnswer((m) => ({ ...m, content: m.content + event.text }));
            break;
          case "sources":
            updateAnswer((m) => ({ ...m, sources: event.items }));
            break;
          case "done":
            updateAnswer((m) => ({ ...m, content: event.answer, stopped: event.stopped, streaming: false }));
            break;
          case "error":
            updateAnswer((m) => ({
              ...m, streaming: false, errorMarkdown: `**${event.message}**${event.hint ? `\n\n${event.hint}` : ""}`,
            }));
            break;
        }
      };

      try {
        await streamChat(
          { message: question, conversation_id: conversationId ?? null, book_id: scope || null },
          onEvent,
          controller.signal,
        );
        if (controller.signal.aborted) updateAnswer((m) => ({ ...m, streaming: false, stopped: true }));
      } catch (error) {
        const err = toApiError(error);
        updateAnswer((m) => ({
          ...m, streaming: false, errorMarkdown: `**${err.message}**${err.hint ? `\n\n${err.hint}` : ""}`,
        }));
      } finally {
        if (runRef.current === runId) runRef.current = null;
        abortRef.current = null;
        streamingConversationRef.current = null;
        setStreaming(false);
        void refreshConversations();
      }
    },
    [conversationId, navigate, refreshConversations, scope, scopeDoc, stick, streaming],
  );

  const deleteConversation = async () => {
    if (!conversationId) return;
    try {
      await api.deleteConversation(conversationId);
    } catch (error) {
      setLoadError(toApiError(error));
      return;
    }
    await refreshConversations();
    navigate("/chat");
  };

  return (
    <div className="chat">
      <div className="chat-header">
        <div className="title">{conversationId ? title ?? "Cuộc trò chuyện" : "Cuộc trò chuyện mới"}</div>
        <div className="row">
          <label className="scope small">
            <span className="muted">Phạm vi</span>
            <select
              className="select"
              value={scopeDoc ? scope : ""}
              onChange={(e) => changeScope(e.target.value)}
              disabled={streaming}
            >
              <option value="">Tất cả tài liệu ({documents.length})</option>
              {documents.map((d) => (
                <option key={d.book_id} value={d.book_id}>
                  {d.book_title}
                </option>
              ))}
            </select>
          </label>
          {conversationId && !streaming && (
            <ConfirmButton label="Xoá" confirmText="Xoá cuộc trò chuyện này?" onConfirm={deleteConversation} />
          )}
        </div>
      </div>

      <div className="messages" ref={scrollRef}>
        <div className="messages-inner">
          {loadError && <ErrorNotice error={loadError} />}
          {loading && <span className="muted">Đang tải cuộc trò chuyện...</span>}
          {!loading && !loadError && messages.length === 0 && (
            <div className="welcome">
              <h2>Hỏi đáp tài liệu</h2>
              <p className="muted">
                Đặt câu hỏi về nội dung các tài liệu trong kho. Chọn phạm vi ở góc trên nếu muốn hỏi riêng một tài
                liệu.
              </p>
            </div>
          )}
          {messages.map((m) =>
            m.role === "user" ? (
              <div key={m.key} className="msg-user">
                {m.content}
              </div>
            ) : (
              <AssistantMessage key={m.key} message={m} />
            ),
          )}
        </div>
      </div>

      <Composer streaming={streaming} onSend={send} onStop={() => abortRef.current?.abort()} />
    </div>
  );
}

function AssistantMessage({ message: m }: { message: ViewMessage }) {
  const thinking = m.streaming && !m.content;
  return (
    <div className="msg-assistant">
      {m.steps.length > 0 && (
        <Collapsible defaultOpen={m.streaming} summary={`Quá trình suy luận (${m.steps.length} lượt tìm kiếm)`}>
          <ol className="steps">
            {m.steps.map((s, i) => (
              <li key={i}>
                <span>Tìm: <em>{s.query}</em></span>{" "}
                <span className="muted">
                  {s.numHits === null ? "— đang tìm..." : `— ${s.numHits} đoạn mới`}
                </span>
              </li>
            ))}
          </ol>
        </Collapsible>
      )}

      {thinking && <span className="muted">{m.steps.length ? "Đang tổng hợp câu trả lời..." : "Đang suy nghĩ..."}</span>}
      {m.content && <Markdown className={`answer${m.streaming ? " streaming" : ""}`}>{m.content}</Markdown>}
      {m.errorMarkdown && (
        <div className="notice error" role="alert">
          <Markdown>{m.errorMarkdown}</Markdown>
        </div>
      )}
      {m.stopped && <span className="muted small">Đã dừng trả lời.</span>}

      {!m.streaming && m.scope && <span className="muted small">Phạm vi tìm kiếm: {m.scope}</span>}
      {m.sources.length > 0 && (
        <Collapsible summary={`Nguồn trích dẫn (${m.sources.length} đoạn)`}>
          {m.sources.map((s) => (
            <SourceItem key={s.chunk_id} source={s} />
          ))}
        </Collapsible>
      )}
    </div>
  );
}

const PREVIEW_CHARS = 300;

function SourceItem({ source: s }: { source: SourceRecord }) {
  const [expanded, setExpanded] = useState(false);
  const isTable = s.type === "table";
  const isLong = s.text.length > PREVIEW_CHARS;
  return (
    <div className="source">
      <div>
        <strong>{s.book_title}</strong> — {s.location}
        {isTable && <span className="muted"> · bảng</span>}
      </div>
      <div className="muted small">
        <code>{s.chunk_id}</code> · rerank {s.rerank_score.toFixed(4)}
      </div>
      {isTable ? (
        <Markdown className="small">{s.text}</Markdown>
      ) : (
        <div className="source-text">
          {expanded || !isLong ? s.text : `${s.text.slice(0, PREVIEW_CHARS)}...`}
          {isLong && (
            <>
              {" "}
              <button className="btn link small" onClick={() => setExpanded(!expanded)}>
                {expanded ? "Thu gọn" : "Xem đầy đủ"}
              </button>
            </>
          )}
        </div>
      )}
    </div>
  );
}

function Composer({
  streaming,
  onSend,
  onStop,
}: {
  streaming: boolean;
  onSend: (text: string) => void;
  onStop: () => void;
}) {
  const [text, setText] = useState("");
  const ref = useRef<HTMLTextAreaElement>(null);

  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 180)}px`;
  }, [text]);

  const submit = () => {
    if (!text.trim() || streaming) return;
    onSend(text);
    setText("");
  };

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    // isComposing: Vietnamese input methods use Enter while composing a character.
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      submit();
    }
  };

  return (
    <div className="composer">
      <div className="composer-inner">
        <textarea
          ref={ref}
          className="textarea"
          rows={1}
          value={text}
          placeholder="Hỏi gì đó về nội dung tài liệu..."
          onChange={(e) => setText(e.target.value)}
          onKeyDown={onKeyDown}
        />
        {streaming ? (
          <button className="btn" onClick={onStop}>
            Dừng
          </button>
        ) : (
          <button className="btn primary" onClick={submit} disabled={!text.trim()}>
            Gửi
          </button>
        )}
      </div>
      <div className="composer-hint">Enter để gửi, Shift + Enter để xuống dòng.</div>
    </div>
  );
}

/** Keep the list scrolled to the newest content -- but only while the reader is
 * already at the bottom. Scrolling up to read stops the auto-scroll (no fighting
 * with the reader); coming back to the bottom resumes it. */
function useStickToBottom(ref: RefObject<HTMLDivElement | null>, content: unknown) {
  const atBottom = useRef(true);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const onScroll = () => {
      atBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
    };
    el.addEventListener("scroll", onScroll, { passive: true });
    return () => el.removeEventListener("scroll", onScroll);
  }, [ref]);

  useLayoutEffect(() => {
    const el = ref.current;
    if (el && atBottom.current) el.scrollTop = el.scrollHeight;
  }, [ref, content]);

  return useRef({
    scrollToBottom: () => {
      atBottom.current = true;
      const el = ref.current;
      if (el) el.scrollTop = el.scrollHeight;
    },
  }).current;
}
