"""
One chat question, end to end: history -> agent run -> SSE events -> saved messages.

Independent of any UI: builds the history, turns the agent's raw stream into
"what to show" events, and saves both turns. The API wraps run_chat() in an SSE response.
"""

import logging
import threading
from collections.abc import Iterator

from app.core.chat_store import add_message, create_conversation, load_messages
from app.core.config import settings
from app.core.errors import AppError, explain_error
from app.core.langgraph_agent import DocumentQAAgent, RunContext, extract_text
from app.core.rag_pipeline import chunk_location
from app.schemas import (
    AnswerReset,
    ChatDone,
    ErrorEvent,
    RunStarted,
    Sources,
    Token,
    ToolCall,
    ToolResult,
)

logger = logging.getLogger(__name__)

TITLE_MAX_CHARS = 60
NO_ANSWER = "_(Không có câu trả lời.)_"
STOPPED_EMPTY = "_(Đã dừng trước khi có câu trả lời.)_"


def conversation_title(message: str) -> str:
    title = " ".join(message.split())
    return title[:TITLE_MAX_CHARS] + ("…" if len(title) > TITLE_MAX_CHARS else "")


def build_history(messages, limit=settings.max_history_messages):
    """Recent turns to send to the LLM. Failed turns are dropped together with the
    question that caused them, so user/assistant turns keep alternating."""
    clean = []
    for m in messages:
        if m.get("error"):
            if clean and clean[-1]["role"] == "user":
                clean.pop()
            continue
        clean.append({"role": m["role"], "content": m["content"]})
    clean = clean[-limit:] if limit else []
    while clean and clean[0]["role"] != "user":
        clean.pop(0)
    return clean


def to_source_records(hits):
    """Keep only what the citation panel needs, so chat history stays small."""
    records = []
    for c in hits:
        src = c["hit"]["_source"]
        records.append({
            "type": src.get("type", "paragraph"),
            "book_title": src["book_title"],
            "location": chunk_location(src),
            "chunk_id": src["chunk_id"],
            "rerank_score": float(c["rerank_score"]),
            "text": src["text"],
        })
    return records


def run_chat(
    agent: DocumentQAAgent,
    message: str,
    conversation_id: str | None,
    book_id: str | None,
    book_title: str | None,
    cancel: threading.Event,
) -> Iterator:
    """Yield the SSE events of one question (see app/schemas.py), saving the user
    turn first and the assistant turn (answer, or error, or partial answer if
    `cancel` was set) at the end. Never raises: failures become an ErrorEvent."""
    title = None
    if conversation_id is None:
        title = conversation_title(message)
        conversation_id = create_conversation(title)
        history = []
    else:
        history = build_history(load_messages(conversation_id))
    add_message(conversation_id, {"role": "user", "content": message})
    yield RunStarted(conversation_id=conversation_id, title=title)

    ctx = RunContext(book_id=book_id, book_title=book_title, conversation_id=conversation_id)
    streamed, final_answer, step, shown_calls, stopped = "", None, 0, 0, False

    stream = agent.stream(message, ctx, history=history)
    try:
        for event in stream:
            if cancel.is_set():
                stopped = True
                break

            if event["type"] == "token":
                streamed += event["text"]
                yield Token(text=event["text"])
                continue

            if event["node"] == "tools":
                # One "tools" step can hold several searches (the LLM may request them
                # in parallel) -- report each one exactly once.
                for call in ctx.calls[shown_calls:]:
                    yield ToolResult(query=call["query"], num_hits=call["num_hits"])
                shown_calls = len(ctx.calls)
                continue

            for msg in event["messages"]:
                tool_calls = getattr(msg, "tool_calls", None) or []
                if tool_calls:
                    if streamed:
                        # Text streamed in this turn was a preamble before a search.
                        streamed = ""
                        yield AnswerReset()
                    for tc in tool_calls:
                        step += 1
                        yield ToolCall(step=step, query=tc["args"].get("query", ""))
                elif msg.content:
                    final_answer = extract_text(msg.content)
    except Exception as e:  # noqa: BLE001 -- every failure becomes an error event
        err = e if isinstance(e, AppError) else explain_error(e)
        logger.error("Chat run failed (conversation %s)", conversation_id, exc_info=e)
        add_message(conversation_id, {"role": "assistant", "content": err.to_markdown(), "error": True})
        yield ErrorEvent(message=err.message, hint=err.hint)
        return
    finally:
        stream.close()  # stopped early: let LangGraph wind the run down now, not at GC

    if stopped:
        answer = streamed or STOPPED_EMPTY
    else:
        answer = final_answer or streamed or NO_ANSWER
    sources = to_source_records(ctx.sources)
    if sources:
        yield Sources(items=sources)

    assistant_msg = {
        "role": "assistant",
        "content": answer,
        "tool_calls": list(ctx.calls),
        "sources": sources,
        "scope": ctx.book_title,
    }
    if stopped:
        assistant_msg["stopped"] = True
    add_message(conversation_id, assistant_msg)
    yield ChatDone(answer=answer, stopped=stopped)
