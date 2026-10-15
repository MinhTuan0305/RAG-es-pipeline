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
from app.services.citations import clean_citations, strip_citations

logger = logging.getLogger(__name__)

TITLE_MAX_CHARS = 60
NO_ANSWER = "_(Không có câu trả lời.)_"
STOPPED_EMPTY = "_(Đã dừng trước khi có câu trả lời.)_"


def conversation_title(message: str) -> str:
    title = " ".join(message.split())
    return title[:TITLE_MAX_CHARS] + ("…" if len(title) > TITLE_MAX_CHARS else "")


# A rough, deliberately generous token estimate (Vietnamese text runs at fewer
# characters per token than English). The budget is a soft limit on cost/latency/focus
# -- Gemini's context window is far larger -- so an exact count isn't worth an extra
# API call per question.
CHARS_PER_TOKEN = 3
# A message cut down to fit the budget keeps at least this much of its beginning.
MIN_SHORTENED_TOKENS = 200
SHORTENED_NOTE = "\n\n...(đã rút gọn)"


def estimate_tokens(text: str) -> int:
    return -(-len(text) // CHARS_PER_TOKEN)  # ceiling division


def shorten(text: str, max_tokens: int) -> str:
    """The beginning of `text`, within max_tokens (note included), cut at a word boundary."""
    limit = max_tokens * CHARS_PER_TOKEN - len(SHORTENED_NOTE)
    if len(text) <= max_tokens * CHARS_PER_TOKEN:
        return text
    cut = text[:limit]
    if " " in cut[limit // 2:]:
        cut = cut[: cut.rfind(" ")]
    return cut.rstrip() + SHORTENED_NOTE


def build_history(messages, token_budget=settings.history_token_budget):
    """The recent turns to send to the LLM with a new question, within token_budget
    (estimated): newest first until the budget is used up, oldest dropped. The latest
    question/answer pair is always kept -- shortened if it alone exceeds the budget --
    since follow-up questions mostly refer to it.

    Failed turns are dropped together with the question that caused them, and the
    result starts with a user turn, so user/assistant turns keep alternating."""
    clean = []
    for m in messages:
        if m.get("error"):
            if clean and clean[-1]["role"] == "user":
                clean.pop()
            continue
        # [n] citations meant passages of THAT question's searches: drop them, or the
        # model may reuse the numbers for this question's (different) passages.
        content = strip_citations(m["content"]) if m["role"] == "assistant" else m["content"]
        clean.append({"role": m["role"], "content": content})
    if token_budget <= 0:
        return []

    newest_first = list(reversed(clean))
    selected, used = [], 0
    for i, m in enumerate(newest_first):
        cost = estimate_tokens(m["content"])
        if used + cost <= token_budget:
            selected.append(m)
            used += cost
        elif len(selected) < 2:  # still within the latest pair: shorten rather than drop
            # Leave room for the other half of the pair (the question this answer
            # replied to), so one long answer can't crowd it out.
            reserve = 0
            if len(selected) == 0 and i + 1 < len(newest_first):
                reserve = min(estimate_tokens(newest_first[i + 1]["content"]), token_budget // 2)
            room = max(token_budget - used - reserve, MIN_SHORTENED_TOKENS)
            selected.append({**m, "content": shorten(m["content"], room)})
            used += estimate_tokens(selected[-1]["content"])
        else:
            break
    selected.reverse()
    while selected and selected[0]["role"] != "user":
        selected.pop(0)
    return selected


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
            "citation": c.get("citation"),
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
    streamed, final_answer, step, shown_calls, shown_sources, stopped = "", None, 0, 0, 0, False

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
                # The passages this round found, with their citation numbers, so [n]
                # in the answer streaming next can be shown as a source right away.
                new_sources = to_source_records(ctx.sources[shown_sources:])
                shown_sources = len(ctx.sources)
                if new_sources:
                    yield Sources(items=new_sources)
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
    answer, cited, invalid = clean_citations(answer, {s["citation"] for s in sources if s["citation"]})
    if invalid:
        logger.info("Removed citations to non-existent passages %s (conversation %s)", invalid, conversation_id)
    for s in sources:
        s["cited"] = s["citation"] in cited

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
    yield ChatDone(answer=answer, stopped=stopped, cited=cited)
