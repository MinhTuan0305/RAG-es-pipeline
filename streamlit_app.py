import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

from chat_store import (
    add_message,
    create_conversation,
    delete_conversation,
    get_conversation,
    init_db,
    list_conversations,
    load_messages,
)
from chunking import build_chunk_records, get_tokenizer
from config import settings
from document_ingest import (
    SUPPORTED_EXTENSIONS,
    drop_tables,
    file_sha256,
    get_converter,
    ingest_document,
    tree_html,
    tree_stats,
    tree_tables,
    tree_to_book,
)
from langgraph_agent import DocumentQAAgent, RunContext, extract_text
from rag_pipeline import (
    chunk_location,
    delete_document,
    embed_chunks_batch,
    ensure_index_exists,
    find_document_by_hash,
    index_chunks,
    list_documents,
    load_embed_model,
    load_es_client,
    load_reranker,
)

MAX_HISTORY_MESSAGES = settings.max_history_messages

PAGE_CHAT = "Trò chuyện"
PAGE_UPLOAD = "Thêm tài liệu"

st.set_page_config(page_title="RAG")


@st.cache_resource(show_spinner="Đang kết nối Elasticsearch...")
def get_es_client():
    return load_es_client()


@st.cache_resource(show_spinner=f"Đang load model embedding ({settings.embed_model})...")
def get_embed_model():
    return load_embed_model()


@st.cache_resource(show_spinner=f"Đang load model reranker ({settings.reranker_model})...")
def get_reranker():
    return load_reranker()


@st.cache_resource(show_spinner="Đang khởi tạo Docling (lần đầu có thể mất khoảng 1 phút)...")
def get_doc_converter():
    return get_converter()


@st.cache_resource(show_spinner=f"Đang khởi tạo agent (LangGraph + {settings.llm_model})...")
def get_agent():
    return DocumentQAAgent(get_es_client(), get_embed_model(), get_reranker())


@st.cache_data(ttl=300, show_spinner=False)
def get_document_list(_es):
    # Leading underscore: tells st.cache_data not to try hashing the ES client.
    return list_documents(_es)


try:
    agent = get_agent()
except Exception as e:
    st.error(f"Không khởi tạo được hệ thống: {e}")
    st.stop()

# ---------------------------------------------------------------- chat history state
#
# st.session_state["messages"] mirrors the current conversation for rendering; every
# message is also written to SQLite (chat_store.py) so it survives a page reload or
# an app restart. The current conversation's id is kept in the URL (?c=...) so a
# reload (a brand-new Streamlit session) knows which conversation to reopen.
# Each message: {"role": "user"|"assistant", "content": str, plus for assistant
# turns optionally "tool_calls", "sources", "scope", "error"}.

def start_new_conversation():
    st.session_state["conversation_id"] = None  # created lazily on the first message
    st.session_state["messages"] = []
    st.query_params.clear()


def open_conversation(conversation_id):
    st.session_state["conversation_id"] = conversation_id
    st.session_state["messages"] = load_messages(conversation_id)
    st.query_params["c"] = conversation_id


def delete_current_conversation():
    if st.session_state["conversation_id"]:
        delete_conversation(st.session_state["conversation_id"])
    start_new_conversation()


def append_message(msg):
    if st.session_state["conversation_id"] is None:
        title = " ".join(msg["content"].split())
        title = title[:60] + ("…" if len(title) > 60 else "")
        conversation_id = create_conversation(title)
        st.session_state["conversation_id"] = conversation_id
        st.query_params["c"] = conversation_id
    st.session_state["messages"].append(msg)
    add_message(st.session_state["conversation_id"], msg)


init_db()
if "conversation_id" not in st.session_state:
    requested = st.query_params.get("c")
    if requested and get_conversation(requested):
        open_conversation(requested)
    else:
        start_new_conversation()


# ---------------------------------------------------------------- rendering helpers

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
            "rerank_score": c["rerank_score"],
            "text": src["text"],
        })
    return records


def render_steps(tool_calls):
    with st.expander(f"Quá trình suy luận ({len(tool_calls)} lượt tìm kiếm)"):
        for i, call in enumerate(tool_calls, start=1):
            st.markdown(f"**Bước {i}** — _{call['query']!r}_: {call['num_hits']} đoạn")


def render_sources(sources):
    with st.expander(f"Nguồn trích dẫn ({len(sources)} đoạn)"):
        for s in sources:
            is_table = s.get("type") == "table"
            st.markdown(
                f"**{s['book_title']}** — {s['location']}{' · bảng' if is_table else ''}  \n"
                f"`{s['chunk_id']}` · rerank score={s['rerank_score']:.4f}"
            )
            if is_table:
                # The first line of a table chunk is its caption (or heading) line.
                first_line = s["text"].split("\n", 1)[0]
                st.caption(first_line if not first_line.startswith("|") else "Bảng")
                with st.popover("Xem bảng"):
                    st.markdown(s["text"])
                st.divider()
                continue
            is_long = len(s["text"]) > 300
            st.caption(s["text"][:300] + ("..." if is_long else ""))
            if is_long:
                with st.popover("Xem đầy đủ"):
                    st.write(s["text"])
            st.divider()


def render_assistant_extras(msg):
    if msg.get("scope"):
        st.caption(f"Phạm vi tìm kiếm: {msg['scope']}")
    if msg.get("sources"):
        render_sources(msg["sources"])


def build_history(messages):
    """Recent turns to send to the LLM. Failed turns are dropped together with the
    question that caused them, so user/assistant turns keep alternating."""
    clean = []
    for m in messages:
        if m.get("error"):
            if clean and clean[-1]["role"] == "user":
                clean.pop()
            continue
        clean.append({"role": m["role"], "content": m["content"]})
    clean = clean[-MAX_HISTORY_MESSAGES:]
    while clean and clean[0]["role"] != "user":
        clean.pop(0)
    return clean


def table_keep_key(book_id, table_id):
    return f"keep_table_{book_id}_{table_id}"


def render_table_review(tables, book_id):
    """One expander per detected table: where it sits, a tick box to leave it out, the
    table as a reader sees it, and the exact Markdown that will be embedded."""
    for path, table in tables:
        name = table["caption"] or f"Bảng {table['table_id'][1:]}"
        dims = f"{len(table['rows'])} hàng × {len(table['header'])} cột"
        with st.expander(f"{name} · {dims}"):
            st.caption("Vị trí: " + (" › ".join(path) or "phần mở đầu"))
            st.checkbox("Đưa bảng này vào hệ thống", value=True, key=table_keep_key(book_id, table["table_id"]))
            view_tab, md_tab = st.tabs(["Bảng", "Nội dung sẽ embed (Markdown)"])
            with view_tab:
                st.markdown(table_markdown_for_display(table))
            with md_tab:
                st.code(table["text"], language="markdown")


def table_markdown_for_display(table):
    """Header + rows only -- the caption is already the expander's title."""
    if not table["caption"]:
        return table["text"]
    return table["text"].split("\n", 2)[2]


# ---------------------------------------------------------------- sidebar

with st.sidebar:
    st.title("Hỏi đáp tài liệu")
    page = st.radio("Chức năng", [PAGE_CHAT, PAGE_UPLOAD], label_visibility="collapsed")
    st.divider()

    selected_doc = None
    if page == PAGE_CHAT:
        try:
            documents = get_document_list(get_es_client())
        except Exception as e:
            st.warning(f"Không lấy được danh sách tài liệu, sẽ tìm trên toàn bộ kho: {e}")
            documents = []

        selected_doc = st.selectbox(
            "Phạm vi tìm kiếm",
            options=[None] + documents,
            format_func=lambda d: (
                f"Tất cả tài liệu ({len(documents)})" if d is None
                else f"{d['book_title']} ({d['num_chunks']} chunk)"
            ),
        )

        st.button("Cuộc trò chuyện mới", on_click=start_new_conversation, width="stretch")

        st.markdown("**Lịch sử trò chuyện**")
        # Filled at the end of the chat page, so a conversation created by the
        # message just sent already shows up in the list on this same run.
        conversation_list = st.container()

        if st.session_state["conversation_id"]:
            with st.popover("Xoá cuộc trò chuyện này", width="stretch"):
                st.write("Xoá vĩnh viễn cuộc trò chuyện đang mở? Không hoàn tác được.")
                st.button("Xác nhận xoá", type="primary", on_click=delete_current_conversation)

    st.divider()
    st.caption(
        f"Hybrid search ({settings.embed_model.split('/')[-1]} dense + sparse) · "
        f"rerank ({settings.reranker_model.split('/')[-1]}) · LangGraph agent ({settings.llm_model})"
    )


# ---------------------------------------------------------------- chat page

if page == PAGE_CHAT:
    # At the page's top level, chat_input is pinned to the bottom of the screen no
    # matter where it's called -- calling it first lets the welcome message below
    # know whether a question was just submitted.
    prompt = st.chat_input("Hỏi gì đó về nội dung tài liệu...")

    if not st.session_state["messages"] and not prompt:
        with st.chat_message("assistant"):
            st.markdown(
                "Xin chào! Mình trả lời câu hỏi dựa trên các tài liệu trong kho. "
                "Chọn phạm vi tìm kiếm ở thanh bên nếu muốn hỏi riêng một tài liệu"
            )

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant" and msg.get("tool_calls"):
                render_steps(msg["tool_calls"])
            if msg.get("error"):
                st.error(msg["content"])
            else:
                st.markdown(msg["content"])
            if msg["role"] == "assistant":
                render_assistant_extras(msg)

    if prompt:
        history = build_history(st.session_state["messages"])
        append_message({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Per-question state: this run's scope plus everything its searches find.
        # Owned by this session alone, so other users/tabs asking at the same time
        # can't overwrite it (the agent object itself is shared across sessions).
        run = RunContext(
            book_id=selected_doc["book_id"] if selected_doc else None,
            book_title=selected_doc["book_title"] if selected_doc else None,
            # Already set: append_message() above creates the conversation if needed.
            conversation_id=st.session_state["conversation_id"],
        )

        with st.chat_message("assistant"):
            # Created in this order so the step log sits above the streaming answer.
            status = st.status("Đang suy luận...", expanded=True)
            answer_placeholder = st.empty()
            streamed_text = ""
            final_answer = None
            step_num = 0
            shown_calls = 0

            try:
                for event in agent.stream(prompt, run, history=history):
                    if event["type"] == "token":
                        streamed_text += event["text"]
                        answer_placeholder.markdown(streamed_text + "▌")
                        continue

                    if event["node"] == "tools":
                        # One "tools" step can contain several searches (the LLM may
                        # request them in parallel) -- report each one exactly once.
                        for call in run.calls[shown_calls:]:
                            status.write(f"　　– _{call['query']!r}_: tìm thấy {call['num_hits']} đoạn liên quan")
                        shown_calls = len(run.calls)
                        continue

                    for msg in event["messages"]:
                        tool_calls = getattr(msg, "tool_calls", None) or []
                        if tool_calls:
                            # Any text streamed during this turn was a preamble before
                            # a tool call, not the answer -- drop it.
                            streamed_text = ""
                            answer_placeholder.empty()
                            for tc in tool_calls:
                                step_num += 1
                                sub_query = tc["args"].get("query", "")
                                status.write(f"**Bước {step_num}** — LLM quyết định tìm kiếm: _{sub_query!r}_")
                        elif msg.content:
                            # An AIMessage with no tool_calls is the agent's final answer.
                            final_answer = extract_text(msg.content)
            except Exception as e:
                answer_placeholder.empty()
                status.update(label="Có lỗi xảy ra", state="error", expanded=False)
                error_text = f"Có lỗi xảy ra: {e}"
                st.error(error_text)
                append_message({"role": "assistant", "content": error_text, "error": True})
            else:
                status.update(label=f"Hoàn tất sau {step_num} lượt tìm kiếm", state="complete", expanded=False)
                final_answer = final_answer or streamed_text or "_(Không có câu trả lời.)_"
                # Re-render from the complete message: removes the cursor, and still
                # shows the answer if the model didn't stream token-by-token.
                answer_placeholder.markdown(final_answer)

                assistant_msg = {
                    "role": "assistant",
                    "content": final_answer,
                    "tool_calls": list(run.calls),
                    "sources": to_source_records(run.sources),
                    "scope": run.book_title,
                }
                render_assistant_extras(assistant_msg)
                append_message(assistant_msg)

    with conversation_list:
        conversations = list_conversations(limit=30)
        if not conversations:
            st.caption("Chưa có cuộc trò chuyện nào.")
        for conv in conversations:
            is_current = conv["id"] == st.session_state["conversation_id"]
            st.button(
                conv["title"],
                key=f"conv_{conv['id']}",
                on_click=open_conversation,
                args=(conv["id"],),
                type="primary" if is_current else "tertiary",
                width="stretch",
            )


# ---------------------------------------------------------------- upload page
#
# Flow on the "add" tab, as a small state machine in st.session_state:
#   pick file -> [Xử lý] -> same file bytes already indexed?   -> duplicate warning
#                        -> Docling -> same text already indexed? -> duplicate warning
#                        -> review ("pending_*") -> [Embed] -> indexed
# A duplicate warning offers Cancel or Replace. Replace continues the upload and
# deletes the old document only AFTER the new one was indexed successfully, so a
# failed embed (e.g. out of memory) never loses the old copy.

def md_escape(text) -> str:
    """User-provided names (titles, file names) shown inside Markdown must not be
    able to turn into formatting, e.g. a "*" or "_" in a file name."""
    return re.sub(r"([\\`*_\[\]<>#|~])", r"\\\1", str(text))


def format_upload_time(value):
    """ES date string (UTC) -> local "dd/mm/yyyy HH:MM", or None."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone().strftime("%d/%m/%Y %H:%M")
    except ValueError:
        return None


def document_meta_line(doc):
    parts = [f"{doc['num_chunks']} chunk"]
    if doc.get("num_tables"):
        parts.append(f"{doc['num_tables']} bảng")
    if doc.get("source_file"):
        parts.append(md_escape(doc['source_file']))
    uploaded = format_upload_time(doc.get("uploaded_at"))
    parts.append(f"thêm lúc {uploaded}" if uploaded else "chưa có thông tin file và ngày thêm")
    return " · ".join(parts)


def clear_duplicate_warning():
    st.session_state.pop("upload_dup", None)
    st.session_state.pop("upload_replace_request", None)


def clear_pending_review():
    for key in ("pending_tree", "pending_metadata", "pending_doc"):
        st.session_state.pop(key, None)


def start_review(result, pending_doc):
    metadata = dict(result["metadata"])
    replace = pending_doc.get("replace")
    if replace:
        metadata["book_title"] = replace["book_title"]  # a replacement keeps the document's name by default
    st.session_state["pending_tree"] = result["tree"]
    st.session_state["pending_metadata"] = metadata
    st.session_state["pending_doc"] = pending_doc
    st.session_state.pop("upload_dup", None)


def replace_existing(dup):
    if "parsed" in dup:
        # Matched on content: the new file is already parsed, go straight to review.
        start_review(dup["parsed"], {**dup["pending_doc"], "replace": dup["existing"]})
    else:
        # Matched on file bytes: still needs parsing, which shows a spinner, so it
        # happens in the script run this callback triggers rather than in here.
        st.session_state["upload_replace_request"] = dup["existing"]
        st.session_state.pop("upload_dup", None)


def delete_document_callback(book_id, title):
    try:
        n_deleted = delete_document(get_es_client(), book_id)
    except Exception as e:
        st.session_state["library_notice"] = ("error", f"Không xoá được **{md_escape(title)}**: {e}")
    else:
        st.session_state["library_notice"] = ("success", f"Đã xoá **{md_escape(title)}** ({n_deleted} chunk).")
    get_document_list.clear()


def parse_and_check(uploaded_file, file_hash, replace=None):
    """Docling-parse the upload, then check its text against the library. Ends at
    the review step, or at a duplicate warning (via rerun)."""
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    try:
        get_doc_converter()  # first use: slow Docling import, shown with its own spinner
        with st.spinner("Đang phân tích cấu trúc tài liệu (Docling)..."):
            result = ingest_document(tmp_path, uploaded_file.name)
    except Exception as e:
        st.error(f"Không phân tích được tài liệu: {e}")
        return
    finally:
        # Docling only needs the file during parsing -- always clean it up.
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    if not result["tree"]["children"] and not result["tree"]["blocks"]:
        st.warning("Không tìm thấy nội dung văn bản nào trong tài liệu này.")
        return

    pending_doc = {
        "file_hash": file_hash,
        "content_hash": result["content_hash"],
        "source_file": uploaded_file.name,
        "replace": replace,
    }
    try:
        dup = find_document_by_hash(get_es_client(), content_hash=result["content_hash"])
    except Exception as e:
        st.warning(f"Không kiểm tra được tài liệu trùng ({e}), vẫn tiếp tục.")
        dup = None
    if dup and not (replace and dup["book_id"] == replace["book_id"]):
        st.session_state["upload_dup"] = {"existing": dup, "parsed": result, "pending_doc": pending_doc}
        st.rerun()

    clear_pending_review()
    start_review(result, pending_doc)


def render_duplicate_warning(dup):
    existing = dup["existing"]
    if existing["matched"] == "file":
        reason = "File này đã được thêm vào kho trước đây."
    else:
        reason = (
            "Nội dung tài liệu này trùng với một tài liệu đã có trong kho "
            "(có thể là cùng tài liệu được lưu lại hoặc xuất lại)."
        )
    with st.container(border=True):
        st.warning(reason)
        st.markdown(f"Tài liệu đã có: **{md_escape(existing['book_title'])}**")
        st.caption(document_meta_line(existing))
        replace_col, cancel_col = st.columns(2)
        replace_col.button(
            "Thay thế bản cũ", on_click=replace_existing, args=(dup,), width="stretch",
            help="Tiếp tục thêm bản mới; bản cũ chỉ bị xoá sau khi bản mới đã được thêm thành công.",
        )
        cancel_col.button("Huỷ", on_click=clear_duplicate_warning, width="stretch")


def render_review():
    tree = st.session_state["pending_tree"]
    metadata = st.session_state["pending_metadata"]
    pending_doc = st.session_state.get("pending_doc", {})
    replace = pending_doc.get("replace")

    st.markdown("#### Xem lại thông tin trước khi embed")
    if replace:
        st.info(
            f"Bản mới sẽ **thay thế** tài liệu **{md_escape(replace['book_title'])}** ({replace['num_chunks']} chunk). "
            "Bản cũ chỉ bị xoá sau khi bản mới đã được thêm thành công."
        )
    metadata["book_title"] = st.text_input("Tên tài liệu", value=metadata["book_title"])
    metadata["author"] = st.text_input("Tác giả (tuỳ chọn)", value=metadata["author"])
    st.caption(f"`book_id` sẽ được gán: `{metadata['book_id']}`")

    stats = tree_stats(tree)
    tables = tree_tables(tree)
    c1, c2, c3 = st.columns(3)
    c1.metric("Mục / tiêu đề", stats["headings"])
    c2.metric("Đoạn văn", stats["paragraphs"])
    c3.metric("Bảng", stats["tables"])

    st.markdown("**Cấu trúc phân cấp**")
    st.caption("Bấm vào một mục để mở/đóng. Số bên cạnh là nội dung nằm trực tiếp dưới mục đó.")
    with st.container(height=420 if stats["headings"] > 12 else "content", border=True):
        st.html(tree_html(tree))

    if tables:
        st.markdown(f"**Bảng phát hiện được ({len(tables)})**")
        st.caption(
            "Mỗi bảng được embed thành chunk riêng (bảng dài sẽ được chia theo hàng, giữ lại hàng tiêu đề). "
            "Bỏ chọn những bảng không muốn đưa vào hệ thống, ví dụ bảng dùng để dàn trang."
        )
        render_table_review(tables, metadata["book_id"])

    col1, col2 = st.columns(2)
    with col1:
        embed_label = "Xác nhận, embed và thay thế bản cũ" if replace else "Xác nhận và embed vào hệ thống"
        if st.button(embed_label, type="primary"):
            embed_pending_document(tree, metadata, pending_doc, tables)
    with col2:
        if st.button("Huỷ"):
            clear_pending_review()
            st.rerun()


def embed_pending_document(tree, metadata, pending_doc, tables):
    replace = pending_doc.get("replace")

    # One bar for the whole job. Embedding is by far the slowest step, so it gets
    # most of the bar (10% -> 90%) and advances after every batch.
    bar = st.progress(0.0, text="Đang chuẩn bị...")
    progress = {"value": 0.0}

    def set_progress(value, text):
        progress["value"] = value
        bar.progress(min(value, 1.0), text=text)

    def on_embed_progress(done, total):
        set_progress(0.10 + 0.80 * done / total, f"Đang embedding: {done}/{total} chunk")

    with st.status("Đang xử lý...", expanded=True) as status:
        try:
            set_progress(0.03, "Đang dựng cấu trúc chương/mục...")
            st.write("Đang dựng cấu trúc chương/mục cuối cùng...")
            excluded = {
                t["table_id"] for _, t in tables
                if not st.session_state.get(table_keep_key(metadata["book_id"], t["table_id"]), True)
            }
            book = tree_to_book(
                drop_tables(tree, excluded), **metadata,
                source_file=pending_doc.get("source_file"),
                uploaded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                file_hash=pending_doc.get("file_hash"),
                content_hash=pending_doc.get("content_hash"),
            )

            set_progress(0.06, "Đang chia nhỏ văn bản...")
            st.write("Đang chia nhỏ văn bản (chunking)...")
            get_tokenizer()  # trigger download/load before the progress line below
            chunks = build_chunk_records(book)
            n_table_chunks = sum(c["type"] == "table" for c in chunks)
            st.write(f"Đã tạo {len(chunks)} chunk ({n_table_chunks} chunk bảng).")

            st.write(
                f"Đang embedding bằng {settings.embed_model} (dense + sparse), "
                f"mỗi lượt {settings.embed_batch_size} chunk..."
            )
            set_progress(0.10, f"Đang embedding: 0/{len(chunks)} chunk")
            chunks = embed_chunks_batch(get_embed_model(), chunks, on_progress=on_embed_progress)

            set_progress(0.92, "Đang index vào Elasticsearch...")
            st.write("Đang tạo/kiểm tra index và index vào Elasticsearch...")
            es = get_es_client()
            ensure_index_exists(es)
            success_count, errors = index_chunks(es, chunks)

            if errors:
                set_progress(progress["value"], f"Dừng lại: {len(errors)} chunk lỗi khi index")
                status.update(label=f"Hoàn tất với {len(errors)} lỗi", state="error")
                st.error(f"{len(errors)} chunk lỗi khi index: {errors[:3]}")
                if replace:
                    st.warning(f"Tài liệu cũ **{md_escape(replace['book_title'])}** được giữ nguyên.")
                return

            replaced_note = ""
            if replace:
                set_progress(0.97, "Đang xoá bản cũ...")
                st.write("Đang xoá bản cũ...")
                n_old = delete_document(es, replace["book_id"])
                replaced_note = f", đã xoá bản cũ ({n_old} chunk)"

            set_progress(1.0, f"Hoàn tất: {success_count} chunk")
            status.update(label=f"Hoàn tất! Đã thêm {success_count} chunk{replaced_note}.", state="complete")
            st.success(
                f"Đã thêm tài liệu **{md_escape(metadata['book_title'])}** "
                f"({success_count} chunk, book_id=`{metadata['book_id']}`){replaced_note}."
            )
            clear_pending_review()
            get_document_list.clear()  # new doc should appear in the picker/library right away
        except Exception as e:
            set_progress(progress["value"], "Dừng lại do lỗi")
            status.update(label="Có lỗi xảy ra", state="error")
            st.error(f"Có lỗi xảy ra: {e}")
            if replace:
                st.warning(f"Tài liệu cũ **{md_escape(replace['book_title'])}** được giữ nguyên.")


def render_library(docs, error):
    notice = st.session_state.pop("library_notice", None)
    if notice:
        kind, text = notice
        (st.success if kind == "success" else st.error)(text)

    if error:
        st.error(f"Không lấy được danh sách tài liệu: {error}")
        return
    if not docs:
        st.info("Kho chưa có tài liệu nào. Thêm tài liệu ở tab bên cạnh.")
        return

    st.caption(f"{len(docs)} tài liệu · {sum(d['num_chunks'] for d in docs)} chunk")
    query = ""
    if len(docs) > 5:
        query = st.text_input(
            "Tìm tài liệu", placeholder="Lọc theo tên tài liệu hoặc tên file...", label_visibility="collapsed",
        ).strip().lower()
    shown = [
        d for d in docs
        if query in d["book_title"].lower() or query in (d.get("source_file") or "").lower()
    ]
    # Newest uploads first; documents indexed before upload info existed go last.
    shown.sort(key=lambda d: d.get("uploaded_at") or "", reverse=True)
    if not shown:
        st.caption("Không có tài liệu nào khớp.")

    for doc in shown:
        with st.container(border=True):
            info, action = st.columns([9, 1], vertical_alignment="center")
            with info:
                st.markdown(f"**{md_escape(doc['book_title'])}**")
                st.caption(document_meta_line(doc))
            with action:
                with st.popover("Xoá", help="Xoá tài liệu này"):
                    st.markdown(f"Xoá vĩnh viễn **{md_escape(doc['book_title'])}**?")
                    st.caption(
                        f"{doc['num_chunks']} chunk sẽ bị xoá khỏi kho, không hoàn tác được. "
                        "Các câu trả lời cũ trong lịch sử chat vẫn giữ nội dung đã trích dẫn."
                    )
                    st.button(
                        "Xoá vĩnh viễn", type="primary", key=f"delete_{doc['book_id']}", width="stretch",
                        on_click=delete_document_callback, args=(doc["book_id"], doc["book_title"]),
                    )


if page == PAGE_UPLOAD:
    st.subheader("Quản lý tài liệu")

    try:
        library_docs, library_error = get_document_list(get_es_client()), None
    except Exception as e:
        library_docs, library_error = [], e

    tab_add, tab_library = st.tabs(["Thêm tài liệu", "Tài liệu trong kho"], key="upload_tabs")

    with tab_add:
        st.caption("Hỗ trợ: PDF (cả text lẫn scan), DOCX, HTML, Markdown · tự nhận diện bảng · tự phát hiện tài liệu trùng")
        uploaded_file = st.file_uploader("Chọn file", type=[ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS])

        # A duplicate warning belongs to the file it was raised for: drop it as soon
        # as a different file is picked (or the file is removed).
        file_key = None
        if uploaded_file is not None:
            file_key = getattr(uploaded_file, "file_id", None) or f"{uploaded_file.name}:{uploaded_file.size}"
        if st.session_state.get("upload_file_key") != file_key:
            st.session_state["upload_file_key"] = file_key
            clear_duplicate_warning()

        replace_request = st.session_state.pop("upload_replace_request", None)
        if uploaded_file is not None and replace_request:
            parse_and_check(uploaded_file, file_sha256(uploaded_file.getvalue()), replace=replace_request)

        if "upload_dup" in st.session_state:
            render_duplicate_warning(st.session_state["upload_dup"])
        elif uploaded_file is not None and st.button("Xử lý tài liệu"):
            file_hash = file_sha256(uploaded_file.getvalue())
            try:
                dup = find_document_by_hash(get_es_client(), file_hash=file_hash)
            except Exception as e:
                st.warning(f"Không kiểm tra được tài liệu trùng ({e}), vẫn tiếp tục xử lý.")
                dup = None
            if dup:
                # Same bytes already indexed: stop before the (slow) Docling parse.
                st.session_state["upload_dup"] = {"existing": dup, "file_hash": file_hash}
                st.rerun()
            parse_and_check(uploaded_file, file_hash)

        if "pending_tree" in st.session_state:
            render_review()

    with tab_library:
        render_library(library_docs, library_error)
