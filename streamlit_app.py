import os
import tempfile
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
    get_converter,
    ingest_document,
    tree_html,
    tree_stats,
    tree_tables,
    tree_to_book,
)
from langgraph_agent import DocumentQAAgent, RunContext, extract_text
from rag_pipeline import (
    embed_chunks_batch,
    ensure_index_exists,
    index_chunks,
    list_documents,
    load_embed_model,
    load_es_client,
    load_reranker,
)

MAX_HISTORY_MESSAGES = settings.max_history_messages

PAGE_CHAT = "Trò chuyện"
PAGE_UPLOAD = "Thêm tài liệu"

st.set_page_config(page_title="RAG", page_icon="📖")


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
        location = f"Chapter {src['chapter_title']}"
        if src.get("section_title"):
            location += f" > {src['section_title']}"
        records.append({
            "type": src.get("type", "paragraph"),
            "book_title": src["book_title"],
            "location": location,
            "chunk_id": src["chunk_id"],
            "rerank_score": c["rerank_score"],
            "text": src["text"],
        })
    return records


def render_steps(tool_calls):
    with st.expander(f"🧠 Quá trình suy luận ({len(tool_calls)} lượt tìm kiếm)"):
        for i, call in enumerate(tool_calls, start=1):
            st.markdown(f"**Bước {i}** — 🔍 _{call['query']!r}_ → {call['num_hits']} đoạn")


def render_sources(sources):
    with st.expander(f"📚 Nguồn trích dẫn ({len(sources)} đoạn)"):
        for s in sources:
            is_table = s.get("type") == "table"
            st.markdown(
                f"{'📊 ' if is_table else ''}**{s['book_title']}** — {s['location']}  \n"
                f"`{s['chunk_id']}` · rerank score={s['rerank_score']:.4f}"
            )
            if is_table:
                # The first line of a table chunk is its caption (or heading) line.
                first_line = s["text"].split("\n", 1)[0]
                st.caption(first_line if not first_line.startswith("|") else "Bảng")
                with st.popover("📊 Xem bảng"):
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
        with st.expander(f"📊 {name} · {dims}"):
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
    st.title("📖 Hỏi đáp tài liệu")
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

        st.button("➕ Cuộc trò chuyện mới", on_click=start_new_conversation, width="stretch")

        st.markdown("**Lịch sử trò chuyện**")
        # Filled at the end of the chat page, so a conversation created by the
        # message just sent already shows up in the list on this same run.
        conversation_list = st.container()

        if st.session_state["conversation_id"]:
            with st.popover("🗑️ Xoá cuộc trò chuyện này", width="stretch"):
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
                            status.write(f"　　↳ _{call['query']!r}_: tìm thấy {call['num_hits']} đoạn liên quan")
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
                                status.write(f"**Bước {step_num}** — 🔍 LLM quyết định tìm kiếm: _{sub_query!r}_")
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

if page == PAGE_UPLOAD:
    st.subheader("Thêm tài liệu mới vào hệ thống")
    st.caption("Hỗ trợ: PDF (cả text lẫn scan), DOCX, HTML, Markdown · tự nhận diện bảng")

    uploaded_file = st.file_uploader(
        "Chọn file",
        type=[ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS],
    )

    if uploaded_file is not None and st.button("1️⃣ Xử lý tài liệu"):
        suffix = Path(uploaded_file.name).suffix
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        try:
            get_doc_converter()  # first use: slow Docling import, shown with its own spinner
            with st.spinner("Đang phân tích cấu trúc tài liệu (Docling)..."):
                try:
                    result = ingest_document(tmp_path, uploaded_file.name)
                except Exception as e:
                    st.error(f"Không phân tích được tài liệu: {e}")
                    result = None
        finally:
            # ingest_document only needs the file on disk during parsing -- always
            # clean up the temp file afterward, even if parsing raised.
            try:
                os.remove(tmp_path)
            except OSError:
                pass

        if result is not None:
            if not result["tree"]["children"] and not result["tree"]["blocks"]:
                st.warning("Không tìm thấy nội dung văn bản nào trong tài liệu này.")
            else:
                st.session_state["pending_tree"] = result["tree"]
                st.session_state["pending_metadata"] = result["metadata"]

    # --- Review step: only shown after a document has been processed ---
    if "pending_tree" in st.session_state:
        tree = st.session_state["pending_tree"]
        metadata = st.session_state["pending_metadata"]

        st.markdown("#### 2️⃣ Xem lại thông tin trước khi embed")
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
            if st.button("✅ 3️⃣ Xác nhận và Embed vào hệ thống", type="primary"):
                with st.status("Đang xử lý...", expanded=True) as status:
                    try:
                        st.write("Đang dựng cấu trúc chương/mục cuối cùng...")
                        excluded = {
                            t["table_id"] for _, t in tables
                            if not st.session_state.get(table_keep_key(metadata["book_id"], t["table_id"]), True)
                        }
                        book = tree_to_book(drop_tables(tree, excluded), **metadata)

                        st.write("Đang chia nhỏ văn bản (chunking)...")
                        get_tokenizer()  # trigger download/load before the progress line below
                        chunks = build_chunk_records(book)
                        n_table_chunks = sum(c["type"] == "table" for c in chunks)
                        st.write(f"Đã tạo {len(chunks)} chunk ({n_table_chunks} chunk bảng).")

                        st.write(f"Đang embedding bằng {settings.embed_model} (dense + sparse)...")
                        chunks = embed_chunks_batch(get_embed_model(), chunks)

                        st.write("Đang tạo/kiểm tra index và index vào Elasticsearch...")
                        ensure_index_exists(get_es_client())
                        success_count, errors = index_chunks(get_es_client(), chunks)

                        if errors:
                            status.update(label=f"Hoàn tất với {len(errors)} lỗi", state="error")
                            st.error(f"{len(errors)} chunk lỗi khi index: {errors[:3]}")
                        else:
                            status.update(label=f"Hoàn tất! Đã thêm {success_count} chunk.", state="complete")
                            st.success(
                                f"Đã thêm tài liệu **{metadata['book_title']}** "
                                f"({success_count} chunk, book_id=`{metadata['book_id']}`) vào hệ thống."
                            )
                            del st.session_state["pending_tree"]
                            del st.session_state["pending_metadata"]
                            get_document_list.clear()  # new doc should appear in the picker right away
                    except Exception as e:
                        status.update(label="Có lỗi xảy ra", state="error")
                        st.error(f"Có lỗi xảy ra: {e}")
        with col2:
            if st.button("❌ Huỷ"):
                del st.session_state["pending_tree"]
                del st.session_state["pending_metadata"]
                st.rerun()
