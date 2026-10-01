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
from document_ingest import SUPPORTED_EXTENSIONS, ingest_document, tree_to_book
from langgraph_agent import DocumentQAAgent, extract_text
from rag_pipeline import (
    embed_chunks_batch,
    ensure_index_exists,
    index_chunks,
    list_documents,
    load_embed_model,
    load_es_client,
    load_reranker,
)

MAX_HISTORY_MESSAGES = 6

PAGE_CHAT = "Trò chuyện"
PAGE_UPLOAD = "Thêm tài liệu"

st.set_page_config(page_title="RAG", page_icon="📖")


@st.cache_resource(show_spinner="Đang kết nối Elasticsearch...")
def get_es_client():
    return load_es_client()


@st.cache_resource(show_spinner="Đang load model embedding (bge-m3)...")
def get_embed_model():
    return load_embed_model()


@st.cache_resource(show_spinner="Đang load model reranker (bge-reranker-v2-m3)...")
def get_reranker():
    return load_reranker()


@st.cache_resource(show_spinner="Đang khởi tạo agent (LangGraph + Gemini)...")
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
            st.markdown(
                f"**{s['book_title']}** — {s['location']}  \n"
                f"`{s['chunk_id']}` · rerank score={s['rerank_score']:.4f}"
            )
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


def render_tree(node, depth=0):
    """Render a heading node and its children with visual indentation, so the real
    parent/child relationship (e.g. an empty "1.2." containing "1.2.1."/"1.2.2." as
    children) is obvious -- Streamlit can't nest st.expander, so indentation + icons
    are used instead of true collapsible nesting."""
    indent = "&nbsp;&nbsp;&nbsp;&nbsp;" * depth
    icon = "📁" if node["children"] else "📄"
    n_direct = len(node["paragraphs"])
    label = node["title"] if depth > 0 else f"**{node['title']}**"
    st.markdown(f"{indent}{icon} {label} &nbsp;·&nbsp; _{n_direct} đoạn trực tiếp_", unsafe_allow_html=True)
    if node["paragraphs"]:
        preview = node["paragraphs"][0][:150]
        suffix = "..." if len(node["paragraphs"][0]) > 150 else ""
        st.markdown(f"{indent}&nbsp;&nbsp;↳ <span style='color:gray'>{preview}{suffix}</span>", unsafe_allow_html=True)
    for child in node["children"]:
        render_tree(child, depth + 1)


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
    st.caption("Hybrid search (bge-m3 dense + sparse) · rerank (bge-reranker-v2-m3) · LangGraph agent (Gemini)")


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

        book_id = selected_doc["book_id"] if selected_doc else None
        book_title = selected_doc["book_title"] if selected_doc else None

        with st.chat_message("assistant"):
            # Created in this order so the step log sits above the streaming answer.
            status = st.status("Đang suy luận...", expanded=True)
            answer_placeholder = st.empty()
            streamed_text = ""
            final_answer = None
            step_num = 0

            try:
                for event in agent.stream(prompt, book_id=book_id, book_title=book_title, history=history):
                    if event["type"] == "token":
                        streamed_text += event["text"]
                        answer_placeholder.markdown(streamed_text + "▌")
                        continue

                    node = event["node"]
                    for msg in event["messages"]:
                        if node == "agent":
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
                        elif node == "tools" and agent.last_calls:
                            n_hits = agent.last_calls[-1]["num_hits"]
                            status.write(f"　　↳ tìm thấy {n_hits} đoạn liên quan")
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
                    "tool_calls": list(agent.last_calls),
                    "sources": to_source_records(agent.last_sources),
                    "scope": book_title,
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
    st.caption("Hỗ trợ: PDF (cả text lẫn scan), DOCX, HTML, Markdown")

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
            with st.spinner("Đang phân tích cấu trúc tài liệu (unstructured)..."):
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
            if not result["tree"]["children"] and not result["tree"]["paragraphs"]:
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

        def count_all_paragraphs(node):
            return len(node["paragraphs"]) + sum(count_all_paragraphs(c) for c in node["children"])

        def count_all_headings(node):
            return len(node["children"]) + sum(count_all_headings(c) for c in node["children"])

        st.write(
            f"Phát hiện **{count_all_headings(tree)} mục/tiêu đề** (đã lồng đúng cấp bậc), "
            f"tổng **{count_all_paragraphs(tree)} đoạn văn**."
        )
        st.markdown("**Cấu trúc phân cấp phát hiện được:**")
        with st.container(border=True):
            if tree["paragraphs"]:
                st.markdown(f"📄 *(Văn bản trước tiêu đề đầu tiên)* &nbsp;·&nbsp; _{len(tree['paragraphs'])} đoạn_")
            for child in tree["children"]:
                render_tree(child, depth=0)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ 3️⃣ Xác nhận và Embed vào hệ thống", type="primary"):
                with st.status("Đang xử lý...", expanded=True) as status:
                    try:
                        st.write("Đang dựng cấu trúc chương/mục cuối cùng...")
                        book = tree_to_book(tree, **metadata)

                        st.write("Đang chia nhỏ văn bản (chunking)...")
                        get_tokenizer()  # trigger download/load before the progress line below
                        chunks = build_chunk_records(book)
                        st.write(f"Đã tạo {len(chunks)} chunk.")

                        st.write("Đang embedding bằng bge-m3 (dense + sparse)...")
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
