import os
import tempfile
from pathlib import Path

import streamlit as st

from chunking import build_chunk_records, get_tokenizer
from document_ingest import SUPPORTED_EXTENSIONS, ingest_document, tree_to_book
from langgraph_agent import GatsbyAgent, extract_text
from rag_pipeline import (
    embed_chunks_batch,
    ensure_index_exists,
    index_chunks,
    list_documents,
    load_embed_model,
    load_es_client,
    load_reranker,
)

st.set_page_config(page_title="Gatsby RAG", page_icon="📖")
st.title("📖 Hỏi đáp về The Great Gatsby")
st.caption("Hybrid search (bge-m3 dense + sparse) + rerank (bge-reranker-v2-m3) + LangGraph agent (Gemini)")


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
    return GatsbyAgent(get_es_client(), get_embed_model(), get_reranker())


@st.cache_data(ttl=300, show_spinner=False)
def get_document_list(_es):
    # Leading underscore: tells st.cache_data not to try hashing the ES client.
    return list_documents(_es)


try:
    agent = get_agent()
except Exception as e:
    st.error(f"Không khởi tạo được hệ thống: {e}")
    st.stop()


tab_ask, tab_upload = st.tabs(["💬 Hỏi đáp", "📤 Thêm tài liệu mới"])

with tab_ask:
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

    query_text = st.text_input(
        "Đặt câu hỏi:",
        placeholder="What does the green light symbolize?",
    )

    if st.button("Hỏi", type="primary") and query_text.strip():
        step_num = 0
        final_answer = None
        had_error = False
        book_id = selected_doc["book_id"] if selected_doc else None
        book_title = selected_doc["book_title"] if selected_doc else None

        with st.status("Agent đang suy luận...", expanded=True) as status:
            try:
                for event in agent.stream(query_text, book_id=book_id, book_title=book_title):
                    node = event["node"]
                    for msg in event["messages"]:
                        if node == "agent":
                            tool_calls = getattr(msg, "tool_calls", None) or []
                            if tool_calls:
                                for tc in tool_calls:
                                    step_num += 1
                                    sub_query = tc["args"].get("query", "")
                                    st.write(f"**Bước {step_num}** — 🔍 LLM quyết định tìm kiếm: _{sub_query!r}_")
                            elif msg.content:
                                # An AIMessage with no tool_calls is the agent's final answer.
                                final_answer = extract_text(msg.content)
                        elif node == "tools" and agent.last_calls:
                            last_call = agent.last_calls[-1]
                            n_hits, n_relevant = last_call["num_hits"], last_call["num_relevant"]
                            if n_relevant == 0 and n_hits > 0:
                                st.write(f"　　↳ tìm thấy {n_hits} đoạn nhưng không đủ liên quan (bị lọc bỏ)")
                            else:
                                st.write(f"　　↳ tìm thấy {n_hits} đoạn, {n_relevant} đoạn đủ liên quan")
            except Exception as e:
                had_error = True
                status.update(label="Có lỗi xảy ra", state="error")
                st.error(f"Có lỗi xảy ra: {e}")
            else:
                status.update(label=f"Hoàn tất sau {step_num} lượt tìm kiếm", state="complete")

        if not had_error and final_answer:
            st.markdown("### Trả lời")
            st.markdown(final_answer)

            with st.expander(f"Nguồn trích dẫn ({len(agent.last_sources)} đoạn)"):
                for i, c in enumerate(agent.last_sources):
                    src = c["hit"]["_source"]
                    text = src["text"]
                    location = f"Chapter {src['chapter_title']}"
                    if src.get("section_title"):
                        location += f" > {src['section_title']}"
                    st.markdown(
                        f"**{src['book_title']}** — {location}  \n"
                        f"`{src['chunk_id']}` · rerank score={c['rerank_score']:.4f}"
                    )
                    is_long = len(text) > 300
                    st.caption(text[:300] + ("..." if is_long else ""))
                    if is_long:
                        with st.popover("Xem đầy đủ"):
                            st.write(text)
                    st.divider()


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


with tab_upload:
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
