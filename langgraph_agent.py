"""
LangGraph agent that wraps the hybrid search + rerank pipeline as a single tool.

Workflow (cyclic graph, not a one-shot pipeline):

    user query -> agent (LLM) --decides--> [tool: search_documents] -> agent (LLM) -> ... -> final answer
                       ^                                                   |
                       +---------------------------------------------------+
                       (loops back until the LLM answers without calling the tool again)

The LLM itself decides how many times to call search_documents and with what query --
including splitting a complex question into several narrower searches (query
decomposition) -- instead of a fixed single retrieve-then-answer pass.

Run directly for debugging, outside Streamlit:
    python langgraph_agent.py
"""

import os
import threading
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from config import settings
from rag_pipeline import (
    EMBED_MODEL_NAME,
    GEMINI_MODEL,
    RERANKER_MODEL_NAME,
    build_context,
    embed_query,
    knn_search,
    load_embed_model,
    load_es_client,
    load_reranker,
    reciprocal_rank_fusion,
    rerank,
    sparse_search,
)
from tracing import flush, make_callback_handler, traced_step

# Retrieval settings for one search_documents call (from .env, see config.py).
DENSE_K = settings.dense_k
SPARSE_K = settings.sparse_k
FUSED_TOP_N = settings.fused_top_n
FINAL_TOP_N = settings.final_top_n
TRACE_NAME = settings.trace_name


def extract_text(content) -> str:
    """LangChain message content is usually a plain string, but some providers
    (e.g. Gemini via langchain-google-genai) can instead return a list of content
    blocks, each like {"type": "text", "text": "...", "extras": {"signature": ...}}.
    This pulls out just the actual text, discarding provider-specific metadata."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content)


SYSTEM_PROMPT = """Bạn là trợ lý trả lời câu hỏi dựa trên kho tài liệu đã được index trong hệ thống -- kho này có thể chứa nhiều tài liệu khác nhau (sách, báo cáo, tài liệu,...), không chỉ riêng một cuốn.

Bạn có 1 công cụ: `search_documents`, dùng để tìm các đoạn trích liên quan trong toàn bộ kho tài liệu. Quy tắc:
- Luôn gọi `search_documents` ít nhất 1 lần trước khi trả lời -- không được trả lời từ kiến thức có sẵn của bạn.
- Với câu hỏi nối tiếp trong cuộc trò chuyện (dùng "nó", "chương đó", "còn ... thì sao?"), hãy dựa vào các lượt trước để hiểu người dùng đang hỏi gì, rồi viết truy vấn `search_documents` đầy đủ ngữ cảnh (nêu rõ tên nhân vật/chủ đề/tài liệu), không truyền nguyên câu hỏi cụt.
- Nếu câu hỏi phức tạp hoặc cần thông tin từ nhiều khía cạnh khác nhau (ví dụ so sánh 2 nhân vật, nhiều sự kiện/tài liệu khác nhau), hãy TÁCH thành các câu hỏi con và gọi `search_documents` riêng cho từng câu hỏi con.
- Nếu kết quả tìm kiếm đầu tiên chưa đủ để trả lời trọn vẹn, hãy gọi lại `search_documents` với một truy vấn khác (diễn đạt lại, hoặc tập trung vào khía cạnh còn thiếu) thay vì trả lời ngay với thông tin chưa đủ.
- Chỉ trả lời dựa trên nội dung trả về từ `search_documents` -- KHÔNG bịa thêm chi tiết không có trong đó.
- Mỗi đoạn trích đều ghi rõ tên tài liệu nguồn -- nếu kết quả có đoạn từ nhiều tài liệu khác nhau, hãy phân biệt rõ thông tin nào đến từ tài liệu nào, không gộp lẫn.
- Nếu sau khi tìm kiếm vẫn không có đủ thông tin, hãy nói rõ là không tìm thấy trong kho tài liệu, đừng đoán bừa.
- Khi trả lời, trích dẫn tên tài liệu và chương/mục liên quan (ví dụ: "(The Great Gatsby, Chapter III)") cho các chi tiết quan trọng.
"""


@dataclass
class RunContext:
    """Everything that belongs to ONE question: its search scope, and what its
    search_documents calls found. A fresh one is created per ask()/stream() call and
    handed to the tool through LangGraph's per-invocation config, so concurrent
    questions (other users, other browser tabs) never see or overwrite each other's
    state -- the agent object itself stays shared and holds no per-question data.
    """

    book_id: str | None = None      # None = search the whole library
    book_title: str | None = None   # shown to the LLM so it knows the scope
    conversation_id: str | None = None  # groups this question's trace with its conversation
    # search_documents must return a plain string (the ToolMessage the LLM reads), so
    # the raw hits -- needed for source citations -- are collected here instead.
    sources: list = field(default_factory=list)
    calls: list = field(default_factory=list)
    # chunk_ids already returned during this question, so a chunk matching several
    # sub-queries is neither cited twice nor sent to the LLM twice.
    seen_chunk_ids: set = field(default_factory=set)
    # The LLM can request several searches in one turn and ToolNode runs them in
    # parallel; the lock keeps the dedup check-and-record step atomic between them.
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


def _location(src) -> str:
    location = f"Chapter {src['chapter_title']}"
    if src.get("section_title"):
        location += f" > {src['section_title']}"
    return location


def _hit_row(rank, hit, **scores):
    """One readable line of a ranked result list in the trace (no passage text)."""
    src = hit["_source"]
    return {"rank": rank, "chunk_id": src["chunk_id"], "document": src["book_title"],
            "location": _location(src), **scores}


class DocumentQAAgent:
    """Bundles the compiled LangGraph graph with the resources search_documents needs.

    Construct once (heavy: builds the LLM + graph) and share it freely -- it holds no
    per-question state, so concurrent `.ask()`/`.stream()` calls are safe. Each call
    gets its own RunContext.
    """

    def __init__(self, es, embed_model, reranker, google_api_key=None, model=GEMINI_MODEL):
        api_key = google_api_key or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Chưa có GOOGLE_API_KEY trong biến môi trường/.env. "
                "Lấy API key tại https://aistudio.google.com/apikey"
            )

        @tool
        def search_documents(query: str, config: RunnableConfig) -> str:
            """Search the indexed document library (which may contain multiple
            different documents/books) for passages relevant to the given query. Use
            this to look up specific details, quotes, or context. Call it again with a
            reformulated or narrower query if the first result does not fully answer
            the question."""
            # `config` is injected by LangGraph and hidden from the LLM -- the model
            # only ever sees and fills in `query`.
            ctx: RunContext = config["configurable"]["run_context"]
            scope = ctx.book_title or "all documents"

            # Same pipeline as rag_pipeline.search(), run step by step so each step is
            # its own span in the trace (with timing and a readable result table).
            # Searches run outside ctx.lock, so parallel searches still overlap.
            dense_vec, sparse_weights = traced_step(
                "1. embed_query (bge-m3)", {"query": query},
                lambda: embed_query(embed_model, query),
                lambda r: {
                    "dense_dims": len(r[0]),
                    "top_sparse_tokens": {
                        tok: round(w, 3) for tok, w in sorted(r[1].items(), key=lambda kv: -kv[1])[:10]
                    },
                },
            )
            dense_hits = traced_step(
                "2a. dense_knn_search",
                {"k": DENSE_K, "num_candidates": DENSE_K * 5, "scope": scope},
                lambda: knn_search(es, dense_vec, k=DENSE_K, num_candidates=DENSE_K * 5, book_id=ctx.book_id),
                lambda hits: [_hit_row(i, h, knn_score=round(h["_score"], 4)) for i, h in enumerate(hits, 1)],
            )
            sparse_hits = traced_step(
                "2b. sparse_search (rank_feature)", {"size": SPARSE_K, "scope": scope},
                lambda: sparse_search(es, sparse_weights, size=SPARSE_K, book_id=ctx.book_id),
                lambda hits: [_hit_row(i, h, sparse_score=round(h["_score"], 4)) for i, h in enumerate(hits, 1)],
            )
            dense_rank = {h["_id"]: i for i, h in enumerate(dense_hits, 1)}
            sparse_rank = {h["_id"]: i for i, h in enumerate(sparse_hits, 1)}
            fused = traced_step(
                "3. rrf_fusion",
                {"rrf_k": settings.rrf_k, "dense_hits": len(dense_hits), "sparse_hits": len(sparse_hits), "keep_top": FUSED_TOP_N},
                lambda: reciprocal_rank_fusion([dense_hits, sparse_hits])[:FUSED_TOP_N],
                lambda entries: [
                    _hit_row(
                        i, e["hit"], rrf_score=round(e["rrf_score"], 5),
                        dense_rank=dense_rank.get(e["hit"]["_id"]), sparse_rank=sparse_rank.get(e["hit"]["_id"]),
                    )
                    for i, e in enumerate(entries, 1)
                ],
            )
            ranked = traced_step(
                "4. rerank (bge-reranker-v2-m3)",
                {"query": query, "candidates": len(fused), "keep_top": FINAL_TOP_N},
                lambda: rerank(reranker, query, fused, top_n=FINAL_TOP_N),
                lambda entries: [
                    _hit_row(i, e["hit"], rerank_score=round(float(e["rerank_score"]), 4), rrf_score=round(e["rrf_score"], 5))
                    for i, e in enumerate(entries, 1)
                ],
            )

            def dedup():
                with ctx.lock:
                    new = [c for c in ranked if c["hit"]["_source"]["chunk_id"] not in ctx.seen_chunk_ids]
                    for c in new:
                        ctx.seen_chunk_ids.add(c["hit"]["_source"]["chunk_id"])
                    ctx.calls.append({"query": query, "num_hits": len(new)})
                    ctx.sources.extend(new)
                return new

            new_results = traced_step(
                "5. dedup_against_previous_searches",
                {"reranked": [c["hit"]["_source"]["chunk_id"] for c in ranked]},
                dedup,
                lambda new: {
                    "kept": [c["hit"]["_source"]["chunk_id"] for c in new],
                    "dropped_already_seen": [
                        c["hit"]["_source"]["chunk_id"] for c in ranked if c not in new
                    ],
                },
            )

            if not new_results:
                return "(Không có đoạn mới nào -- các đoạn khớp với truy vấn này đã được tìm thấy ở lần tìm kiếm trước.)"
            return build_context(new_results)

        self.model_name = model
        llm = ChatGoogleGenerativeAI(model=model, temperature=settings.llm_temperature, google_api_key=api_key)
        # run_name: shows up as e.g. "gemini-2.5-flash" in the trace instead of the class name.
        llm_with_tools = llm.bind_tools([search_documents]).with_config(run_name=model)

        def call_agent(state: MessagesState):
            response = llm_with_tools.invoke(state["messages"])
            return {"messages": [response]}

        graph_builder = StateGraph(MessagesState)
        graph_builder.add_node("agent", call_agent)
        graph_builder.add_node("tools", ToolNode([search_documents]))
        graph_builder.add_edge(START, "agent")
        graph_builder.add_conditional_edges("agent", tools_condition)  # -> "tools" if a tool call was requested, else END
        graph_builder.add_edge("tools", "agent")  # after running the tool, always loop back to the agent

        self.graph = graph_builder.compile()

    def _run_config(self, ctx: RunContext, history, handler):
        config = {"configurable": {"run_context": ctx}, "run_name": TRACE_NAME}
        if handler is None:
            return config

        # Keys prefixed "langfuse_" become trace attributes (name, session, tags);
        # the rest is trace metadata. Langfuse coerces metadata values to strings and
        # caps them at 200 chars, so only short, scalar facts go here -- the detailed
        # per-step data lives in each step span's input/output instead.
        metadata = {
            "langfuse_trace_name": TRACE_NAME,
            "langfuse_tags": [
                f"scope:{ctx.book_id or 'all'}",
                "follow-up" if history else "first-question",
            ],
            "scope_document": (ctx.book_title or "all documents")[:200],
            "history_messages": len(history or []),
            "llm_model": self.model_name,
            "embed_model": EMBED_MODEL_NAME,
            "reranker_model": RERANKER_MODEL_NAME,
            "retrieval_params": (
                f"dense_k={DENSE_K} sparse_k={SPARSE_K} fused_top_n={FUSED_TOP_N} final_top_n={FINAL_TOP_N}"
            ),
        }
        if ctx.conversation_id:
            metadata["langfuse_session_id"] = ctx.conversation_id
        config["callbacks"] = [handler]
        config["metadata"] = metadata
        return config

    def _initial_state(self, query_text: str, book_title=None, history=None):
        messages = [SystemMessage(content=SYSTEM_PROMPT)]
        if book_title:
            messages.append(SystemMessage(content=(
                f'Phạm vi tìm kiếm của câu hỏi này đã được giới hạn trong tài liệu "{book_title}". '
                "Mọi kết quả `search_documents` trả về đều chỉ thuộc tài liệu này."
            )))
        # Earlier turns are passed as plain text only (no tool calls/results) -- enough
        # for the LLM to resolve follow-up questions without replaying old searches.
        for turn in history or []:
            if turn["role"] == "user":
                messages.append(HumanMessage(content=turn["content"]))
            else:
                messages.append(AIMessage(content=turn["content"]))
        messages.append(HumanMessage(content=query_text))
        return {"messages": messages}

    def ask(self, query_text: str, book_id=None, book_title=None, history=None):
        """Non-streaming: run the full agent loop, return (answer, sources, tool_calls).
        book_id=None searches the whole library; otherwise only that document.
        history: earlier chat turns as [{"role": "user"|"assistant", "content": str}, ...]."""
        ctx = RunContext(book_id=book_id, book_title=book_title)
        result = self.graph.invoke(
            self._initial_state(query_text, ctx.book_title, history),
            config=self._run_config(ctx, history, make_callback_handler()),
        )
        final_answer = extract_text(result["messages"][-1].content)
        return final_answer, ctx.sources, ctx.calls

    def stream(self, query_text: str, ctx: RunContext, history=None):
        """Streaming: yields two kinds of events, in order:

        - {"type": "token", "text": str} -- a piece of text as the LLM generates it
          inside the agent node. Usually this is the final answer, but it can also be
          a short preamble in a turn that ends up calling a tool -- the caller should
          discard buffered tokens when the following "update" turns out to contain
          tool_calls.
        - {"type": "update", "node": "agent"|"tools", "messages": [...]} -- a finished
          graph step (the full AIMessage with tool_calls/final text, or the tool
          results), for showing how the LLM broke down the query.

        ctx: the caller creates a fresh RunContext(book_id=..., book_title=...) for this
        question and keeps a reference to it -- ctx.calls / ctx.sources fill up while
        the generator runs (readable live, e.g. after each "tools" update) and hold the
        complete results once it's exhausted.
        history: earlier chat turns as [{"role": "user"|"assistant", "content": str}, ...].
        """
        initial_state = self._initial_state(query_text, ctx.book_title, history)
        for mode, payload in self.graph.stream(
            initial_state,
            config=self._run_config(ctx, history, make_callback_handler()),
            stream_mode=["updates", "messages"],
        ):
            if mode == "messages":
                chunk, metadata = payload
                # Only incremental LLM output from the agent node -- skips ToolMessages
                # and any non-chunk full message, which the "update" event covers.
                if metadata.get("langgraph_node") != "agent" or not isinstance(chunk, AIMessageChunk):
                    continue
                text = extract_text(chunk.content)
                if text:
                    yield {"type": "token", "text": text}
            else:
                for node_name, node_output in payload.items():
                    yield {"type": "update", "node": node_name, "messages": node_output["messages"]}


if __name__ == "__main__":
    es = load_es_client()
    embed_model = load_embed_model()
    reranker = load_reranker()
    agent = DocumentQAAgent(es, embed_model, reranker)

    demo_queries = [
        "What does the green light symbolize?",
        "So sánh cách Tom đối xử với Daisy và với Myrtle",
    ]
    for q in demo_queries:
        print("=" * 80)
        print("Query:", q)
        answer, sources, tool_calls = agent.ask(q)
        print("\nAnswer:\n", answer)
        print(f"\nsearch_documents was called {len(tool_calls)} time(s):")
        for i, call in enumerate(tool_calls, start=1):
            print(f"  {i}. query={call['query']!r} -> {call['num_hits']} hits")
        print(f"\nTotal source hits across all calls ({len(sources)}):")
        for c in sources:
            src = c["hit"]["_source"]
            print(f"  - {src['chunk_id']} (Chapter {src['chapter_title']}, rerank_score={c['rerank_score']:.4f})")

    # Langfuse sends traces in the background in batches -- send what's left before
    # this short-lived script exits (the long-running Streamlit app doesn't need this).
    flush()
