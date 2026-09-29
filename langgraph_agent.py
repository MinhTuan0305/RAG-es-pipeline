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

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from rag_pipeline import (
    GEMINI_MODEL,
    build_context,
    load_embed_model,
    load_es_client,
    load_reranker,
    search,
)

load_dotenv()


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


# bge-reranker-v2-m3 with normalize=True outputs a sigmoid score in [0, 1]. Chunks
# scoring below this are treated as "not actually relevant" and withheld from the
# LLM (rather than trusting the prompt alone to notice weak context and not
# hallucinate). Tune this based on observed scores -- genuinely relevant Gatsby
# passages in testing scored ~0.3-0.8, so 0.2 leaves some margin without being so
# strict that borderline-but-useful matches get discarded.
MIN_RELEVANCE_SCORE = 0.2

SYSTEM_PROMPT = """Bạn là trợ lý trả lời câu hỏi dựa trên kho tài liệu đã được index trong hệ thống -- kho này có thể chứa nhiều tài liệu khác nhau (sách, báo cáo, tài liệu,...), không chỉ riêng một cuốn.

Bạn có 1 công cụ: `search_documents`, dùng để tìm các đoạn trích liên quan trong toàn bộ kho tài liệu. Quy tắc:
- Luôn gọi `search_documents` ít nhất 1 lần trước khi trả lời -- không được trả lời từ kiến thức có sẵn của bạn.
- Nếu câu hỏi phức tạp hoặc cần thông tin từ nhiều khía cạnh khác nhau (ví dụ so sánh 2 nhân vật, nhiều sự kiện/tài liệu khác nhau), hãy TÁCH thành các câu hỏi con và gọi `search_documents` riêng cho từng câu hỏi con.
- Nếu kết quả tìm kiếm đầu tiên chưa đủ để trả lời trọn vẹn, hãy gọi lại `search_documents` với một truy vấn khác (diễn đạt lại, hoặc tập trung vào khía cạnh còn thiếu) thay vì trả lời ngay với thông tin chưa đủ.
- Chỉ trả lời dựa trên nội dung trả về từ `search_documents` -- KHÔNG bịa thêm chi tiết không có trong đó.
- Mỗi đoạn trích đều ghi rõ tên tài liệu nguồn -- nếu kết quả có đoạn từ nhiều tài liệu khác nhau, hãy phân biệt rõ thông tin nào đến từ tài liệu nào, không gộp lẫn.
- Nếu sau khi tìm kiếm vẫn không có đủ thông tin, hãy nói rõ là không tìm thấy trong kho tài liệu, đừng đoán bừa.
- Khi trả lời, trích dẫn tên tài liệu và chương/mục liên quan (ví dụ: "(The Great Gatsby, Chapter III)") cho các chi tiết quan trọng.
"""


class GatsbyAgent:
    """Bundles the compiled LangGraph graph with the resources search_documents needs.

    Construct once (heavy: builds the LLM + graph), reuse `.ask()`/`.stream()` for
    every query. Not thread-safe for truly concurrent requests -- `last_sources`/
    `last_calls`/`_seen_chunk_ids` are shared mutable state reset at the start of each
    call, which is fine for a single-user local app but would need per-request scoping
    for a multi-user deployment.
    """

    def __init__(self, es, embed_model, reranker, google_api_key=None, model=GEMINI_MODEL):
        api_key = google_api_key or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Chưa có GOOGLE_API_KEY trong biến môi trường/.env. "
                "Lấy API key tại https://aistudio.google.com/apikey"
            )

        # search_documents must return a plain string (that's what becomes the
        # ToolMessage content the LLM reads), so the raw hit objects -- needed later to
        # show proper source citations -- are collected here as a side channel instead.
        self.last_sources: list = []
        self.last_calls: list = []
        # Tracks chunk_ids already surfaced in this ask()/stream() call. When the agent
        # splits a question into several search_documents calls, the same chunk can
        # legitimately match more than one sub-query -- without this, it would show up
        # twice in the citation list AND get sent to the LLM twice (wasting context
        # tokens on a passage it already has in the conversation history).
        self._seen_chunk_ids: set = set()
        # Document scope for the current call (None = whole library). Chosen by the
        # user in the UI, not by the LLM, so it's not a tool argument.
        self._book_id = None

        @tool
        def search_documents(query: str) -> str:
            """Search the indexed document library (which may contain multiple
            different documents/books) for passages relevant to the given query. Use
            this to look up specific details, quotes, or context. Call it again with a
            reformulated or narrower query if the first result does not fully answer
            the question."""
            results = search(es, embed_model, reranker, query, final_top_n=5, book_id=self._book_id)

            new_results = [
                c for c in results
                if c["hit"]["_source"]["chunk_id"] not in self._seen_chunk_ids
            ]
            # Only chunks that clear the relevance bar are marked "seen" and cited --
            # a chunk that's weak for THIS sub-query is left eligible to be picked up
            # again by a later, differently-worded sub-query where it might score well.
            relevant_results = [c for c in new_results if c["rerank_score"] >= MIN_RELEVANCE_SCORE]
            for c in relevant_results:
                self._seen_chunk_ids.add(c["hit"]["_source"]["chunk_id"])

            self.last_calls.append({
                "query": query,
                "num_hits": len(new_results),
                "num_relevant": len(relevant_results),
            })
            self.last_sources.extend(relevant_results)

            if not new_results:
                return "(Không có đoạn mới nào -- các đoạn khớp với truy vấn này đã được tìm thấy ở lần tìm kiếm trước.)"
            if not relevant_results:
                top_score = max(c["rerank_score"] for c in new_results)
                return (
                    "(Không tìm thấy đoạn nào đủ liên quan trong kho tài liệu cho truy vấn này -- "
                    f"điểm liên quan cao nhất chỉ đạt {top_score:.2f}/1.0, dưới ngưỡng tin cậy "
                    f"({MIN_RELEVANCE_SCORE}). Hãy báo cho người dùng là không tìm thấy thông tin này "
                    "trong kho tài liệu, đừng suy đoán hay dùng kiến thức ngoài.)"
                )
            return build_context(relevant_results)

        llm = ChatGoogleGenerativeAI(model=model, temperature=0.2, google_api_key=api_key)
        llm_with_tools = llm.bind_tools([search_documents])

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

    def _reset(self, book_id):
        self.last_sources = []
        self.last_calls = []
        self._seen_chunk_ids = set()
        self._book_id = book_id

    def _initial_state(self, query_text: str, book_title=None):
        messages = [SystemMessage(content=SYSTEM_PROMPT)]
        if book_title:
            messages.append(SystemMessage(content=(
                f'Phạm vi tìm kiếm của câu hỏi này đã được giới hạn trong tài liệu "{book_title}". '
                "Mọi kết quả `search_documents` trả về đều chỉ thuộc tài liệu này."
            )))
        messages.append(HumanMessage(content=query_text))
        return {"messages": messages}

    def ask(self, query_text: str, book_id=None, book_title=None):
        """Non-streaming: run the full agent loop, return (answer, sources, tool_calls).
        book_id=None searches the whole library; otherwise only that document."""
        self._reset(book_id)

        result = self.graph.invoke(self._initial_state(query_text, book_title))
        final_answer = extract_text(result["messages"][-1].content)
        return final_answer, list(self.last_sources), list(self.last_calls)

    def stream(self, query_text: str, book_id=None, book_title=None):
        """Streaming: yields one {"node": "agent"|"tools", "messages": [...]} event per
        graph step, in order, as the agent decomposes/searches/reasons -- for live
        display of "how the LLM broke down the query" instead of only the final answer.
        After the generator is exhausted, self.last_sources/self.last_calls hold the
        full accumulated results for this call, same as after `.ask()`.
        book_id=None searches the whole library; otherwise only that document.
        """
        self._reset(book_id)

        for step in self.graph.stream(self._initial_state(query_text, book_title), stream_mode="updates"):
            for node_name, node_output in step.items():
                yield {"node": node_name, "messages": node_output["messages"]}


if __name__ == "__main__":
    es = load_es_client()
    embed_model = load_embed_model()
    reranker = load_reranker()
    agent = GatsbyAgent(es, embed_model, reranker)

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
            print(f"  {i}. query={call['query']!r} -> {call['num_hits']} hits, {call['num_relevant']} relevant")
        print(f"\nTotal source hits across all calls ({len(sources)}):")
        for c in sources:
            src = c["hit"]["_source"]
            print(f"  - {src['chunk_id']} (Chapter {src['chapter_title']}, rerank_score={c['rerank_score']:.4f})")
