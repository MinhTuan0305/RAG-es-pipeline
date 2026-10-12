"""
Test fixtures: the real API (routes, services, SSE, job manager, Docling parsing,
chunking) running on fakes for everything heavy or external -- Elasticsearch, the
embedding model, the reranker and the LLM agent.
"""

import json
import time
from datetime import datetime

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from app.core import chat_store
from app.deps import Resources
from app.main import create_app
from app.services import ingest_service
from app.services.ingest_service import ParseStore
from app.services.jobs import JobManager


class FakeES:
    """Just enough of Elasticsearch for document listing/lookup/deletion."""

    def __init__(self):
        self.docs = []
        self.up = True

    def ping(self):
        return self.up

    @staticmethod
    def _match(doc, query):
        if not query or "match_all" in query:
            return True
        (field, value), = query["term"].items()
        return doc.get(field) == value

    def search(self, index, size=0, query=None, aggs=None, **kwargs):
        groups = {}
        for d in self.docs:
            if self._match(d, query):
                groups.setdefault(d["book_id"], []).append(d)
        buckets = []
        for book_id, ds in groups.items():
            ups = [d["uploaded_at"] for d in ds if d.get("uploaded_at")]
            files = [d["source_file"] for d in ds if d.get("source_file")]
            buckets.append({
                "key": book_id,
                "doc_count": len(ds),
                "title": {"buckets": [{"key": ds[0]["book_title"]}]},
                "file": {"buckets": [{"key": files[0]}] if files else []},
                "uploaded_at": (
                    {"value": 1.0, "value_as_string": datetime.fromisoformat(max(ups)).strftime("%Y-%m-%dT%H:%M:%S.000Z")}
                    if ups else {"value": None}
                ),
                "tables": {"doc_count": sum(d.get("type") == "table" for d in ds)},
            })
        return {"hits": {"hits": []}, "aggregations": {"docs": {"buckets": buckets}}}

    def delete_by_query(self, index, query, refresh=None, conflicts=None):
        before = len(self.docs)
        self.docs = [d for d in self.docs if not self._match(d, query)]
        return {"deleted": before - len(self.docs)}


class FakeAgent:
    """Mimics DocumentQAAgent.stream(): a preamble that turns into a search, the
    search results, then the streamed final answer. `fail` raises mid-run;
    `token_delay` slows the answer down (for stop/disconnect tests)."""

    def __init__(self):
        self.fail = None
        self.token_delay = 0.0
        self.answer_tokens = ["Gatsby ", "là ", "một ", "triệu phú ", "bí ẩn."]
        self.histories = []

    def stream(self, query_text, ctx, history=None):
        self.histories.append(history)
        yield {"type": "token", "text": "Để mình tìm..."}
        yield {"type": "update", "node": "agent", "messages": [
            AIMessage(content="", tool_calls=[{"name": "search_documents", "args": {"query": "Gatsby là ai"}, "id": "1"}]),
        ]}
        if self.fail:
            raise self.fail
        ctx.calls.append({"query": "Gatsby là ai", "num_hits": 1})
        ctx.sources.append({
            "hit": {"_source": {"type": "paragraph", "book_title": "The Great Gatsby", "chapter_title": "I",
                                "section_title": None, "chunk_id": "gatsby-ch01-c001", "text": "In my younger..."}},
            "rerank_score": 0.93, "rrf_score": 0.03,
        })
        yield {"type": "update", "node": "tools", "messages": []}
        for t in self.answer_tokens:
            if self.token_delay:
                time.sleep(self.token_delay)
            yield {"type": "token", "text": t}
        yield {"type": "update", "node": "agent", "messages": [AIMessage(content="".join(self.answer_tokens))]}


def fake_embed(model, chunks, on_progress=None, on_retry=None, batch_size=8, **kwargs):
    if on_retry and len(chunks) > 2:
        on_retry(4, None)  # pretend the first batch ran out of memory
    for start in range(0, len(chunks), batch_size):
        if on_progress:
            on_progress(min(start + batch_size, len(chunks)), len(chunks))
    return chunks


@pytest.fixture
def fakes(tmp_path, monkeypatch):
    monkeypatch.setattr(chat_store, "DB_PATH", tmp_path / "chat.db")
    es = FakeES()

    def fake_index(es_, chunks):
        es_.docs.extend(dict(c) for c in chunks)
        return len(chunks), []

    monkeypatch.setattr(ingest_service, "embed_chunks_batch", fake_embed)
    monkeypatch.setattr(ingest_service, "index_chunks", fake_index)
    monkeypatch.setattr(ingest_service, "ensure_index_exists", lambda es_: None)

    agent = FakeAgent()

    def factory():
        chat_store.init_db()
        return Resources(es=es, embed_model=object(), reranker=object(), agent=agent,
                         jobs=JobManager(), parses=ParseStore())

    return {"es": es, "agent": agent, "factory": factory}


@pytest.fixture
def live_server(fakes):
    """The app on a real uvicorn server (random port), for behaviour TestClient can't
    show -- e.g. a client hanging up mid-stream."""
    import socket
    import threading

    import uvicorn

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(
        create_app(resources_factory=fakes["factory"]), host="127.0.0.1", port=port, log_level="warning",
    ))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture
def client(fakes):
    with TestClient(create_app(resources_factory=fakes["factory"])) as c:
        yield c


def read_sse(response, limit=None):
    """Parse an SSE response into [(event, data_dict, id)], optionally stopping early."""
    events, event, data, event_id = [], None, [], None
    for line in response.iter_lines():
        if line == "":
            if data:
                events.append((event, json.loads("\n".join(data)), event_id))
                if limit and len(events) >= limit:
                    break
            event, data, event_id = None, [], None
        elif line.startswith(":"):
            continue  # ping
        elif line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].strip())
        elif line.startswith("id:"):
            event_id = line[3:].strip()
    return events
