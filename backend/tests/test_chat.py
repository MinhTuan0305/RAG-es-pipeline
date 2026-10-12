import time

import httpx

from app.core import chat_store
from app.core.errors import AppError
from tests.conftest import read_sse


def ask(client, **body):
    with client.stream("POST", "/api/chat", json=body) as r:
        assert r.status_code == 200, r.read()
        assert r.headers["content-type"].startswith("text/event-stream")
        return read_sse(r)


def test_chat_event_sequence_and_saved_history(client):
    events = ask(client, message="Gatsby là ai?")
    types = [e for e, _, _ in events]
    assert types == ["run_started", "token", "answer_reset", "tool_call", "tool_result", "sources",
                     "token", "token", "token", "token", "token", "done"]
    started = events[0][1]
    assert started["title"] == "Gatsby là ai?"
    assert events[3][1] == {"type": "tool_call", "step": 1, "query": "Gatsby là ai"}
    assert events[4][1]["num_hits"] == 2
    # sources arrive right after their search, numbered, before the answer streams
    sources = events[5][1]["items"]
    assert [(s["citation"], s["chunk_id"]) for s in sources] == [(1, "gatsby-ch01-c001"), (2, "gatsby-ch03-c010")]
    assert sources[0]["location"] == "Chapter I"
    done = events[-1][1]
    # the made-up [7] is removed; [1] is kept and reported as cited
    assert done == {"type": "done", "answer": "Gatsby là một triệu phú [1] bí ẩn.", "stopped": False, "cited": [1]}

    detail = client.get(f"/api/conversations/{started['conversation_id']}").json()
    assert [m["role"] for m in detail["messages"]] == ["user", "assistant"]
    assistant = detail["messages"][1]
    assert assistant["content"] == done["answer"]
    assert assistant["tool_calls"] == [{"query": "Gatsby là ai", "num_hits": 2}]
    assert [(s["citation"], s["cited"]) for s in assistant["sources"]] == [(1, True), (2, False)]


def test_follow_up_sends_history_and_keeps_conversation(client, fakes):
    first = ask(client, message="Gatsby là ai?")
    conv = first[0][1]["conversation_id"]
    second = ask(client, message="Còn Daisy thì sao?", conversation_id=conv)
    assert second[0][1] == {"type": "run_started", "conversation_id": conv, "title": None}
    assert fakes["agent"].histories[-1] == [
        {"role": "user", "content": "Gatsby là ai?"},
        {"role": "assistant", "content": "Gatsby là một triệu phú bí ẩn."},  # citations stripped
    ]
    assert len(client.get("/api/conversations").json()) == 1


def test_agent_failure_becomes_error_event_and_is_saved(client, fakes):
    fakes["agent"].fail = AppError("Không kết nối được Elasticsearch (http://localhost:9200).", "Kiểm tra Docker.")
    events = ask(client, message="Gatsby là ai?")
    assert events[-1][0] == "error"
    assert events[-1][1]["message"].startswith("Không kết nối được Elasticsearch")
    assert events[-1][1]["hint"] == "Kiểm tra Docker."
    conv = events[0][1]["conversation_id"]
    saved = client.get(f"/api/conversations/{conv}").json()["messages"]
    assert saved[-1]["error"] is True

    # a failed turn is left out of the next question's history (with its question)
    fakes["agent"].fail = None
    ask(client, message="Thử lại", conversation_id=conv)
    assert fakes["agent"].histories[-1] == []


def test_unexpected_exception_is_explained(client, fakes):
    fakes["agent"].fail = ValueError("boom")
    events = ask(client, message="x")
    assert events[-1][0] == "error"
    assert "ValueError" in events[-1][1]["message"]


def test_disconnect_saves_partial_answer_as_stopped(live_server, fakes):
    """Real server + real HTTP connection: hanging up mid-answer must stop the run
    and save what was streamed so far, flagged as stopped."""
    fakes["agent"].token_delay = 0.3
    fakes["agent"].answer_tokens = [f"t{i} " for i in range(30)]
    with httpx.Client(base_url=live_server, timeout=10) as http:
        with http.stream("POST", "/api/chat", json={"message": "dài"}) as r:
            events = read_sse(r, limit=7)  # run_started .. a couple of answer tokens, then hang up
    conv = events[0][1]["conversation_id"]
    for _ in range(50):
        msgs = chat_store.load_messages(conv)
        if len(msgs) == 2:
            break
        time.sleep(0.2)
    assert msgs[1].get("stopped") is True
    assert msgs[1]["content"].startswith("t0 ")
    assert len(msgs[1]["content"].split()) < 30


def test_validation_and_not_found(client):
    assert client.post("/api/chat", json={"message": ""}).status_code == 422
    r = client.post("/api/chat", json={"message": "hi", "conversation_id": "nope"})
    assert r.status_code == 404 and r.json()["message"] == "Không tìm thấy cuộc trò chuyện."
    r = client.post("/api/chat", json={"message": "hi", "book_id": "nope"})
    assert r.status_code == 404


def test_delete_conversation(client):
    conv = ask(client, message="Gatsby là ai?")[0][1]["conversation_id"]
    assert client.delete(f"/api/conversations/{conv}").status_code == 204
    assert client.get(f"/api/conversations/{conv}").status_code == 404
    assert client.delete(f"/api/conversations/{conv}").status_code == 404


def test_health(client, fakes):
    body = client.get("/api/health").json()
    assert body["elasticsearch"] is True and body["models_loaded"] is True and body["status"] == "ok"
    fakes["es"].up = False
    assert client.get("/api/health").json()["status"] == "degraded"
