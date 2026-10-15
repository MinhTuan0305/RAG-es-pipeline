from tests.conftest import read_sse

DOC = """# Báo cáo doanh thu

## Tổng quan

Doanh thu năm 2024 tăng trưởng mạnh.

| Quý | Miền Bắc | Miền Nam |
|-----|----------|----------|
| Q1  | 120      | 95       |
| Q2  | 130      | 101      |

## Kết luận

Tiếp tục mở rộng.
"""


def parse(client, content=DOC, name="bao_cao.md", **form):
    r = client.post("/api/documents/parse", files={"file": (name, content.encode("utf-8"), "text/markdown")}, data=form)
    assert r.status_code == 200, r.text
    return r.json()


def run_index_job(client, parse_id, title, **extra):
    r = client.post("/api/documents/index", json={"parse_id": parse_id, "title": title, **extra})
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    with client.stream("GET", f"/api/jobs/{job_id}/events") as s:
        events = read_sse(s)
    return job_id, events


def test_parse_index_duplicate_replace_delete(client, fakes):
    # 1) parse -> review
    first = parse(client)
    assert first["status"] == "ready"
    review = first["review"]
    assert review["metadata"]["book_title"] == "Báo cáo doanh thu"
    assert review["stats"] == {"headings": 2, "paragraphs": 2, "tables": 1}
    assert [c["title"] for c in review["tree"]["children"]] == ["Tổng quan", "Kết luận"]
    assert "n_paragraphs" in review["tree"] and "blocks" not in review["tree"]  # slim tree, no texts
    table = review["tables"][0]
    assert table["path"] == ["Tổng quan"] and table["n_rows"] == 2 and table["n_cols"] == 3
    assert "| Q1 | 120 | 95 |" in table["markdown"]

    # 2) index as a background job, followed over SSE
    job_id, events = run_index_job(client, first["parse_id"], "Báo cáo 2024", author="Phòng KD")
    kinds = [e for e, _, _ in events]
    assert kinds[0] == "progress" and kinds[-1] == "done" and "retry" in kinds
    ids = [int(i) for _, _, i in events]
    assert ids == list(range(1, len(events) + 1))
    values = [d["value"] for e, d, _ in events if e == "progress"]
    assert values == sorted(values) and values[-1] == 1.0
    done = events[-1][1]
    assert done["book_title"] == "Báo cáo 2024" and done["table_chunks"] == 1 and done["replaced_chunks"] is None
    book_id = done["book_id"]
    docs = client.get("/api/documents").json()
    assert docs[0]["book_id"] == book_id and docs[0]["source_file"] == "bao_cao.md" and docs[0]["num_tables"] == 1
    assert all(d["file_hash"] and d["content_hash"] and d["author"] == "Phòng KD" for d in fakes["es"].docs)
    info = client.get(f"/api/jobs/{job_id}").json()
    assert info["status"] == "done" and info["result"]["book_id"] == book_id

    # 3) reconnect with Last-Event-ID: only what came after
    with client.stream("GET", f"/api/jobs/{job_id}/events", headers={"Last-Event-ID": str(len(events) - 2)}) as s:
        tail = read_sse(s)
    assert [int(i) for _, _, i in tail] == [len(events) - 1, len(events)]

    # 4) the parse was consumed
    r = client.post("/api/documents/index", json={"parse_id": first["parse_id"], "title": "x"})
    assert r.status_code == 404

    # 5) same file again (renamed): duplicate before parsing
    dup = parse(client, name="ban_sao.md")
    assert dup["status"] == "duplicate" and dup["match"] == "file"
    assert dup["existing"]["book_id"] == book_id and dup["parse_id"] is None

    # 6) ...replace it: parses, keeps the old title, swaps the documents
    again = parse(client, name="ban_sao.md", replace_book_id=book_id)
    assert again["status"] == "ready" and again["review"]["metadata"]["book_title"] == "Báo cáo 2024"
    _, events = run_index_job(client, again["parse_id"], "Báo cáo 2024", replace_book_id=book_id)
    done = events[-1][1]
    assert done["replaced_chunks"] > 0 and done["book_id"] != book_id
    assert [d["book_id"] for d in client.get("/api/documents").json()] == [done["book_id"]]
    new_id = done["book_id"]

    # 7) same text, different bytes (extra blank lines): content duplicate, parsed anyway
    resaved = parse(client, content=DOC.replace("\n\n", "\n\n\n"), name="resaved.md")
    assert resaved["status"] == "duplicate" and resaved["match"] == "content"
    assert resaved["existing"]["book_id"] == new_id and resaved["parse_id"] and resaved["review"]

    # 8) delete
    assert client.delete(f"/api/documents/{new_id}").json()["deleted_chunks"] > 0
    assert client.get("/api/documents").json() == []
    assert client.delete(f"/api/documents/{new_id}").status_code == 404


def test_excluded_tables_are_not_indexed(client, fakes):
    parsed = parse(client)
    table_id = parsed["review"]["tables"][0]["table_id"]
    _, events = run_index_job(client, parsed["parse_id"], "Không bảng", excluded_table_ids=[table_id])
    assert events[-1][1]["table_chunks"] == 0
    assert not any(d["type"] == "table" for d in fakes["es"].docs)


def test_parse_errors(client):
    r = client.post("/api/documents/parse", files={"file": ("x.exe", b"MZ", "application/octet-stream")})
    assert r.status_code == 400 and "Không hỗ trợ" in r.json()["message"]
    r = client.post("/api/documents/parse", files={"file": ("x.md", b"", "text/markdown")})
    assert r.status_code == 400
    r = client.post("/api/documents/parse", files={"file": ("a.md", b"x", "text/markdown")}, data={"replace_book_id": "nope"})
    assert r.status_code == 404
    assert client.get("/api/jobs/nope").status_code == 404
    assert client.get("/api/jobs/nope/events").status_code == 404


def test_failed_job_reports_error_event(client, fakes, monkeypatch):
    from app.core.errors import ModelRunError
    from app.services import ingest_service

    def oom(*a, **k):
        raise ModelRunError("Không đủ bộ nhớ để chạy model embedding (bge-m3).", "Đóng bớt ứng dụng.", is_memory=True)

    monkeypatch.setattr(ingest_service, "embed_chunks_batch", oom)
    parsed = parse(client)
    job_id, events = run_index_job(client, parsed["parse_id"], "x")
    assert events[-1][0] == "error" and events[-1][1]["message"].startswith("Không đủ bộ nhớ")
    assert client.get(f"/api/jobs/{job_id}").json()["status"] == "error"
    assert fakes["es"].docs == []


def test_openapi_includes_event_models(client):
    schemas = client.get("/openapi.json").json()["components"]["schemas"]
    for name in ("RunStarted", "Token", "ToolCall", "Sources", "ChatDone", "ErrorEvent", "Progress", "Retry", "JobDone"):
        assert name in schemas
