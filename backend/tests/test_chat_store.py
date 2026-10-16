"""
The chat store against SQLite and -- when reachable -- a real PostgreSQL.

PostgreSQL runs in a throwaway schema (created and dropped by the fixture) on the
server from TEST_DATABASE_URL, or DATABASE_URL if unset, so it never touches the real
tables. Without a reachable server those cases are skipped.
"""

import os
import time
import uuid

import numpy as np
import pytest
import sqlalchemy as sa
from sqlalchemy.engine import make_url

from app.core import chat_store
from app.core.config import settings


def _postgres_url():
    url = os.environ.get("TEST_DATABASE_URL", settings.database_url)
    if not url.startswith("postgresql"):
        return None
    engine = sa.create_engine(url, connect_args={"connect_timeout": 2})
    try:
        with engine.connect():
            return url
    except Exception:  # noqa: BLE001 -- no server: skip the postgres cases
        return None
    finally:
        engine.dispose()


@pytest.fixture(params=["sqlite", "postgresql"])
def store(request, tmp_path):
    if request.param == "sqlite":
        chat_store.configure(f"sqlite:///{(tmp_path / 'store.db').as_posix()}")
        chat_store.init_db()
        yield "sqlite"
        chat_store.configure(None)
        return

    base = _postgres_url()
    if base is None:
        pytest.skip("PostgreSQL not reachable (docker compose up -d postgres)")
    schema = f"test_{uuid.uuid4().hex[:12]}"
    admin = sa.create_engine(base, connect_args={"connect_timeout": 5})
    with admin.begin() as conn:
        conn.execute(sa.text(f'CREATE SCHEMA "{schema}"'))
    # Every connection of the store works inside the throwaway schema.
    url = make_url(base).update_query_dict({"options": f"-csearch_path={schema}"})
    chat_store.configure(url.render_as_string(hide_password=False))
    try:
        chat_store.init_db()
        yield "postgresql"
    finally:
        chat_store.configure(None)
        with admin.begin() as conn:
            conn.execute(sa.text(f'DROP SCHEMA "{schema}" CASCADE'))
        admin.dispose()


def test_conversation_lifecycle(store):
    cid = chat_store.create_conversation("Gatsby là ai?")
    conv = chat_store.get_conversation(cid)
    assert conv["id"] == cid and conv["title"] == "Gatsby là ai?"
    assert conv["created_at"].endswith("+00:00")  # timezone-aware ISO string, as the API returns

    assert chat_store.get_conversation("missing") is None
    chat_store.delete_conversation(cid)
    assert chat_store.get_conversation(cid) is None


def test_messages_round_trip_with_json_extras(store):
    cid = chat_store.create_conversation("q")
    chat_store.add_message(cid, {"role": "user", "content": "Gatsby là ai?"})
    chat_store.add_message(cid, {
        "role": "assistant",
        "content": "Một triệu phú [1].",
        "tool_calls": [{"query": "Gatsby", "num_hits": 2}],
        "sources": [{"chunk_id": "c1", "citation": 1, "cited": True, "rerank_score": np.float32(0.93), "text": "Đoạn"}],
        "scope": None,
        "stopped": True,
    })
    msgs = chat_store.load_messages(cid)
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[0] == {"role": "user", "content": "Gatsby là ai?"}  # no extras -> nothing added
    a = msgs[1]
    assert a["tool_calls"] == [{"query": "Gatsby", "num_hits": 2}]
    assert a["sources"][0]["citation"] == 1 and a["sources"][0]["cited"] is True
    assert abs(a["sources"][0]["rerank_score"] - 0.93) < 1e-6  # numpy float stored as plain JSON
    assert a["stopped"] is True and a["scope"] is None


def test_list_orders_by_last_activity_and_limits(store):
    first = chat_store.create_conversation("first")
    time.sleep(0.01)
    second = chat_store.create_conversation("second")
    assert [c["id"] for c in chat_store.list_conversations()] == [second, first]
    time.sleep(0.01)
    chat_store.add_message(first, {"role": "user", "content": "bump"})  # activity moves it to the top
    listed = chat_store.list_conversations()
    assert [c["id"] for c in listed] == [first, second]
    assert set(listed[0]) == {"id", "title", "updated_at"}
    assert len(chat_store.list_conversations(limit=1)) == 1


def test_deleting_a_conversation_deletes_its_messages(store):
    cid = chat_store.create_conversation("q")
    chat_store.add_message(cid, {"role": "user", "content": "x"})
    chat_store.delete_conversation(cid)
    with chat_store.get_engine().connect() as conn:
        left = conn.execute(sa.select(sa.func.count()).select_from(chat_store.messages)).scalar()
    assert left == 0


def test_role_is_constrained(store):
    cid = chat_store.create_conversation("q")
    with pytest.raises(sa.exc.IntegrityError):
        chat_store.add_message(cid, {"role": "system", "content": "x"})


def test_migrations_are_idempotent_and_db_is_healthy(store):
    chat_store.init_db()  # already at head: no-op
    assert chat_store.database_ok() is True


def test_unreachable_database_is_reported(tmp_path):
    chat_store.configure("postgresql+psycopg://nobody:nothing@127.0.0.1:1/none")
    try:
        assert chat_store.database_ok() is False
    finally:
        chat_store.configure(None)
