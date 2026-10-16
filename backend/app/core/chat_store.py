"""
Persistent chat history in PostgreSQL (SQLAlchemy Core).

Stores exactly what the chat UI keeps per message (role, content, plus the
assistant's tool_calls / sources / scope / error / stopped as JSON "extras") so an old
conversation can be reopened with its reasoning steps and citations intact.

The database comes from DATABASE_URL. The schema is created and evolved by Alembic
migrations (backend/migrations), applied by init_db() at startup. Any SQLAlchemy URL
works; the tests use a temporary SQLite file.
"""

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import (
    JSON,
    BigInteger,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    delete,
    event,
    insert,
    select,
    update,
)
from sqlalchemy.dialects.postgresql import JSONB

from app.core.config import settings

BACKEND_DIR = Path(__file__).resolve().parents[2]

metadata = MetaData()

conversations = Table(
    "conversations",
    metadata,
    Column("id", String(32), primary_key=True),
    # Not used yet: reserved for per-user conversations once there is a login.
    Column("user_id", String(64), nullable=True),
    Column("title", Text, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    Index("ix_conversations_updated_at", "updated_at"),
    Index("ix_conversations_user_id", "user_id"),
)

messages = Table(
    "messages",
    metadata,
    # BIGINT on PostgreSQL; plain INTEGER on SQLite, where only that is an autoincrementing rowid.
    Column("id", BigInteger().with_variant(Integer(), "sqlite"), primary_key=True, autoincrement=True),
    Column("conversation_id", String(32), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False),
    Column("role", String(16), nullable=False),
    Column("content", Text, nullable=False),
    Column("extras", JSON().with_variant(JSONB(), "postgresql"), nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    CheckConstraint("role IN ('user', 'assistant')", name="ck_messages_role"),
    Index("ix_messages_conversation_id_id", "conversation_id", "id"),
)


# ---------------------------------------------------------------- engine

_engine = None
_engine_url = None
_engine_lock = threading.Lock()


def _make_engine(url: str):
    if url.startswith("sqlite"):
        # The API serves requests from several threads.
        engine = create_engine(url, connect_args={"check_same_thread": False})

        @event.listens_for(engine, "connect")
        def _enable_foreign_keys(dbapi_connection, _record):
            dbapi_connection.execute("PRAGMA foreign_keys = ON")  # for ON DELETE CASCADE

        return engine
    # pool_pre_ping: a connection dropped while idle (DB restart) is replaced instead
    # of failing the next request. connect_timeout: fail fast when the DB is down.
    return create_engine(url, pool_pre_ping=True, connect_args={"connect_timeout": 5})


def get_engine():
    """The process-wide engine (connection pool), created on first use."""
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = _make_engine(_engine_url or settings.database_url)
        return _engine


def configure(url: str | None):
    """Use another database from now on (None = DATABASE_URL). For tests/scripts."""
    global _engine_url
    dispose()
    _engine_url = url


def dispose():
    """Close all pooled connections (on shutdown)."""
    global _engine
    with _engine_lock:
        if _engine is not None:
            _engine.dispose()
            _engine = None


def database_ok() -> bool:
    """Whether the database answers (for the health check)."""
    try:
        with get_engine().connect() as conn:
            conn.execute(select(1))
        return True
    except Exception:  # noqa: BLE001 -- any failure means "not reachable"
        return False


def init_db():
    """Bring the schema up to date (apply pending Alembic migrations)."""
    from alembic import command
    from alembic.config import Config

    cfg = Config(str(BACKEND_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND_DIR / "migrations"))
    with get_engine().begin() as connection:
        cfg.attributes["connection"] = connection  # migrations/env.py runs on this connection
        command.upgrade(cfg, "head")


# ---------------------------------------------------------------- helpers

def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    # SQLite gives naive datetimes back; everything is stored in UTC.
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat()


def _plain_json(extras: dict) -> dict:
    """Extras as plain JSON types (rerank scores may be numpy floats)."""
    return json.loads(json.dumps(extras, ensure_ascii=False, default=float))


# ---------------------------------------------------------------- the store

def create_conversation(title: str) -> str:
    conversation_id = uuid.uuid4().hex
    now = _now()
    with get_engine().begin() as conn:
        conn.execute(insert(conversations).values(id=conversation_id, title=title, created_at=now, updated_at=now))
    return conversation_id


def get_conversation(conversation_id: str):
    with get_engine().connect() as conn:
        row = conn.execute(
            select(conversations.c.id, conversations.c.title, conversations.c.created_at, conversations.c.updated_at)
            .where(conversations.c.id == conversation_id)
        ).first()
    if row is None:
        return None
    return {"id": row.id, "title": row.title, "created_at": _iso(row.created_at), "updated_at": _iso(row.updated_at)}


def list_conversations(limit: int = 50):
    with get_engine().connect() as conn:
        rows = conn.execute(
            select(conversations.c.id, conversations.c.title, conversations.c.updated_at)
            .order_by(conversations.c.updated_at.desc())
            .limit(limit)
        ).all()
    return [{"id": r.id, "title": r.title, "updated_at": _iso(r.updated_at)} for r in rows]


def delete_conversation(conversation_id: str):
    with get_engine().begin() as conn:
        conn.execute(delete(conversations).where(conversations.c.id == conversation_id))  # messages cascade


def add_message(conversation_id: str, message: dict):
    """message: {"role", "content", ...any extra UI fields}. Extras are stored as JSON."""
    extras = {k: v for k, v in message.items() if k not in ("role", "content")}
    now = _now()
    with get_engine().begin() as conn:  # one transaction: the message and the conversation's updated_at
        conn.execute(insert(messages).values(
            conversation_id=conversation_id,
            role=message["role"],
            content=message["content"],
            extras=_plain_json(extras) if extras else None,
            created_at=now,
        ))
        conn.execute(update(conversations).where(conversations.c.id == conversation_id).values(updated_at=now))


def load_messages(conversation_id: str):
    with get_engine().connect() as conn:
        rows = conn.execute(
            select(messages.c.role, messages.c.content, messages.c.extras)
            .where(messages.c.conversation_id == conversation_id)
            .order_by(messages.c.id)
        ).all()
    result = []
    for r in rows:
        msg = {"role": r.role, "content": r.content}
        if r.extras:
            msg.update(r.extras)
        result.append(msg)
    return result
