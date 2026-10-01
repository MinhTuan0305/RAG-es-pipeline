"""
Persistent chat history in a local SQLite file.

Stores exactly what the chat UI keeps per message (role, content, plus the
assistant's tool_calls / sources / scope / error) so an old conversation can be
reopened with its reasoning steps and citations intact.
"""

import json
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent / "chat_history.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id  TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role             TEXT NOT NULL,
    content          TEXT NOT NULL,
    extras           TEXT,
    created_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect():
    # A fresh connection per call: cheap for SQLite, and avoids sharing one
    # connection across the threads Streamlit runs scripts on.
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")  # needed for ON DELETE CASCADE
    return conn


def init_db():
    with closing(_connect()) as conn, conn:
        conn.executescript(_SCHEMA)


def create_conversation(title: str) -> str:
    conversation_id = uuid.uuid4().hex
    now = _now()
    with closing(_connect()) as conn, conn:
        conn.execute(
            "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (conversation_id, title, now, now),
        )
    return conversation_id


def get_conversation(conversation_id: str):
    with closing(_connect()) as conn:
        row = conn.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
    return dict(row) if row else None


def list_conversations(limit: int = 50):
    with closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT id, title, updated_at FROM conversations ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def delete_conversation(conversation_id: str):
    with closing(_connect()) as conn, conn:
        conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))


def add_message(conversation_id: str, message: dict):
    """message: {"role", "content", ...any extra UI fields}. Extras are stored as JSON."""
    extras = {k: v for k, v in message.items() if k not in ("role", "content")}
    now = _now()
    with closing(_connect()) as conn, conn:
        conn.execute(
            "INSERT INTO messages (conversation_id, role, content, extras, created_at) VALUES (?, ?, ?, ?, ?)",
            (
                conversation_id,
                message["role"],
                message["content"],
                # default=float: rerank scores may come back as numpy floats.
                json.dumps(extras, ensure_ascii=False, default=float) if extras else None,
                now,
            ),
        )
        conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conversation_id))


def load_messages(conversation_id: str):
    with closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT role, content, extras FROM messages WHERE conversation_id = ? ORDER BY id",
            (conversation_id,),
        ).fetchall()
    messages = []
    for r in rows:
        msg = {"role": r["role"], "content": r["content"]}
        if r["extras"]:
            msg.update(json.loads(r["extras"]))
        messages.append(msg)
    return messages
