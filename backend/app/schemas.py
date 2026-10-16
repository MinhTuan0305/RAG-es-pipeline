"""
Request/response bodies and the SSE event contract.

Every SSE event is one of the models below; its `type` is also the SSE `event:` name,
and the model (type included) is the JSON in `data:`. The frontend generates its
TypeScript types from the OpenAPI schema, which includes these models through
GET /api/schema/events.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------- shared

class ErrorBody(BaseModel):
    """Error response of every endpoint, and the `error` SSE event's payload."""

    message: str
    hint: str | None = None


class SourceRecord(BaseModel):
    type: str = "paragraph"
    book_title: str
    location: str
    chunk_id: str
    rerank_score: float
    text: str
    citation: int | None = None  # the [n] the answer cites it by (None: saved before citations)
    cited: bool = False  # whether the final answer actually cites it


class DocumentSummary(BaseModel):
    book_id: str
    book_title: str
    num_chunks: int
    num_tables: int = 0
    source_file: str | None = None
    uploaded_at: str | None = None


# ---------------------------------------------------------------- conversations

class ConversationSummary(BaseModel):
    id: str
    title: str
    updated_at: str


class ToolCallRecord(BaseModel):
    query: str
    num_hits: int


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    tool_calls: list[ToolCallRecord] = []
    sources: list[SourceRecord] = []
    scope: str | None = None
    error: bool = False
    stopped: bool = False


class ConversationDetail(BaseModel):
    id: str
    title: str
    messages: list[Message]


# ---------------------------------------------------------------- chat

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    conversation_id: str | None = None  # None = start a new conversation
    book_id: str | None = None  # None = search the whole library


class RunStarted(BaseModel):
    type: Literal["run_started"] = "run_started"
    conversation_id: str
    title: str | None = None  # set when this question created the conversation


class ToolCall(BaseModel):
    type: Literal["tool_call"] = "tool_call"
    step: int
    query: str


class ToolResult(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    query: str
    num_hits: int


class AnswerReset(BaseModel):
    """The text streamed so far was a preamble before a search, not the answer:
    clear it."""

    type: Literal["answer_reset"] = "answer_reset"


class Token(BaseModel):
    type: Literal["token"] = "token"
    text: str


class Sources(BaseModel):
    """Passages found by a search round, sent right away (citations in the streaming
    answer can be shown immediately). Each event ADDS items; numbers keep increasing."""

    type: Literal["sources"] = "sources"
    items: list[SourceRecord]


class ChatDone(BaseModel):
    """The final answer, citations cleaned up (normalized, made-up numbers removed):
    it replaces the streamed text. `cited` = the citation numbers it uses."""

    type: Literal["done"] = "done"
    answer: str
    stopped: bool = False
    cited: list[int] = []


class ErrorEvent(ErrorBody):
    type: Literal["error"] = "error"


ChatEvent = Annotated[
    RunStarted | ToolCall | ToolResult | AnswerReset | Token | Sources | ChatDone | ErrorEvent,
    Field(discriminator="type"),
]


# ---------------------------------------------------------------- documents

class TableInfo(BaseModel):
    table_id: str
    caption: str | None = None
    path: list[str]  # headings from the document root down to the table
    n_rows: int
    n_cols: int
    markdown: str  # exactly what gets embedded (caption line included)


class TreeNode(BaseModel):
    title: str
    depth: int
    n_paragraphs: int
    table_ids: list[str] = []
    children: list["TreeNode"] = []


class TreeStats(BaseModel):
    headings: int
    paragraphs: int
    tables: int


class ParsedMetadata(BaseModel):
    book_id: str
    book_title: str
    author: str = ""


class ParseReview(BaseModel):
    metadata: ParsedMetadata
    stats: TreeStats
    tree: TreeNode
    tables: list[TableInfo]


class ParseResult(BaseModel):
    """status "ready": parse_id + review. status "duplicate": `existing` is the
    already-indexed copy; match "file" = same bytes (not parsed: re-send with
    replace_book_id to go on), match "content" = same text (parsed: review included,
    index with replace_book_id to replace the old copy)."""

    status: Literal["ready", "duplicate"]
    match: Literal["file", "content"] | None = None
    existing: DocumentSummary | None = None
    parse_id: str | None = None
    review: ParseReview | None = None


class IndexRequest(BaseModel):
    parse_id: str
    title: str = Field(min_length=1, max_length=300)
    author: str = ""
    excluded_table_ids: list[str] = []
    replace_book_id: str | None = None


class DeleteResult(BaseModel):
    deleted_chunks: int


# ---------------------------------------------------------------- jobs

class JobStarted(BaseModel):
    job_id: str


class Progress(BaseModel):
    type: Literal["progress"] = "progress"
    stage: Literal["prepare", "chunk", "embed", "index", "replace", "done"]
    value: float  # 0..1 for the whole job
    text: str
    done: int | None = None
    total: int | None = None


class Retry(BaseModel):
    type: Literal["retry"] = "retry"
    batch_size: int
    text: str


class JobDone(BaseModel):
    type: Literal["done"] = "done"
    book_id: str
    book_title: str
    chunks: int
    table_chunks: int
    replaced_chunks: int | None = None


JobEvent = Annotated[Progress | Retry | JobDone | ErrorEvent, Field(discriminator="type")]


class JobInfo(BaseModel):
    job_id: str
    kind: str
    status: Literal["running", "done", "error"]
    created_at: str
    events: int  # how many events so far (the SSE ids are 1..events)
    result: JobDone | None = None
    error: ErrorBody | None = None


class EventSchemas(BaseModel):
    """Not an endpoint payload: lists the SSE event models so they are part of the
    OpenAPI schema (for frontend type generation)."""

    chat: ChatEvent
    job: JobEvent


# ---------------------------------------------------------------- health

class Health(BaseModel):
    status: Literal["ok", "degraded"]
    elasticsearch: bool
    database: bool
    models_loaded: bool
    llm_key_configured: bool
    tracing_enabled: bool
