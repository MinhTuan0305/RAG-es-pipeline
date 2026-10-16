import os

from fastapi import APIRouter, Depends

from app.core.chat_store import database_ok
from app.core.tracing import tracing_enabled
from app.deps import Resources, get_resources
from app.schemas import EventSchemas, Health, JobDone, Token

router = APIRouter(tags=["system"])


@router.get("/health", response_model=Health)
def health(res: Resources = Depends(get_resources)):
    try:
        es_ok = bool(res.es.ping())
    except Exception:  # noqa: BLE001 -- unreachable ES is exactly what this reports
        es_ok = False
    models_loaded = res.embed_model is not None and res.reranker is not None and res.agent is not None
    db_ok = database_ok()
    return {
        "status": "ok" if es_ok and db_ok and models_loaded else "degraded",
        "elasticsearch": es_ok,
        "database": db_ok,
        "models_loaded": models_loaded,
        "llm_key_configured": bool(os.environ.get("GOOGLE_API_KEY")),
        "tracing_enabled": tracing_enabled(),
    }


@router.get("/schema/events", response_model=EventSchemas, summary="SSE event models (documentation only)")
def event_schemas():
    """Exists so the SSE event models appear in the OpenAPI schema, for frontend type
    generation. Returns an example; the real events come from /api/chat and
    /api/jobs/{id}/events."""
    return EventSchemas(
        chat=Token(text="..."),
        job=JobDone(book_id="example", book_title="Example", chunks=0, table_chunks=0),
    )
