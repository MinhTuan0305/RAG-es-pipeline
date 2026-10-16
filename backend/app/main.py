"""
Document QA API.

    cd backend
    uvicorn app.main:app --port 8000

Run ONE worker: every worker process loads its own copy of the models. Requests are
served concurrently on threads; app.core.rag_pipeline.MODEL_LOCK keeps the GPU to one
model run at a time.
"""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.api import chat, conversations, documents, health, jobs
from app.core.errors import ApiError, explain_error
from app.deps import Resources
from app.services.ingest_service import ParseStore
from app.services.jobs import JobManager

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("app").setLevel(logging.INFO)
# The Gemini SDK logs each retry (429/5xx/timeouts) at INFO -- worth seeing.
logging.getLogger("google_genai._api_client").setLevel(logging.INFO)
logger = logging.getLogger("app.main")


def load_resources() -> Resources:
    """Everything heavy, once per process: ES client, embedding model, reranker and
    the agent. (Docling is created lazily on the first upload: its import is slow.)"""
    from app.core.chat_store import init_db
    from app.core.langgraph_agent import DocumentQAAgent
    from app.core.rag_pipeline import load_embed_model, load_es_client, load_reranker

    init_db()
    es = load_es_client()
    logger.info("Loading embedding model and reranker...")
    embed_model = load_embed_model()
    reranker = load_reranker()
    agent = DocumentQAAgent(es, embed_model, reranker)
    logger.info("Ready.")
    return Resources(
        es=es, embed_model=embed_model, reranker=reranker, agent=agent,
        jobs=JobManager(), parses=ParseStore(),
    )


def create_app(resources_factory=load_resources) -> FastAPI:
    """resources_factory is swappable so tests can run the API on fakes."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.resources = await asyncio.to_thread(resources_factory)  # model loading blocks
        yield
        from app.core.tracing import flush

        flush()  # send the last Langfuse traces before exiting
        from app.core.chat_store import dispose

        dispose()  # close the database connection pool

    app = FastAPI(title="Document QA API", version="1.0", lifespan=lifespan)

    @app.exception_handler(ApiError)
    async def api_error(_request, exc: ApiError):
        return JSONResponse(status_code=exc.status_code, content={"message": exc.message, "hint": exc.hint})

    @app.exception_handler(Exception)
    async def unexpected_error(request, exc: Exception):
        logger.error("Unhandled error on %s %s", request.method, request.url.path, exc_info=exc)
        err = explain_error(exc)
        return JSONResponse(status_code=500, content={"message": err.message, "hint": err.hint})

    for router in (health.router, conversations.router, chat.router, documents.router, jobs.router):
        app.include_router(router, prefix="/api")
    return app


app = create_app()
