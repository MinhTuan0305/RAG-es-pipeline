"""Shared, process-wide resources (loaded once at startup) and their FastAPI dependency."""

from dataclasses import dataclass

from fastapi import Request

from app.services.ingest_service import ParseStore
from app.services.jobs import JobManager


@dataclass
class Resources:
    es: object
    embed_model: object
    reranker: object
    agent: object
    jobs: JobManager
    parses: ParseStore


def get_resources(request: Request) -> Resources:
    return request.app.state.resources
