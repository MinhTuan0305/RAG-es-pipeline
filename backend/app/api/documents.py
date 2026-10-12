from fastapi import APIRouter, Depends, File, Form, UploadFile

from app.core.errors import ApiError
from app.core.rag_pipeline import delete_document, get_document, list_documents
from app.deps import Resources, get_resources
from app.schemas import DeleteResult, DocumentSummary, ErrorBody, IndexRequest, JobStarted, ParseResult
from app.services.ingest_service import parse_upload, start_index_job

router = APIRouter(tags=["documents"])

MAX_UPLOAD_MB = 50


@router.get("/documents", response_model=list[DocumentSummary])
def documents(res: Resources = Depends(get_resources)):
    return list_documents(res.es)


@router.delete("/documents/{book_id}", response_model=DeleteResult, responses={404: {"model": ErrorBody}})
def remove_document(book_id: str, res: Resources = Depends(get_resources)):
    if get_document(res.es, book_id) is None:
        raise ApiError(404, "Không tìm thấy tài liệu (có thể đã bị xoá).")
    return {"deleted_chunks": delete_document(res.es, book_id)}


@router.post(
    "/documents/parse",
    response_model=ParseResult,
    responses={400: {"model": ErrorBody}, 404: {"model": ErrorBody}, 413: {"model": ErrorBody}, 422: {"model": ErrorBody}},
)
def parse(
    file: UploadFile = File(...),
    replace_book_id: str | None = Form(None),
    res: Resources = Depends(get_resources),
):
    """Step 1 of adding a document: duplicate checks + Docling parse. Returns the
    review data (heading tree, tables) under a parse_id, or a duplicate notice.
    Pass replace_book_id to continue past a duplicate by replacing that document."""
    data = file.file.read(MAX_UPLOAD_MB * 1024 * 1024 + 1)
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise ApiError(413, f"File quá lớn (tối đa {MAX_UPLOAD_MB} MB).")
    return parse_upload(res.es, res.parses, file.filename or "document", data, replace_book_id)


@router.post(
    "/documents/index",
    response_model=JobStarted,
    status_code=202,
    responses={404: {"model": ErrorBody}},
)
def index(req: IndexRequest, res: Resources = Depends(get_resources)):
    """Step 2: chunk + embed + index the reviewed document as a background job.
    Follow it with GET /api/jobs/{job_id}/events."""
    job = start_index_job(
        res.jobs, res.parses, res.es, res.embed_model,
        req.parse_id, req.title, req.author, req.excluded_table_ids, req.replace_book_id,
    )
    return {"job_id": job.id}
