import asyncio

from fastapi import APIRouter, Depends, Header, Query
from sse_starlette.sse import EventSourceResponse

from app.core.errors import ApiError
from app.deps import Resources, get_resources
from app.schemas import ErrorBody, JobInfo
from app.sse import PING_SECONDS, to_sse

router = APIRouter(tags=["jobs"])

POLL_SECONDS = 0.25


def _job(res: Resources, job_id: str):
    job = res.jobs.get(job_id)
    if job is None:
        raise ApiError(404, "Không tìm thấy job (đã hết hạn hoặc backend đã khởi động lại).")
    return job


@router.get("/jobs/{job_id}", response_model=JobInfo, responses={404: {"model": ErrorBody}})
def job_info(job_id: str, res: Resources = Depends(get_resources)):
    return _job(res, job_id).info()


@router.get(
    "/jobs/{job_id}/events",
    response_class=EventSourceResponse,
    responses={
        200: {"description": "SSE stream of JobEvent (see GET /api/schema/events)", "content": {"text/event-stream": {}}},
        404: {"model": ErrorBody},
    },
)
def job_events(
    job_id: str,
    after: int = Query(0, ge=0, description="Only events with an id above this"),
    last_event_id: str | None = Header(None),
    res: Resources = Depends(get_resources),
):
    """Every event of the job (progress, retry, then done or error), each with an id.
    Reconnecting -- EventSource does it automatically with Last-Event-ID -- resumes
    after the last event received, so nothing is missed or repeated. The stream ends
    after the final event."""
    job = _job(res, job_id)
    start = int(last_event_id) if last_event_id and last_event_id.isdigit() else after

    async def events():
        last = start
        while True:
            for event_id, event in job.events_after(last):
                yield to_sse(event, event_id)
                last = event_id
            if job.finished and not job.events_after(last):
                return
            await asyncio.sleep(POLL_SECONDS)

    return EventSourceResponse(events(), ping=PING_SECONDS)
