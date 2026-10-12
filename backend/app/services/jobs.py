"""
Background jobs (document embedding/indexing) with a replayable event log.

A job runs in its own thread and reports through emit(event). Every event gets an
increasing id (1, 2, ...) and is kept, so a client can (re)connect at any time --
after a page reload, say -- and continue from the last id it saw (SSE Last-Event-ID).
Jobs live in memory: restarting the backend forgets them (fine for one worker).
"""

import logging
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from app.core.errors import AppError, explain_error
from app.schemas import ErrorBody, ErrorEvent, JobDone, JobInfo

logger = logging.getLogger(__name__)

FINISHED_JOB_TTL = timedelta(hours=1)


@dataclass
class Job:
    id: str
    kind: str
    created_at: datetime
    status: str = "running"  # running | done | error
    events: list = field(default_factory=list)  # event models; id of events[i] is i + 1
    result: JobDone | None = None
    error: ErrorBody | None = None
    finished_at: datetime | None = None
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def events_after(self, last_id: int) -> list[tuple[int, object]]:
        with self.lock:
            return [(i + 1, e) for i, e in enumerate(self.events) if i + 1 > last_id]

    @property
    def finished(self) -> bool:
        return self.status != "running"

    def info(self) -> JobInfo:
        with self.lock:
            return JobInfo(
                job_id=self.id, kind=self.kind, status=self.status,
                created_at=self.created_at.isoformat(timespec="seconds"),
                events=len(self.events), result=self.result, error=self.error,
            )


class JobManager:
    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def start(self, kind: str, work: Callable[[Callable], JobDone]) -> Job:
        """Run work(emit) in a thread. work returns the final JobDone (emitted as the
        last event); any exception becomes the job's final ErrorEvent instead."""
        self._cleanup()
        job = Job(id=uuid.uuid4().hex, kind=kind, created_at=datetime.now(timezone.utc))
        with self._lock:
            self._jobs[job.id] = job

        def emit(event):
            with job.lock:
                job.events.append(event)

        def run():
            try:
                result = work(emit)
            except Exception as e:  # noqa: BLE001 -- reported to the client as the final event
                err = e if isinstance(e, AppError) else explain_error(e)
                logger.error("Job %s (%s) failed", job.id, kind, exc_info=e)
                with job.lock:
                    job.error = ErrorBody(message=err.message, hint=err.hint)
                    job.events.append(ErrorEvent(message=err.message, hint=err.hint))
                    job.status = "error"
            else:
                with job.lock:
                    job.result = result
                    job.events.append(result)
                    job.status = "done"
            finally:
                job.finished_at = datetime.now(timezone.utc)

        threading.Thread(target=run, name=f"job-{kind}-{job.id[:8]}", daemon=True).start()
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def _cleanup(self):
        cutoff = datetime.now(timezone.utc) - FINISHED_JOB_TTL
        with self._lock:
            for job_id in [j.id for j in self._jobs.values() if j.finished_at and j.finished_at < cutoff]:
                del self._jobs[job_id]
