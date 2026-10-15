import { api, ApiError } from "./client";
import type { JobEvent } from "./types";

/**
 * Follow a background job's events (GET /api/jobs/{id}/events). EventSource
 * reconnects by itself after a network blip, resuming after the last event id, so
 * nothing is missed or repeated. Stops after the final event (done/error), or when the
 * job turns out not to exist (expired, or the backend was restarted). Returns a
 * function that stops following.
 */
export function watchJob(
  jobId: string,
  onEvent: (event: JobEvent) => void,
  onLost: (error: ApiError) => void,
): () => void {
  const source = new EventSource(`/api/jobs/${encodeURIComponent(jobId)}/events`);
  let finished = false;

  const close = () => {
    finished = true;
    source.close();
  };

  const handle = (raw: MessageEvent) => {
    const event = JSON.parse(raw.data) as JobEvent;
    onEvent(event);
    if (event.type === "done" || event.type === "error") close();
  };
  for (const type of ["progress", "retry", "done", "error"]) {
    source.addEventListener(type, handle as EventListener);
  }

  source.onerror = () => {
    if (finished) return;
    // Either a blip (EventSource retries on its own) or the job is gone (404, which
    // EventSource can't tell apart): ask the job endpoint which one it is.
    api.job(jobId).catch((error: ApiError) => {
      if (error.status === 404 && !finished) {
        close();
        onLost(error);
      }
    });
  };

  return close;
}
