import { fetchEventSource } from "@microsoft/fetch-event-source";

import { ApiError, errorFromResponse, toApiError } from "./client";
import type { ChatEvent, ChatRequest } from "./types";

class StopStreaming extends Error {}

/**
 * POST /api/chat and hand every SSE event to onEvent until the run ends.
 * EventSource can't POST a body, hence fetch-event-source. Resolves when the stream
 * ends (done or error event, or `signal` aborted by the Stop button); rejects with an
 * ApiError when the request itself fails (validation, not found, backend down).
 */
export async function streamChat(
  body: ChatRequest,
  onEvent: (event: ChatEvent) => void,
  signal: AbortSignal,
): Promise<void> {
  try {
    await fetchEventSource("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal,
      // Keep streaming while the tab is in the background (the default would close
      // and re-POST the question when the tab becomes visible again).
      openWhenHidden: true,
      async onopen(response) {
        if (!response.ok || !response.headers.get("content-type")?.startsWith("text/event-stream")) {
          throw await errorFromResponse(response);
        }
      },
      onmessage(message) {
        if (!message.data) return; // keep-alive ping
        onEvent(JSON.parse(message.data) as ChatEvent);
      },
      onclose() {
        throw new StopStreaming(); // the run ended: don't reconnect
      },
      onerror(error) {
        throw error; // never retry: retrying would ask the question again
      },
    });
  } catch (error) {
    if (error instanceof StopStreaming || signal.aborted) return;
    throw error instanceof ApiError ? error : toApiError(error);
  }
}
