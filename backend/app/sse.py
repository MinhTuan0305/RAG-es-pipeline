"""
Server-Sent Events plumbing.

The agent and the models are synchronous and block (GPU, network), so a stream is
produced by a plain generator running in its own thread. Its items are handed to the
event loop through an asyncio.Queue, which the SSE response reads -- the server
itself never blocks. When the client disconnects, the response stops reading and the
`cancel` event tells the producer to stop at its next item.
"""

import asyncio
import logging
import threading
from collections.abc import AsyncIterator, Callable, Iterator

from pydantic import BaseModel
from sse_starlette.sse import ServerSentEvent

logger = logging.getLogger(__name__)

PING_SECONDS = 15

_END = object()


class _Failure:
    def __init__(self, exc: BaseException):
        self.exc = exc


def to_sse(event: BaseModel, event_id: int | None = None) -> ServerSentEvent:
    """`event:` = the model's type, `data:` = the model as JSON (type included)."""
    return ServerSentEvent(
        data=event.model_dump_json(),
        event=getattr(event, "type", None),
        id=str(event_id) if event_id is not None else None,
    )


async def iterate_in_thread(
    make_iter: Callable[[threading.Event], Iterator], cancel: threading.Event | None = None,
) -> AsyncIterator:
    """Run make_iter(cancel) in a worker thread and yield its items here, in order.
    An exception raised by the producer is re-raised here. Leaving this generator
    early (client gone) sets `cancel`; the producer is expected to check it and wind
    down on its own (e.g. the chat service still saves the partial answer)."""
    cancel = cancel or threading.Event()
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def put(item):
        try:
            loop.call_soon_threadsafe(queue.put_nowait, item)
        except RuntimeError:  # event loop already closed: nobody is listening any more
            pass

    def worker():
        gen = make_iter(cancel)
        try:
            for item in gen:
                put(item)
        except BaseException as e:  # noqa: BLE001 -- handed over to the async side
            put(_Failure(e))
        finally:
            gen.close()
            put(_END)

    threading.Thread(target=worker, name="sse-producer", daemon=True).start()
    try:
        while True:
            item = await queue.get()
            if item is _END:
                return
            if isinstance(item, _Failure):
                raise item.exc
            yield item
    finally:
        cancel.set()
