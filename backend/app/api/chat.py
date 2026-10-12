from fastapi import APIRouter, Depends
from sse_starlette.sse import EventSourceResponse

from app.core.chat_store import get_conversation
from app.core.errors import ApiError
from app.core.rag_pipeline import get_document
from app.deps import Resources, get_resources
from app.schemas import ChatRequest, ErrorBody
from app.services.chat_service import run_chat
from app.sse import PING_SECONDS, iterate_in_thread, to_sse

router = APIRouter(tags=["chat"])


@router.post(
    "/chat",
    response_class=EventSourceResponse,
    responses={
        200: {"description": "SSE stream of ChatEvent (see GET /api/schema/events)", "content": {"text/event-stream": {}}},
        404: {"model": ErrorBody},
    },
)
def chat(req: ChatRequest, res: Resources = Depends(get_resources)):
    """Ask one question. Streams: run_started, then tool_call / tool_result /
    answer_reset / token while the agent works, then sources and done -- or error.
    Closing the connection stops the run; the partial answer is saved as stopped."""
    # Validate before streaming, so these come back as plain HTTP errors.
    if req.conversation_id and not get_conversation(req.conversation_id):
        raise ApiError(404, "Không tìm thấy cuộc trò chuyện.")
    book_title = None
    if req.book_id:
        doc = get_document(res.es, req.book_id)
        if doc is None:
            raise ApiError(404, "Không tìm thấy tài liệu đã chọn làm phạm vi tìm kiếm (có thể đã bị xoá).")
        book_title = doc["book_title"]

    async def events():
        async for event in iterate_in_thread(
            lambda cancel: run_chat(res.agent, req.message, req.conversation_id, req.book_id, book_title, cancel)
        ):
            yield to_sse(event)

    return EventSourceResponse(events(), ping=PING_SECONDS)
