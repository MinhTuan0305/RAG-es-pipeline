from fastapi import APIRouter, Query, Response

from app.core.chat_store import delete_conversation, get_conversation, list_conversations, load_messages
from app.core.errors import ApiError
from app.schemas import ConversationDetail, ConversationSummary, ErrorBody

router = APIRouter(tags=["conversations"])


@router.get("/conversations", response_model=list[ConversationSummary])
def conversations(limit: int = Query(50, ge=1, le=200)):
    return list_conversations(limit=limit)


@router.get("/conversations/{conversation_id}", response_model=ConversationDetail, responses={404: {"model": ErrorBody}})
def conversation(conversation_id: str):
    conv = get_conversation(conversation_id)
    if conv is None:
        raise ApiError(404, "Không tìm thấy cuộc trò chuyện.")
    return {"id": conv["id"], "title": conv["title"], "messages": load_messages(conversation_id)}


@router.delete("/conversations/{conversation_id}", status_code=204, responses={404: {"model": ErrorBody}})
def remove_conversation(conversation_id: str):
    if get_conversation(conversation_id) is None:
        raise ApiError(404, "Không tìm thấy cuộc trò chuyện.")
    delete_conversation(conversation_id)
    return Response(status_code=204)
