"""
User-facing errors.

Every failure that reaches the UI goes through explain_error(), which turns the raw
exception (an Elasticsearch connection error, a Gemini 429, a LangGraph recursion
limit, a CUDA out-of-memory, ...) into an AppError: a short message saying what went
wrong plus a hint saying what to do about it, both in Vietnamese. The original
exception stays chained (`raise explain_error(e) from e`), so logs keep the full
traceback.
"""

from config import settings


class AppError(Exception):
    """An error whose message is meant for the user, plus an optional hint."""

    def __init__(self, message: str, hint: str | None = None):
        super().__init__(message)
        self.message = message
        self.hint = hint

    def to_markdown(self) -> str:
        return f"**{self.message}**" + (f"\n\n{self.hint}" if self.hint else "")


class ModelRunError(AppError):
    """A local model (embedding / reranker) failed to run. `is_memory` says whether
    the cause was running out of RAM/VRAM, i.e. whether a smaller batch may help."""

    def __init__(self, message: str, hint: str | None = None, is_memory: bool = False):
        super().__init__(message, hint)
        self.is_memory = is_memory


MEMORY_HINT = (
    "Máy đang thiếu bộ nhớ (RAM/VRAM hoặc paging file). Hãy đóng bớt ứng dụng, khởi động lại app, "
    "hoặc tăng paging file của Windows; có thể giảm EMBED_BATCH_SIZE trong .env."
)

_MEMORY_MARKERS = ("out of memory", "memory allocation", "not enough memory", "paging file")


def is_memory_error(exc: BaseException) -> bool:
    try:
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except ImportError:
        pass
    return isinstance(exc, MemoryError) or any(m in str(exc).lower() for m in _MEMORY_MARKERS)


def _chain(exc):
    """exc and everything it was raised from, outermost first."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        yield exc
        exc = exc.__cause__ or exc.__context__


def _short(exc, limit=300) -> str:
    text = " ".join(str(exc).split())
    return text[:limit] + ("..." if len(text) > limit else "")


def explain_error(exc: BaseException) -> AppError:
    """Map any exception to an AppError. Looks through the whole cause chain, since
    libraries wrap the interesting error (e.g. LangChain wraps Google's ClientError)."""
    import elasticsearch
    import httpx
    from google.genai import errors as genai_errors
    from langgraph.errors import GraphRecursionError

    chain = list(_chain(exc))

    for e in chain:
        if isinstance(e, AppError):
            return e

    for e in chain:
        if isinstance(e, GraphRecursionError):
            return AppError(
                "Agent đã tìm kiếm quá nhiều bước mà vẫn chưa đưa ra được câu trả lời.",
                "Thử hỏi cụ thể hơn hoặc tách thành nhiều câu hỏi nhỏ. "
                f"(Giới hạn hiện tại: AGENT_RECURSION_LIMIT={settings.agent_recursion_limit})",
            )

        if isinstance(e, elasticsearch.ConnectionTimeout):
            return AppError(
                f"Elasticsearch phản hồi quá lâu (quá {settings.es_timeout:g} giây).",
                "Elasticsearch có thể đang quá tải hoặc thiếu RAM. Thử lại sau ít phút, hoặc tăng ES_TIMEOUT.",
            )
        if isinstance(e, elasticsearch.ConnectionError):
            return AppError(
                f"Không kết nối được Elasticsearch ({settings.es_host}).",
                "Kiểm tra Docker đã chạy chưa: `docker compose up -d` trong thư mục project.",
            )
        if isinstance(e, elasticsearch.NotFoundError) and "index_not_found" in str(e):
            return AppError(
                "Kho tài liệu chưa có dữ liệu.",
                "Thêm tài liệu ở trang **Thêm tài liệu** trước khi đặt câu hỏi.",
            )
        if isinstance(e, elasticsearch.ApiError):
            return AppError(f"Elasticsearch báo lỗi: {_short(e)}")

        if type(e).__name__ == "GoogleContextOverflowError":
            return AppError(
                "Câu hỏi cùng ngữ cảnh vượt quá giới hạn token của model.",
                "Bắt đầu cuộc trò chuyện mới, hoặc giảm MAX_HISTORY_MESSAGES / RETRIEVAL_FINAL_TOP_N trong .env.",
            )
        if isinstance(e, genai_errors.APIError):
            code = e.code
            if code == 429:
                return AppError(
                    "Đã vượt giới hạn gọi Gemini (rate limit / quota).",
                    "Bản miễn phí giới hạn số request mỗi phút và mỗi ngày; hệ thống đã tự thử lại nhưng vẫn bị "
                    "từ chối. Đợi một lát rồi hỏi lại, hoặc xem quota trong Google AI Studio.",
                )
            if code in (401, 403):
                return AppError("Gemini từ chối API key.", "Kiểm tra GOOGLE_API_KEY trong file .env.")
            if code == 404:
                return AppError(
                    f"Không tìm thấy model Gemini '{settings.llm_model}'.", "Kiểm tra LLM_MODEL trong file .env."
                )
            if code is not None and code >= 500:
                return AppError(
                    f"Gemini đang gặp sự cố hoặc quá tải (lỗi {code}).",
                    "Hệ thống đã tự thử lại nhưng vẫn lỗi. Thử lại sau ít phút.",
                )
            return AppError(f"Gemini từ chối yêu cầu (lỗi {code}): {_short(e)}")
        if isinstance(e, httpx.TimeoutException):
            return AppError(
                f"Gemini phản hồi quá lâu (quá {settings.llm_timeout:g} giây).",
                "Mạng chậm hoặc Gemini đang quá tải. Thử lại, hoặc tăng LLM_TIMEOUT trong .env.",
            )
        if isinstance(e, httpx.NetworkError):
            return AppError("Không kết nối được tới Gemini.", "Kiểm tra kết nối Internet rồi thử lại.")

        if is_memory_error(e):
            return AppError("Không đủ bộ nhớ để chạy model.", MEMORY_HINT)

    return AppError(
        f"Lỗi không xác định ({type(exc).__name__}): {_short(exc)}",
        "Xem log trong terminal đang chạy app để biết chi tiết.",
    )
