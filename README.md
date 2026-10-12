# Hỏi đáp tài liệu (RAG)

Chatbot trả lời câu hỏi dựa trên kho tài liệu của bạn: upload PDF / DOCX / HTML / Markdown, hệ thống tách nội dung (kể cả bảng), embed bằng **bge-m3** (dense + sparse), lưu vào **Elasticsearch**, tìm kiếm hybrid + rerank (**bge-reranker-v2-m3**) và trả lời bằng agent **LangGraph + Gemini**.

```
Trình duyệt ─► Frontend React (:5173) ─/api─► Backend FastAPI (:8000) ─► Elasticsearch (:9200)
                                              model, agent, SQLite         └─► Gemini, Langfuse
```

Câu trả lời và tiến độ embed được đẩy về trình duyệt qua **SSE** (Server-Sent Events).

## Yêu cầu

- Python 3.10+, Node.js 20+
- Docker (Elasticsearch + Kibana)
- GPU NVIDIA (khuyến nghị; không có GPU vẫn chạy được trên CPU nhưng chậm)
- Google API key cho Gemini: https://aistudio.google.com/apikey

## Cài đặt

```bash
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate

# PyTorch bản CUDA phải cài TRƯỚC, nếu không pip sẽ kéo bản chỉ chạy CPU
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

copy .env.example .env          # Linux/macOS: cp .env.example .env

cd frontend && npm install
```

Mở `.env`, điền `GOOGLE_API_KEY` (bắt buộc) và key Langfuse nếu muốn tracing. Các tham số khác (model, timeout, số kết quả tìm kiếm, kích thước chunk...) đều có giá trị mặc định, xem giải thích trong `.env.example`.

## Chạy

```bash
docker compose up -d                                  # Elasticsearch :9200, Kibana :5601

cd backend && uvicorn app.main:app --port 8000        # terminal 1: backend (load model ~1 phút)
cd frontend && npm run dev                            # terminal 2: frontend
```

Mở http://localhost:5173. Lần chạy đầu sẽ tải model từ HuggingFace (vài GB). Kho tài liệu ban đầu trống: vào **Tài liệu** để upload, xem lại cấu trúc và bảng phát hiện được, rồi xác nhận để embed.

- Chỉ chạy **1 worker** uvicorn: mỗi worker load một bản model riêng.
- Tài liệu API (thử trực tiếp được): http://localhost:8000/docs

## Phát triển

```bash
cd backend && python -m pytest -q                     # test backend (chạy trên fake, không cần ES/GPU/Gemini)

python backend/scripts/export_openapi.py              # sau khi đổi schema ở backend:
cd frontend && npm run gen:api                        #   sinh lại kiểu TypeScript cho frontend
```

## Quản lý index

Code luôn dùng alias `document-chunks`, dữ liệu thật nằm ở index có version (`document-chunks-v1`, `-v2`...).

```bash
python backend/scripts/index_admin.py status                     # alias trỏ vào version nào, mỗi version bao nhiêu docs
python backend/scripts/index_admin.py reindex                    # sau khi sửa mapping trong es_index.py: tạo version mới, copy dữ liệu, chuyển alias
python backend/scripts/index_admin.py switch document-chunks-v1  # quay lại version cũ
python backend/scripts/index_admin.py delete document-chunks-v1  # xoá version không còn dùng
```

Thêm field mới vào mapping thì không cần reindex: backend tự cập nhật ở lần upload tiếp theo.

## Cấu trúc

| Đường dẫn | Vai trò |
|---|---|
| `backend/app/main.py` | FastAPI: load model một lần, đăng ký các API |
| `backend/app/api/` | Endpoint: chat (SSE), cuộc trò chuyện, tài liệu, job (SSE), health |
| `backend/app/services/` | Logic chat, parse + job embed/index, quản lý job nền |
| `backend/app/schemas.py` | Request/response và các loại event SSE |
| `backend/app/core/` | Lõi RAG: agent LangGraph, hybrid search + rerank, Docling, chunking, mapping ES, SQLite, config, lỗi, tracing |
| `backend/scripts/` | `index_admin.py` (quản lý version index), `export_openapi.py` |
| `backend/tests/` | Test API trên fake |
| `frontend/src/` | React + TypeScript: trang Trò chuyện, trang Tài liệu, tầng gọi API/SSE |
| `notebook/` | Notebook từng bước: tách + embed + index *The Great Gatsby*, thử search |
