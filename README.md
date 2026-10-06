# Hỏi đáp tài liệu (RAG)

Chatbot trả lời câu hỏi dựa trên kho tài liệu của bạn: upload PDF / DOCX / HTML / Markdown, hệ thống tách nội dung (kể cả bảng), embed bằng **bge-m3** (dense + sparse), lưu vào **Elasticsearch**, tìm kiếm hybrid + rerank (**bge-reranker-v2-m3**) và trả lời bằng agent **LangGraph + Gemini**.

## Yêu cầu

- Python 3.10+
- Docker (chạy Elasticsearch + Kibana)
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
```

Mở `.env`, điền `GOOGLE_API_KEY` (bắt buộc) và key Langfuse nếu muốn tracing. Các tham số khác (model, số kết quả tìm kiếm, kích thước chunk...) đều có giá trị mặc định, xem giải thích trong `.env.example`.

## Chạy

```bash
docker compose up -d            # Elasticsearch :9200, Kibana :5601
streamlit run streamlit_app.py
```

Lần chạy đầu sẽ tải model từ HuggingFace (vài GB). Kho tài liệu ban đầu trống: vào trang **Thêm tài liệu** để upload, xem lại cấu trúc + bảng phát hiện được, rồi xác nhận để embed.

## Quản lý index

Code luôn dùng alias `document-chunks`, dữ liệu thật nằm ở index có version (`document-chunks-v1`, `-v2`...).

```bash
python index_admin.py status                     # alias đang trỏ vào version nào, mỗi version bao nhiêu docs
python index_admin.py reindex                    # sau khi sửa mapping trong es_index.py: tạo version mới, copy dữ liệu, chuyển alias
python index_admin.py switch document-chunks-v1  # quay lại version cũ
python index_admin.py delete document-chunks-v1  # xoá version không còn dùng
```

Thêm field mới vào mapping thì không cần reindex: app tự cập nhật ở lần upload tiếp theo.

## Cấu trúc

| File | Vai trò |
|---|---|
| `streamlit_app.py` | Giao diện chat + upload tài liệu |
| `langgraph_agent.py` | Agent LangGraph, tool `search_documents` |
| `rag_pipeline.py` | Embed, hybrid search (kNN + sparse), RRF, rerank |
| `document_ingest.py` | Đọc tài liệu bằng Docling → cây heading + bảng |
| `chunking.py` | Chia chunk theo token, overlap, tách bảng theo hàng |
| `es_index.py` / `index_admin.py` | Mapping index / công cụ quản lý version |
| `chat_store.py` | Lưu lịch sử chat (SQLite) |
| `config.py` | Đọc cấu hình từ `.env` |
| `tracing.py` | Tracing Langfuse (tuỳ chọn) |
| `notebook/` | Notebook từng bước: tách + embed + index *The Great Gatsby*, thử search |
