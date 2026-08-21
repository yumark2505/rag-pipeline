# RAG Pipeline
> Ứng dụng hỏi đáp trên tài liệu PDF bằng pipeline RAG (Retrieval-Augmented Generation) — từ bước nạp tài liệu đến bước sinh câu trả lời bằng LLM.

---

## Mô tả

**RAG Pipeline** là phiên bản nâng cao của chatbot hỏi đáp trên kho tài liệu riêng, được thiết kế theo mẫu **Strategy + Registry**:

- Mỗi giai đoạn của pipeline (loader, chunker, embedder, vector store, retriever, prompt, ...) là một **interface trừu tượng** trong `base.py`.
- Các cài đặt cụ thể tự đăng ký vào registry qua decorator `@register_*`.
- Tại thời điểm khởi tạo, bạn **chọn tham số cho từng giai đoạn ** — không cần sửa code.

Nhờ vậy, bạn có thể thử nghiệm nhanh các tổ hợp khác nhau (ví dụ: so sánh `faiss` với `chroma`, hay `bm25` với `hybrid`) chỉ bằng cách đổi tham số dòng lệnh.

### Luồng xử lý 9 giai đoạn

```text
1. Loader          -->  2. Chunking     -->  3. Embedding    -->  4. Vector DB
                                                                       |
6. Retriever       <--  5. Pre-retrieval (rewrite + decompose câu hỏi)
   |
7. Post-retrieval  -->  8. Prompt       -->  9. Generation   -->  Answer
   (rerank + compress)
```

---

## Tính năng chính

- **Kiến trúc pluggable**: chọn chiến lược cho từng stage bằng tên qua CLI flag hoặc menu tương tác.
- **3 loader PDF**: `pypdf` (nhanh), `unstructured` (đa định dạng), `opendataloader` (chất lượng cao nhất, hỗ trợ bảng/bounding box).
- **3 chunking**: `recursive`, `token` (kích thước + overlap), `semantic` (chia theo ngữ nghĩa).
- **3 embedding**: Ollama (local), OpenAI API, HuggingFace local — LLM sinh câu trả lời **tự động khớp** với provider của embedder.
- **3 vector store**: FAISS, Chroma, PGVector (PostgreSQL).
- **3 retriever**:
  - `vector` — tìm kiếm ngữ nghĩa thuần.
  - `bm25` — tìm kiếm từ khóa (lexical).
  - `hybrid` — hợp nhất BM25 + vector, chuẩn hóa min-max rồi cộng trọng số (mặc định 50/50).
- **Pre-retrieval**: viết lại câu hỏi thành truy vấn độc lập và phân rã thành tối đa 3 câu hỏi con; kết quả truy xuất được hợp nhất và loại trùng theo nội dung.
- **Post-retrieval**: rerank bằng LLM rồi nén context về tối đa 6000 ký tự.
- **Prompt**: chế độ multi-turn giữ lịch sử chat (6 lượt gần nhất) để hỏi câu hỏi nối tiếp.
- **Trả lời kèm trích dẫn** `(source:page)`, từ chối trả lời nếu context không chứa thông tin (chống hallucination).
- **Index bền vững**: FAISS/Chroma lưu đĩa và tự tải lại ở lần chạy sau.
- **Logging cấu hình được**: `DEBUG` hiển thị chi tiết retrieval, `INFO` hiển thị tóm tắt từng stage.

---

## Công nghệ sử dụng

| Thành phần | Công nghệ |
| ---------- | --------- |
| Ngôn ngữ | Python |
| Framework LLM | [LangChain](https://www.langchain.com/) (Core, Community, Experimental, Ollama, OpenAI, HuggingFace) |
| LLM mặc định | `qwen2.5:7b` (Ollama), `gpt-4o-mini` (OpenAI), `Qwen2.5-1.5B-Instruct` (HF local) |
| Embedding | `nomic-embed-text`, `text-embedding-3-small`, `all-MiniLM-L6-v2` |
| Vector DB | FAISS, Chroma, PGVector |
| Retrieval bổ sung | `rank-bm25` |
| Cấu hình | `python-dotenv` |

---

## Hướng dẫn cài đặt và chạy local

### 1. Yêu cầu

- **Python 3.10 - 3.12**
- **[Ollama](https://ollama.com/)** (mặc định `http://127.0.0.1:11434`) nếu dùng embedder `ollama`.

```bash
ollama pull nomic-embed-text
ollama pull qwen2.5:7b
```

> Nếu dùng `pgvector`, cần một PostgreSQL có extension `pgvector` và database khớp `PGVECTOR_URL`.

### 2. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

> Nên dùng virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate # macOS / Linux
```

> Gói `langchain-opendataloader-pdf` và `opendataloader-pdf` chỉ cần cài thêm nếu chọn loader `opendataloader`.

### 3. Chuẩn bị dữ liệu

Thư mục `papers/` **không đi kèm repo** — hãy tự tạo ở gốc dự án rồi đặt tài liệu PDF của bạn vào:

```text
papers/              <- tu tao
├── paper1.pdf
└── paper2.pdf
```

### 4. Biến môi trường (.env)

Tạo file `.env` ở gốc dự án (được tự động nạp bởi `python-dotenv`):

| Biến | Mô tả | Mặc định |
| ---- | ----- | -------- |
| `OLLAMA_HOST` | Địa chỉ server Ollama | `http://127.0.0.1:11434` |
| `PGVECTOR_URL` | Chuỗi kết nối PostgreSQL (khi dùng `pgvector`) | `postgresql://postgres:postgres@localhost:5432/rag` |
| `RAG_LOG_LEVEL` | Mức log: `DEBUG` / `INFO` / `WARNING` / `ERROR` | `INFO` |

```bash
# .env
OLLAMA_HOST=http://127.0.0.1:11434
RAG_LOG_LEVEL=DEBUG
```

> API key OpenAI hiện được đọc từ `OPENAI_API_KEY` trong `config.py` — cập nhật tại đó nếu dùng embedder/LLM của OpenAI.

### 5. Chạy pipeline

**Cách 1 — Interactive menu** :

```bash
python -m rag_pipeline.main
```

**Cách 2 — command line**:

```bash
python -m rag_pipeline.pipeline \
    --loader pypdf \
    --chunker semantic \
    --embedder ollama \
    --vectorstore faiss \
    --retriever hybrid \
    --pre-retrieval query_transform \
    --post-retrieval rerank \
    --prompt conversational \
    --log-level INFO
```

Các giá trị cho từng flag:

| Flag | Lựa chọn | Mặc định |
| ---- | -------- | -------- |
| `--loader` | `pypdf`, `unstructured`, `opendataloader` | `pypdf` |
| `--chunker` | `recursive`, `token`, `semantic` | `semantic` |
| `--embedder` | `ollama`, `openai`, `huggingface` | `ollama` |
| `--vectorstore` | `faiss`, `chroma`, `pgvector` | `faiss` |
| `--retriever` | `vector`, `bm25`, `hybrid` | `vector` |
| `--pre-retrieval` | `identity`, `query_transform` | `query_transform` |
| `--post-retrieval` | `basic`, `rerank` | `rerank` |
| `--prompt` | `basic`, `conversational` | `basic` |

Sau khi index sẵn sàng, phiên chat CLI mở ra:

```text
Chat ready (loader=pypdf, chunker=semantic, ...)
Type 'exit' or 'quit' to stop.

Your question: ...
Answer:
...
```

---

## Cấu trúc thư mục

```text
rag_pipeline/
├── main.py                  # Trình chọn chiến lược tương tác
├── pipeline.py              # RAGPipelineAdvanced + CLI argparse
├── config.py                # Toàn bộ hằng số cấu hình
├── base.py                  # Registry + interface Base* cho từng stage
├── logging_setup.py         # Cấu hình logging chung
├── loaders/                 # Stage 1: pypdf | unstructured | opendataloader
├── chunkers/                # Stage 2: recursive | token_based | semantic
├── embedders/               # Stage 3: ollama | openai | huggingface
├── vectorstores/            # Stage 4: faiss | chroma | pgvector
├── pre_retrieval/           # Stage 5: identity | query_transform
├── retrieval/               # Stage 6: vector | bm25 | hybrid
├── post_retrieval/          # Stage 7: basic_format | rerank_compress
├── prompts/                 # Stage 8: basic | conversational
└── data/
    ├── faiss/               # Index FAISS đã lưu
    └── chroma/              # Database Chroma đã lưu
```

### Mở rộng pipeline

Thêm một chiến lược mới chỉ cần 3 bước:

1. Tạo class mới kế thừa interface `Base*` tương ứng trong `base.py`.
2. Đăng ký class vào registry của stage bằng decorator `@register_<stage>("New")`.
3. Import module chứa class đó trong `pipeline.py` để decorator được thực thi khi khởi động.

```python
from rag_pipeline.base import BaseChunker
from rag_pipeline.chunkers import register_chunker


@register_chunker("my_chunker")
class MyChunker(BaseChunker):
    def split_documents(self, docs):
        ...
```

---

## Nguồn cảm hứng (Acknowledgements)

Đồ án được lấy cảm hứng từ [**RAG-pipeline-visualizer**](https://github.com/vietnh1009/RAG-pipeline-visualizer) của [vietnh1009](https://github.com/vietnh1009) — ứng dụng giúp xây dựng, cấu hình và trực quan hóa từng bước trong pipeline RAG. Kiến trúc pluggable theo stage (loader / chunking / embedding / vector DB / pre-retrieval / retrieval / post-retrieval / prompt) của dự án này được kế thừa từ cách tổ chức module của repo gốc.

---

## Giấy phép

Dự án sử dụng cho mục đích học tập và nghiên cứu. Mọi nhãn hiệu thuộc về chủ sở hữu tương ứng.

