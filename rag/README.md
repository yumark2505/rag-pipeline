# RAG Chatbot

> Chatbot trả lời câu hỏi dựa trên kho tài liệu riêng (private knowledge base) bằng kỹ thuật **Retrieval-Augmented Generation (RAG)** — truy xuất ngữ nghĩa và sinh câu trả lời có trích dẫn nguồn.

---

## Mô tả

**RAG Chatbot** là một hệ thống hỏi đáp thông minh cho phép người dùng đặt câu hỏi bằng ngôn ngữ tự nhiên về nội dung các tài liệu (PDF, TXT, MD, ...) trong kho tài liệu của mình.

Thay vì chỉ dựa vào kiến thức có sẵn của mô hình ngôn ngữ, hệ thống sẽ:

1. **Đọc** toàn bộ tài liệu và chia nhỏ thành các đoạn ngữ nghĩa.
2. **Mã hóa** các đoạn thành vector và lưu vào cơ sở dữ liệu vector.
3. Mỗi lần hỏi, **truy xuất** những đoạn liên quan nhất và **sinh câu trả lời** chỉ dựa trên các đoạn đó (kèm trích dẫn nguồn `(source:page)`).

Điều này giúp câu trả lời **chính xác, cập nhật theo dữ liệu riêng của bạn** và **giảm thiểu hiện tượng "ảo giác" (hallucination)** nhờ ràng buộc nghiêm ngặt chỉ dùng context được cung cấp.

---

## Tính năng chính

- **Pipeline RAG 9 giai đoạn** hoàn chỉnh, từ nạp tài liệu đến sinh câu trả lời:
  `Loader -> Chunking -> Embedding -> Vector DB -> Pre-retrieval -> Retrieval -> Post-retrieval -> Prompt -> Generation`.
- **Semantic Chunking**: chia tài liệu theo ngữ nghĩa (`SemanticChunker`), không cắt cứng theo số token.
- **Tiền xử lý truy vấn (Pre-retrieval)**: viết lại câu hỏi thành câu truy vấn rõ ràng, sau đó phân rã thành nhiều câu hỏi con để bao phủ các khía cạnh khác nhau.
- **Đa truy vấn + chống trùng lặp**: truy xuất theo nhiều câu hỏi con, loại bỏ kết quả trùng và lọc theo ngưỡng điểm tương đồng.
- **Hậu xử lý (Post-retrieval)**: sắp hạng lại (rerank) các đoạn theo độ liên quan và nén context để vừa ngân sách token.
- **Trả lời kèm trích dẫn**: câu trả lời chỉ dựa trên context, trích nguồn theo định dạng `(source:page)`.
- **Index bền vững**: chỉ số FAISS được lưu đĩa và tải lại ở lần chạy sau, không phải dựng lại từ đầu.
- **Giao diện CLI** tương tác trực tiếp.

---

## Công nghệ sử dụng

| Thành phần | Công nghệ |
| ---------- | --------- |
| Ngôn ngữ | Python |
| Framework LLM | [LangChain](https://www.langchain.com/) (Core, Community, Experimental) |
| Embedding model | `nomic-embed-text` (chạy qua Ollama) |
| Chat model | `qwen2.5:7b` (chạy qua Ollama) |
| Vector database | FAISS (`faiss-cpu`) |
| Loader tài liệu | `DirectoryLoader` + `UnstructuredFileLoader` (hỗ trợ PDF, TXT, MD, ...) |
| Serving LLM | [Ollama](https://ollama.com/) |

---

## Hướng dẫn cài đặt và chạy local

### 1. Yêu cầu tiên quyết (Prerequisites)

- **Python 3.10+**
- **[Ollama](https://ollama.com/)** đang chạy trên máy (mặc định tại `http://127.0.0.1:11434`).

Kéo các model cần thiết về:

```bash
ollama pull nomic-embed-text
ollama pull qwen2.5:7b
```

### 2. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

> Nên sử dụng **virtual environment** để tránh xung đột gói:

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate # macOS / Linux
```

### 3. Chuẩn bị dữ liệu

Đặt các tài liệu (PDF, TXT, MD, ...) vào thư mục `papers/` trong gốc dự án:

```text
papers/
├── paper1.pdf
├── paper2.pdf
└── ...
```

> Mặc định loader quét tất cả file `**/*.pdf`. Có thể chỉnh sửa trong `rag/config.py`.

### 4. Cấu hình biến môi trường (tùy chọn)

Tạo file `.env` ở gốc dự án nếu bạn cần thay đổi cấu hình mặc định:

| Biến | Mô tả | Giá trị mặc định |
| ---- | ----- | ---------------- |
| `OLLAMA_HOST` | Địa chỉ server Ollama | `http://127.0.0.1:11434` |

```bash
# .env
OLLAMA_HOST=http://127.0.0.1:11434
```

### 5. Chạy chatbot

```bash
python -m rag.main
```

Khi khởi động, hệ thống sẽ tự dựng index (lần đầu) hoặc tải index đã lưu (các lần sau), sau đó mở phiên chat CLI:

```text
Chat ready. Type 'exit' or 'quit' to stop.

Your question: Tóm tắt ý chính của tài liệu về...
Answer:
...
```

Gõ `exit` hoặc `quit` để thoát.

---

## Cấu trúc thư mục

```text
rag/
├── main.py            # Điểm khởi chạy CLI
├── pipeline.py        # Điều phối toàn bộ pipeline 9 giai đoạn
├── config.py          # Cấu hình tập trung (model, host, tham số)
├── loader.py          # Giai đoạn 1: Nạp tài liệu
├── chunker.py         # Giai đoạn 2: Chia đoạn ngữ nghĩa
├── embedder.py        # Giai đoạn 3: Mã hóa vector (Ollama, batch-safe)
├── vectorstore.py     # Giai đoạn 4: FAISS index (build/save/load/query)
├── retriever.py       # Giai đoạn 5-7: Pre-retrieval, Retrieval, Post-retrieval
├── prompt.py          # Giai đoạn 8-9: Prompt + Generator
└── data/faiss/        # Index FAISS đã lưu
```

---

## Hướng dẫn đóng góp (Contribution)

Mọi đóng góp đều được hoan nghênh! Để đóng góp:

1. **Fork** repository và tạo nhánh mới:

```bash
git checkout -b feature/ten-tinh-nang
```

2. **Commit** thay đổi với thông điệp rõ ràng:

```bash
git commit -m "feat: mô tả ngắn gọn về thay đổi"
```

3. **Push** nhánh lên và tạo **Pull Request**:

```bash
git push origin feature/ten-tinh-nang
```

> Vui lòng đảm bảo mã của bạn chạy được, trước khi tạo PR hãy kiểm tra kỹ luồng `build_index()` và `answer()`.

---

## Nguồn cảm hứng (Acknowledgements)

Đồ án được lấy cảm hứng từ [**RAG-pipeline-visualizer**](https://github.com/vietnh1009/RAG-pipeline-visualizer) của [vietnh1009](https://github.com/vietnh1009) — ứng dụng giúp xây dựng, cấu hình và trực quan hóa từng bước trong pipeline RAG, từ load tài liệu PDF đến sinh câu trả lời bằng LLM.

---

## Giấy phép

Dự án sử dụng cho mục đích học tập và nghiên cứu. Mọi nhãn hiệu thuộc về chủ sở hữu tương ứng.
