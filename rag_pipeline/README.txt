================================================================================
                                RAG PIPELINE
     Pipeline Retrieval-Augmented Generation (RAG) dang module hoa (pluggable)
================================================================================

Pipeline RAG moi giai doan deu co the thay the chien luoc (strategy) chi bang
mot ten, chay qua CLI hoac trinh chon tuong tac.

--------------------------------------------------------------------------------
1. MO TA
--------------------------------------------------------------------------------

RAG Pipeline la phien ban nang cao cua chatbot hoi dap tren kho tai lieu rieng,
thiet ke theo mau STRATEGY + REGISTRY:

  - Moi giai doan cua pipeline (loader, chunker, embedder, vector store,
    retriever, prompt, ...) la mot interface truu tuong trong base.py.
  - Cai cu the tu dang ky vao registry qua decorator @register_*.
  - Tai thoi diem khoi tao, ban chon chien luoc cho tung giai doan bang ten,
    khong can sua code.

Nho vay, ban co the thu nghiem nhanh cac to hop khac nhau (vi du: so sanh
faiss voi chroma, hay bm25 voi hybrid) chi bang cach doi tham so dong lenh.

LUU XU LY 9 GIAI DOAN:

  1. Loader          -->  2. Chunking      -->  3. Embedding   -->  4. Vector DB
                                                                       |
  6. Retriever      <--  5. Pre-retrieval (rewrite + decompose cau hoi)
      |
  7. Post-retrieval -->  8. Prompt       -->  9. Generation  -->  Answer
      (rerank + compress)

--------------------------------------------------------------------------------
2. TINH NANG CHINH
--------------------------------------------------------------------------------

  [+] Kien truc pluggable: chon chien luoc tung stage bang ten qua CLI flag
      hoac menu tuong tac.

  [+] 3 loader PDF:
        - pypdf         : nhanh
        - unstructured  : da dinh dang
        - opendataloader: chat luong cao nhat, ho tro bang / bounding box

  [+] 3 chien luoc chunking:
        - recursive : chia de quy theo ky tu
        - token     : kich thuoc + overlap
        - semantic  : chia theo ngu nghia

  [+] 3 nha cung cap embedding:
        - ollama     : nomic-embed-text (local)
        - openai     : text-embedding-3-small (API)
        - huggingface: all-MiniLM-L6-v2 (local)
      LLM sinh cau tra loi TU DONG KHOP voi provider cua embedder.

  [+] 3 vector store: FAISS, Chroma, PGVector (PostgreSQL).

  [+] 3 retriever:
        - vector : tim kiem ngu nghia thuan
        - bm25   : tim kiem tu khoa (lexical)
        - hybrid : hop nhat BM25 + vector, chuan hoa min-max roi cong
                   trong so (mac dinh 50/50)

  [+] Pre-retrieval: viet lai cau hoi thanh truy van doc lap va phan ra thanh
      toi da 3 cau hoi con; ket qua truy xuat duoc hop nhat va loai trung
      theo noi dung.

  [+] Post-retrieval: rerank bang LLM roi nen context ve toi da 6000 ky tu.

  [+] Prompt hoi thoai: che do multi-turn giu lich su chat (6 luot gan nhat)
      de hoi cau hoi noi tiep.

  [+] Tra loi kem trich dan (source:page), tu choi tra loi neu context khong
      chua thong tin (chong hallucination).

  [+] Index ben vung: FAISS/Chroma luu dia va tu tai lai o lan chay sau.

  [+] Logging cau hinh duoc:
        - DEBUG: hien thi chi tiet retrieval
        - INFO : hien thi tom tat tung stage

--------------------------------------------------------------------------------
3. CONG NGHE SU DUNG
--------------------------------------------------------------------------------

  Ngon ngu            : Python
  Framework LLM       : LangChain (Core, Community, Experimental,
                        Ollama, OpenAI, HuggingFace)
  LLM mac dinh        : qwen2.5:7b (Ollama)
                        gpt-4o-mini (OpenAI)
                        Qwen2.5-1.5B-Instruct (HF local)
  Embedding           : nomic-embed-text | text-embedding-3-small |
                        all-MiniLM-L6-v2
  Vector DB           : FAISS | Chroma | PGVector
  Retrieval bo sung   : rank-bm25
  Cau hinh            : python-dotenv

--------------------------------------------------------------------------------
4. HUONG DAN CAI DAT VA CHAY LOCAL
--------------------------------------------------------------------------------

4.1. YEU CAU TIEN QUYET
-----------------------
  - Python 3.10+
  - Ollama dang chay (mac dinh http://127.0.0.1:11434) neu dung embedder
    "ollama". Tai model:

        ollama pull nomic-embed-text
        ollama pull qwen2.5:7b

  - Neu dung pgvector: can PostgreSQL co extension pgvector va database
    khop PGVECTOR_URL.

4.2. CAI DAT THU VIEN
---------------------
  Nen dung virtual environment:

        python -m venv .venv
        .venv\Scripts\activate          (Windows)
        source .venv/bin/activate       (macOS / Linux)

        pip install -r requirements.txt

  Ghi chu: go langchain-opendataloader-pdf va opendataloader-pdf chi can cai
  them neu chon loader "opendataloader".

4.3. CHUAN BI DU LIEU
---------------------
  Dat tai lieu PDF vao thu muc papers/ o goc du an:

        papers/
            paper1.pdf
            paper2.pdf

4.4. BIEN MOI TRUONG (.env)
---------------------------
  Tao file .env o goc du an (tu dong nap boi python-dotenv):

  +----------------+---------------------------------------+------------------------+
  | Bien           | Mo ta                                 | Mac dinh               |
  +----------------+---------------------------------------+------------------------+
  | OLLAMA_HOST    | Dia chi server Ollama                 | http://127.0.0.1:11434 |
  | PGVECTOR_URL   | Chuoi ket noi PostgreSQL (pgvector)   | postgresql://postgres: |
  |                |                                       | postgres@localhost:    |
  |                |                                       | 5432/rag               |
  | RAG_LOG_LEVEL  | Muc log: DEBUG/INFO/WARNING/ERROR     | INFO                   |
  +----------------+---------------------------------------+------------------------+

  Vi du file .env:

        OLLAMA_HOST=http://127.0.0.1:11434
        RAG_LOG_LEVEL=DEBUG

  Ghi chu: API key OpenAI hien doc tu OPENAI_API_KEY trong config.py -
  cap nhat tai do neu dung embedder/LLM cua OpenAI.

4.5. CHAY PIPELINE
------------------
  Cach 1 - Trinh chon tuong tac (hoi tung stage):

        python -m rag_pipeline.main

  Cach 2 - Dong lenh day du (khong tuong tac):

        python -m rag_pipeline.pipeline ^
            --loader pypdf ^
            --chunker semantic ^
            --embedder ollama ^
            --vectorstore faiss ^
            --retriever hybrid ^
            --pre-retrieval query_transform ^
            --post-retrieval rerank ^
            --prompt conversational ^
            --log-level INFO

  (Tren macOS/Linux thay ^ bang \)

  Cac gia tri kha dung cho tung flag:

  Flag              Lua chon                                    Mac dinh
  ----------------  ------------------------------------------  ----------------
  --loader          pypdf | unstructured | opendataloader       pypdf
  --chunker         recursive | token | semantic                semantic
  --embedder        ollama | openai | huggingface               ollama
  --vectorstore     faiss | chroma | pgvector                   faiss
  --retriever       vector | bm25 | hybrid                      vector
  --pre-retrieval   identity | query_transform                  query_transform
  --post-retrieval  basic | rerank                              rerank
  --prompt          basic | conversational                      basic

  Sau khi index san sang, phien chat CLI mo ra:

        Chat ready (loader=pypdf, chunker=semantic, ...)
        Type 'exit' or 'quit' to stop.

        Your question: ...
        Answer:
        ...

--------------------------------------------------------------------------------
5. CAU TRUC THU MUC
--------------------------------------------------------------------------------

  rag_pipeline/
  |-- main.py                  Trinh chon chien luoc tuong tac
  |-- pipeline.py              RAGPipelineAdvanced + CLI argparse
  |-- config.py                Toan bo hang so cau hinh
  |-- base.py                  Registry + interface Base* cho tung stage
  |-- logging_setup.py         Cau hinh logging chung
  |-- loaders/                 Stage 1: pypdf | unstructured | opendataloader
  |-- chunkers/                Stage 2: recursive | token_based | semantic
  |-- embedders/               Stage 3: ollama | openai | huggingface
  |-- vectorstores/            Stage 4: faiss | chroma | pgvector
  |-- pre_retrieval/           Stage 5: identity | query_transform
  |-- retrieval/               Stage 6: vector | bm25 | hybrid
  |-- post_retrieval/          Stage 7: basic_format | rerank_compress
  |-- prompts/                 Stage 8: basic | conversational
  \-- data/
      |-- faiss/               Index FAISS da luu
      \-- chroma/              Database Chroma da luu

5.1. MO RONG PIPELINE
---------------------
  Them mot chien luoc moi chi can 3 buoc:

  Buoc 1. Tao class ke thua Base* tuong ung trong base.py.
  Buoc 2. Trang tri bang decorator @register_<stage>("ten-moi").
  Buoc 3. Import module do trong pipeline.py de decorator chay.

  Vi du:

        from rag_pipeline.base import BaseChunker
        from rag_pipeline.chunkers import register_chunker


        @register_chunker("my_chunker")
        class MyChunker(BaseChunker):
            def split_documents(self, docs):
                ...

--------------------------------------------------------------------------------
6. HUONG DAN DONG GOP (CONTRIBUTION)
--------------------------------------------------------------------------------

  Moi dong gop deu duoc hoan nghenh!

  Buoc 1. Fork repository va tao nhanh moi:

        git checkout -b feature/ten-tinh-nang

  Buoc 2. Commit voi thong diep ro rang:

        git commit -m "feat: them strategy moi cho stage X"

  Buoc 3. Push va tao Pull Request:

        git push origin feature/ten-tinh-nang

  Luu y: Khi them strategy moi, vui long dang ky dung registry cua stage,
  giu nguyen interface Base*, va kiem tra pipeline chay duoc voi ca to hop
  mac dinh lan to hop co strategy moi.

--------------------------------------------------------------------------------
7. NGUON CAM HUNG (ACKNOWLEDGEMENTS)
--------------------------------------------------------------------------------

  Do an duoc lay cam hung tu RAG-pipeline-visualizer cua vietnh1009:

      https://github.com/vietnh1009/RAG-pipeline-visualizer

  Ung dung giup xay dung, cau hinh va truc quan hoa tung buoc trong pipeline
  RAG. Kien truc pluggable theo stage (loader / chunking / embedding /
  vector DB / pre-retrieval / retrieval / post-retrieval / prompt) cua du an
  nay duoc ke thua tu cach to chuc module cua repo goc.

--------------------------------------------------------------------------------
8. GIAY PHEP (LICENSE)
--------------------------------------------------------------------------------

  Du an su dung cho muc dich hoc tap va nghien cuu.
  Moi nhan hieu thuoc ve chu so huu tuong ung.

================================================================================
