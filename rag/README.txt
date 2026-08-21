================================================================================
                                 RAG CHATBOT
   Chatbot tra loi cau hoi dua tren kho tai lieu rieng bang ky thuat
   Retrieval-Augmented Generation (RAG)
================================================================================

Truy xuat ngu nghia va sinh cau tra loi co trich dan nguon.

--------------------------------------------------------------------------------
1. MO TA
--------------------------------------------------------------------------------

RAG Chatbot la he thong hoi dap thong minh cho phep nguoi dung dat cau hoi
bang ngon ngu tu nhien ve noi dung cac tai lieu (PDF, TXT, MD, ...) trong kho
tai lieu cua minh.

Thay vi chi dua vao kien thuc co san cua mo hinh ngon ngu, he thong se:

  1. DOC toan bo tai lieu va chia nho thanh cac doan ngu nghia.
  2. MA HOA cac doan thanh vector va luu vao co so du lieu vector.
  3. Moi lan hoi, TRUY XUAT nhung doan lien quan nhat va SINH CAU TRA LOI
     chi dua tren cac doan do (kem trich dan nguon theo dang source:page).

Dieu nay giup cau tra loi CHINH XAC, CAP NHAT theo du lieu rieng cua ban va
GIAM THIEU hien tuong "ao giac" (hallucination) nhan rang buoc nghiem ngat
chi dung context duoc cung cap.

--------------------------------------------------------------------------------
2. TINH NANG CHINH
--------------------------------------------------------------------------------

  [+] Pipeline RAG 9 giai doan hoan chinh:

        Loader -> Chunking -> Embedding -> Vector DB -> Pre-retrieval
        -> Retrieval -> Post-retrieval -> Prompt -> Generation -> Answer

  [+] Semantic Chunking: chia tai lieu theo ngu nghia (SemanticChunker),
      khong cat cung theo so token.

  [+] Tien xu ly truy van (Pre-retrieval): viet lai cau hoi thanh cau truy
      van ro rang, sau do phan ra thanh nhieu cau hoi con de bao phu cac
      khia canh khac nhau.

  [+] Da truy van + chong trung lap: truy xuat theo nhieu cau hoi con,
      loai bo ket qua trung va loc theo nguong diem tuong dong.

  [+] Hau xu ly (Post-retrieval): sap hang lai (rerank) cac doan theo do
      lien quan va nen context de vua ngan sach token.

  [+] Tra loi kem trich dan: cau tra loi chi dua tren context, trich nguon
      theo dinh dang (source:page).

  [+] Index ben vung: chi so FAISS duoc luu dia va tai lai o lan chay sau,
      khong phai dung lai tu dau.

  [+] Giao dien CLI tuong tac truc tiep.

--------------------------------------------------------------------------------
3. CONG NGHE SU DUNG
--------------------------------------------------------------------------------

  Thanh phant        | Cong nghe
  -------------------+----------------------------------------------------------
  Ngon ngu           | Python
  Framework LLM      | LangChain (Core, Community, Experimental)
  Embedding model    | nomic-embed-text (chay qua Ollama)
  Chat model         | qwen2.5:7b (chay qua Ollama)
  Vector database    | FAISS (faiss-cpu)
  Loader tai lieu    | DirectoryLoader + UnstructuredFileLoader
                     | (ho tro PDF, TXT, MD, ...)
  Serving LLM        | Ollama

--------------------------------------------------------------------------------
4. HUONG DAN CAI DAT VA CHAY LOCAL
--------------------------------------------------------------------------------

4.1. YEU CAU TIEN QUYET
-----------------------
  - Python 3.10+
  - Ollama dang chay tren may (mac dinh tai http://127.0.0.1:11434)

  Keo cac model can thiet ve:

        ollama pull nomic-embed-text
        ollama pull qwen2.5:7b

4.2. CAI DAT THU VIEN
---------------------
  Nen su dung virtual environment de tranh xung dot goi:

        python -m venv .venv
        .venv\Scripts\activate          (Windows)
        source .venv/bin/activate       (macOS / Linux)

        pip install -r requirements.txt

4.3. CHUAN BI DU LIEU
---------------------
  Dat cac tai lieu (PDF, TXT, MD, ...) vao thu muc papers/ trong goc du an:

        papers/
            paper1.pdf
            paper2.pdf
            ...

  Mac dinh loader quet tat ca file **/*.pdf. Co the chinh sua trong
  rag/config.py.

4.4. BIEN MOI TRUONG (.env)
---------------------------
  Tao file .env o goc du an neu ban can thay doi cau hinh mac dinh:

  Bien          Mo ta                    Gia tri mac dinh
  ------------  -----------------------  ------------------------
  OLLAMA_HOST   Dia chi server Ollama    http://127.0.0.1:11434

  Vi du file .env:

        OLLAMA_HOST=http://127.0.0.1:11434

4.5. CHAY CHATBOT
-----------------
        python -m rag.main

  Khi khoi dong, he thong se tu dung index (lan dau) hoac tai index da luu
  (cac lan sau), sau do mo phien chat CLI:

        Chat ready. Type 'exit' or 'quit' to stop.

        Your question: Tom tat y chinh cua tai lieu ve...
        Answer:
        ...

  Go "exit" hoac "quit" de thoat.

--------------------------------------------------------------------------------
5. CAU TRUC THU MUC
--------------------------------------------------------------------------------

  rag/
  |-- main.py            Diem khoi chay CLI
  |-- pipeline.py        Dieu phoi toan bo pipeline 9 giai doan
  |-- config.py          Cau hinh tap trung (model, host, tham so)
  |-- loader.py          Giai doan 1: Nap tai lieu
  |-- chunker.py         Giai doan 2: Chia doan ngu nghia
  |-- embedder.py        Giai doan 3: Ma hoa vector (Ollama, batch-safe)
  |-- vectorstore.py     Giai doan 4: FAISS index (build/save/load/query)
  |-- retriever.py       Giai doan 5-7: Pre-retrieval, Retrieval,
  |                      Post-retrieval
  |-- prompt.py          Giai doan 8-9: Prompt + Generator
  \-- data/faiss/        Index FAISS da luu

--------------------------------------------------------------------------------
6. HUONG DAN DONG GOP (CONTRIBUTION)
--------------------------------------------------------------------------------

  Moi dong gop deu duoc hoan nghenh! De dong gop:

  Buoc 1. Fork repository va tao nhanh moi:

        git checkout -b feature/ten-tinh-nang

  Buoc 2. Commit thay doi voi thong diep ro rang:

        git commit -m "feat: mo ta ngan gon ve thay doi"

  Buoc 3. Push nhanh len va tao Pull Request:

        git push origin feature/ten-tinh-nang

  Luu y: Vui long dam bao ma cua ban chay duoc, truoc khi tao PR hay kiem tra
  ky luong build_index() va answer().

--------------------------------------------------------------------------------
7. NGUON CAM HUNG (ACKNOWLEDGEMENTS)
--------------------------------------------------------------------------------

  Do an duoc lay cam hung tu RAG-pipeline-visualizer cua vietnh1009:

      https://github.com/vietnh1009/RAG-pipeline-visualizer

  Ung dung giup xay dung, cau hinh va truc quan hoa tung buoc trong pipeline
  RAG - tu buoc load tai lieu PDF den buoc sinh cau tra loi bang LLM.

--------------------------------------------------------------------------------
8. GIAY PHEP (LICENSE)
--------------------------------------------------------------------------------

  Du an su dung cho muc dich hoc tap va nghien cuu.
  Moi nhan hieu thuoc ve chu so huu tuong ung.

================================================================================
