import os
from typing import Literal

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")

EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "qwen2.5:7b"

PAPERS_DIR = "./papers"
DOC_GLOB = "**/*.pdf"

EMBED_BATCH_SIZE = 64

BreakpointType = Literal["percentile", "standard_deviation", "interquartile", "gradient"]
CHUNK_BREAKPOINT_TYPE: BreakpointType = "percentile"
CHUNK_BREAKPOINT_AMOUNT = 85

RETRIEVER_K = 5
RETRIEVER_SCORE_THRESHOLD = 1.0

FAISS_INDEX_DIR = "./rag/data/faiss"

GENERATION_TEMPERATURE = 0.0