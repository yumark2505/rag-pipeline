import os
from typing import Literal

from dotenv import load_dotenv

load_dotenv()

# Ollama
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")

# Embedding models
EMBED_MODEL_OLLAMA = "nomic-embed-text"
EMBED_MODEL_OPENAI = "text-embedding-3-small"
EMBED_MODEL_HF = "sentence-transformers/all-MiniLM-L6-v2"

# OpenAI API key
OPENAI_API_KEY = None

# Chat models
CHAT_MODEL = "qwen2.5:7b"
CHAT_MODEL_OPENAI = "gpt-4o-mini"
CHAT_MODEL_HF = "Qwen/Qwen2.5-1.5B-Instruct"

# Loaders
PAPERS_DIR = "./papers"
PDF_GLOB = "**/*.pdf"
OPENDATALOADER_MODEL = "ds4sd/SmolDocling-256M-preview"

# Chunkers
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
BreakpointType = Literal["percentile", "standard_deviation", "interquartile", "gradient"]
SEMANTIC_BREAKPOINT_TYPE: BreakpointType = "percentile"
SEMANTIC_BREAKPOINT_AMOUNT = 85

# Embedders
EMBED_BATCH_SIZE = 64

# Vector stores
FAISS_INDEX_DIR = "./rag_pipeline/data/faiss"
CHROMA_DIR = "./rag_pipeline/data/chroma"
PGVECTOR_URL = os.environ.get("PGVECTOR_URL", "postgresql://postgres:postgres@localhost:5432/rag")

# Retrieval
RETRIEVER_K = 5
RETRIEVER_SCORE_THRESHOLD = 1.0
BM25_K = 5
HYBRID_WEIGHT_VECTOR = 0.5
HYBRID_WEIGHT_BM25 = 0.5
RERANK_TOP_K = 5

# Generation
GENERATION_TEMPERATURE = 0.0
CHAT_HISTORY_K = 6  # last N messages to keep for conversational prompt
CONTEXT_MAX_CHARS = 6000

# Logging: DEBUG shows retrieval internals, INFO shows stage/query summaries
LOG_LEVEL = os.environ.get("RAG_LOG_LEVEL", "INFO")