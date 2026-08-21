import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from rag.pipeline import RAGPipeline

__all__ = ["RAGPipeline"]