import logging
import os

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy

from rag_pipeline.base import BaseVectorStore
from rag_pipeline.config import FAISS_INDEX_DIR
from rag_pipeline.vectorstores import ensure_dir, register_vectorstore

logger = logging.getLogger(__name__)


@register_vectorstore("faiss")
class FaissStore(BaseVectorStore):
    """Strategy: FAISS. In-memory, fast on CPU, no external server.

    Simplest to run; persistence via local files.
    """

    # FAISS returns distances (lower = more similar).
    higher_is_better = False

    def __init__(self, embeddings, index_dir: str = FAISS_INDEX_DIR):
        self.embeddings = embeddings
        self.index_dir = index_dir
        self.store = None

    def build(self, chunks):
        self.store = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            distance_strategy=DistanceStrategy.COSINE,
        )
        logger.info("Built faiss index over %d chunks", len(chunks))
        return self.store

    def similarity_search_with_score(self, query, k):
        if self.store is None:
            raise RuntimeError("Vector store not built yet. Call build() first.")
        return self.store.similarity_search_with_score(query, k=k)

    def save(self, path=None):
        path = path or self.index_dir
        ensure_dir(path)
        self.store.save_local(path)
        return path

    def load(self, path=None):
        path = path or self.index_dir
        if not os.path.exists(path):
            raise FileNotFoundError(f"No FAISS index at {path}")
        self.store = FAISS.load_local(
            path,
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True,
        )
        logger.info("Loaded faiss index from %s", path)
        return self.store