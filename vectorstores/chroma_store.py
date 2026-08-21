import logging
import os

from langchain_chroma import Chroma
from langchain_core.documents import Document

from base import BaseVectorStore
from config import CHROMA_DIR
from vectorstores import ensure_dir, register_vectorstore

logger = logging.getLogger(__name__)


@register_vectorstore("chroma")
class ChromaStore(BaseVectorStore):
    """Strategy: Chroma. Persistent, built-in metadata filtering.

    Good for larger corpora + filtering by metadata (source, page).
    """

    def __init__(self, embeddings, persist_dir: str = CHROMA_DIR):
        self.embeddings = embeddings
        self.persist_dir = persist_dir
        self.store = None

    def build(self, chunks):
        self.store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_dir,
        )
        logger.info("Built chroma index over %d chunks at %s", len(chunks), self.persist_dir)
        return self.store

    def similarity_search_with_score(self, query, k):
        if self.store is None:
            raise RuntimeError("Vector store not built yet. Call build() first.")
        return self.store.similarity_search_with_score(query, k=k)

    def save(self, path=None):
        path = path or self.persist_dir
        ensure_dir(path)
        if hasattr(self.store, "persist"):
            self.store.persist()
        logger.debug("Persisted to %s", path)
        return path

    def load(self, path=None):
        path = path or self.persist_dir
        if not os.path.exists(path):
            raise FileNotFoundError(f"No Chroma collection at {path}")
        self.store = Chroma(
            embedding_function=self.embeddings,
            persist_directory=path,
        )
        logger.info("Loaded chroma collection from %s", path)
        return self.store