import os

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy

from rag.config import FAISS_INDEX_DIR, RETRIEVER_K, RETRIEVER_SCORE_THRESHOLD


class VectorStore:
    """Stage 4: Build, persist, and query the FAISS vector index."""

    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.store = None
        self.distance_strategy = DistanceStrategy.COSINE

    def build(self, chunks):
        self.store = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            distance_strategy=self.distance_strategy,
        )
        return self.store

    def save(self, path: str = FAISS_INDEX_DIR):
        if self.store is None:
            raise RuntimeError("Vector store not built yet. Call build() first.")
        os.makedirs(path, exist_ok=True)
        self.store.save_local(path)

    def load(self, path: str = FAISS_INDEX_DIR):
        if not os.path.exists(path):
            return False
        self.store = FAISS.load_local(
            path,
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True,
        )
        return True

    def similarity_search(self, query, k: int = RETRIEVER_K):
        if self.store is None:
            raise RuntimeError("Vector store not built yet. Call build() first.")
        return self.store.similarity_search_with_score(query, k=k)

    def get_retriever(self, k: int = RETRIEVER_K, score_threshold: float = RETRIEVER_SCORE_THRESHOLD):
        if self.store is None:
            raise RuntimeError("Vector store not built yet. Call build() first.")
        return self.store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": k, "score_threshold": score_threshold},
        )