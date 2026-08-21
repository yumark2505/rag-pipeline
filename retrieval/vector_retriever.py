import logging

from base import BaseRetriever
from config import RETRIEVER_K, RETRIEVER_SCORE_THRESHOLD
from retrieval import register_retriever

logger = logging.getLogger(__name__)


@register_retriever("vector")
class VectorRetriever(BaseRetriever):
    """Strategy: pure dense vector search (embedding similarity).

    Good for semantic matches (paraphrases, conceptual queries).
    Weakness: keyword/rare-token matches can be missed.
    """

    def __init__(
        self,
        vectorstore,
        k: int = RETRIEVER_K,
        score_threshold: float = RETRIEVER_SCORE_THRESHOLD,
    ):
        self.vectorstore = vectorstore
        self.k = k
        self.score_threshold = score_threshold

    def _passes_threshold(self, score: float) -> bool:
        """True if the score is on the "relevant" side of the threshold.

        FAISS returns distances (lower = more similar); stores that return
        similarity scores declare higher_is_better = True.
        """
        if getattr(self.vectorstore, "higher_is_better", False):
            return score >= self.score_threshold
        return score <= self.score_threshold

    def retrieve(self, query):
        if self.vectorstore.store is None:
            raise RuntimeError("Vector store not built yet.")
        results = self.vectorstore.similarity_search_with_score(query, k=self.k)
        docs = [doc for doc, score in results if self._passes_threshold(score)]
        logger.debug("Retrieved %d/%d chunks (k=%d, threshold=%s)", len(docs), len(results), self.k, self.score_threshold)
        return docs