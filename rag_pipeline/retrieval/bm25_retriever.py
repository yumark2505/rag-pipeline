import logging
import re

from rank_bm25 import BM25Okapi

from rag_pipeline.base import BaseRetriever
from rag_pipeline.config import BM25_K
from rag_pipeline.retrieval import register_retriever

logger = logging.getLogger(__name__)


@register_retriever("bm25")
class BM25Retriever(BaseRetriever):
    """Strategy: BM25 (classic keyword scoring).

    Good for exact-term / rare-token queries, acronyms, code.
    Weakness: no semantic understanding of synonyms.
    """

    def __init__(
        self,
        chunks=None,
        k: int = BM25_K,
    ):
        self.k = k
        self.chunks = chunks or []
        self._tokenized = None
        self._bm25 = None

    def index(self, chunks):
        self.chunks = chunks
        self._tokenized = [self._tokenize(c.page_content) for c in chunks]
        self._bm25 = BM25Okapi(self._tokenized)
        return self

    @staticmethod
    def _tokenize(text: str):
        return re.findall(r"\w+", text.lower())

    def retrieve(self, query):
        if self._bm25 is None:
            raise RuntimeError("BM25 not indexed. Call index(chunks) first.")
        scores = self._bm25.get_scores(self._tokenize(query))
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: self.k]
        docs = [self.chunks[i] for i in top_idx if scores[i] > 0]
        logger.debug("Retrieved %d chunks (k=%d)", len(docs), self.k)
        return docs