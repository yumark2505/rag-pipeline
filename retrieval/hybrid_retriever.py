import logging

from base import BaseRetriever
from config import (
    HYBRID_WEIGHT_BM25,
    HYBRID_WEIGHT_VECTOR,
    RETRIEVER_K,
    RETRIEVER_SCORE_THRESHOLD,
)
from retrieval import register_retriever

logger = logging.getLogger(__name__)


@register_retriever("hybrid")
class HybridRetriever(BaseRetriever):
    """Strategy: hybrid = BM25 (lexical) + vector (semantic) fusion.

    Combines both worlds: keyword precision + semantic recall.
    Scores are min-max normalized then fused with a weighted sum.
    """

    def __init__(
        self,
        vectorstore,
        chunks=None,
        k: int = RETRIEVER_K,
        score_threshold: float = RETRIEVER_SCORE_THRESHOLD,
        weight_vector: float = HYBRID_WEIGHT_VECTOR,
        weight_bm25: float = HYBRID_WEIGHT_BM25,
    ):
        from retrieval.bm25_retriever import BM25Retriever

        self.vectorstore = vectorstore
        self.chunks = chunks or []
        self.k = k
        self.score_threshold = score_threshold
        self.weight_vector = weight_vector
        self.weight_bm25 = weight_bm25
        self.bm25 = BM25Retriever(chunks=chunks, k=k)

    def index(self, chunks):
        self.chunks = chunks
        self.bm25.index(chunks)
        return self

    @staticmethod
    def _minmax_normalize(scores):
        if not scores:
            return [0.0] * len(scores)
        lo, hi = min(scores), max(scores)
        if hi == lo:
            return [1.0 if s > 0 else 0.0 for s in scores]
        return [(s - lo) / (hi - lo) for s in scores]

    def retrieve(self, query):
        if not self.chunks:
            raise RuntimeError("Hybrid not indexed. Call index(chunks) first.")
        if self.vectorstore.store is None:
            raise RuntimeError("Vector store not built yet.")

        # Key everything by chunk position. FAISS's docstore returns fresh
        # Document copies, so object identity (id()) cannot be used to map
        # vector results back to the indexed chunks.
        pos_by_content = {c.page_content: i for i, c in enumerate(self.chunks)}

        # Vector scores (cosine distance -> invert to similarity)
        vec_results = self.vectorstore.similarity_search_with_score(query, k=self.k)
        vec_scores = {pos_by_content[d.page_content]: 1.0 - s for d, s in vec_results}

        # BM25 scores
        bm25_scores = self.bm25._bm25.get_scores(self.bm25._tokenize(query))
        bm25_map = dict(enumerate(bm25_scores))

        # Normalize both to [0,1]
        vec_norm = dict(zip(vec_scores.keys(), self._minmax_normalize(list(vec_scores.values()))))
        bm25_norm = dict(zip(bm25_map.keys(), self._minmax_normalize(list(bm25_map.values()))))

        # Fuse
        all_pos = set(vec_scores) | set(bm25_map)
        fused = {
            pos: (
                self.weight_vector * vec_norm.get(pos, 0.0)
                + self.weight_bm25 * bm25_norm.get(pos, 0.0)
            )
            for pos in all_pos
        }

        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[: self.k]
        docs = [self.chunks[pos] for pos, score in ranked if score > 0]
        logger.debug(
            "Retrieved %d chunks (k=%d, w_vec=%s, w_bm25=%s)",
            len(docs), self.k, self.weight_vector, self.weight_bm25,
        )
        return docs