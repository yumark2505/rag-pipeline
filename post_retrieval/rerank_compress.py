import json
import logging

from langchain_core.documents import Document

from base import BasePostRetrieval
from config import CONTEXT_MAX_CHARS, RERANK_TOP_K
from post_retrieval import register_post_retrieval
from post_retrieval.basic_format import BasicFormatStrategy

logger = logging.getLogger(__name__)


@register_post_retrieval("rerank")
class RerankCompressStrategy(BasicFormatStrategy):
    """Strategy: LLM reranking + context compression.

    Asks the chat LLM to order the retrieved chunks by relevance to the
    question, keeps the top-k, then fits them into the char budget.
    Falls back to the original order when the model's reply is unparsable.
    """

    RERANK_PROMPT = (
        "Rank these chunks by how relevant they are to the question. "
        "Return only a JSON list of indices ordered from most to least "
        'relevant, e.g. [3,1,2].\n\n'
        "Question: {question}\n\nChunks:\n{chunks}"
    )

    def __init__(
        self,
        llm,
        max_chars: int = CONTEXT_MAX_CHARS,
        top_k: int = RERANK_TOP_K,
        rerank_candidates: int = 8,
    ):
        super().__init__(max_chars=max_chars)
        self.llm = llm
        self.top_k = top_k
        self.rerank_candidates = rerank_candidates

    def _rerank(self, docs, query: str) -> list[Document]:
        candidates = docs[: self.rerank_candidates]  # keep the prompt small
        numbered = "\n\n".join(
            f"[{i}] {doc.page_content[:500]}"
            for i, doc in enumerate(candidates, start=1)
        )
        prompt = self.RERANK_PROMPT.format(question=query, chunks=numbered)
        ranked = []
        try:
            order = json.loads(self.llm.invoke(prompt).content.strip())
            if isinstance(order, list):
                ranked = [
                    candidates[i - 1] for i in order if 1 <= i <= len(candidates)
                ]
        except (json.JSONDecodeError, ValueError):
            logger.warning("Reranker returned unparsable output; keeping original order")
        # Keep every doc: ranked order first, then the rest as-is.
        seen = {id(d) for d in ranked}
        ordered = ranked + [d for d in docs if id(d) not in seen]
        logger.debug("Reranked %d docs -> top %d", len(docs), self.top_k)
        return ordered

    def process(self, docs, query: str) -> str:
        return super().process(self._rerank(docs, query)[: self.top_k], query)