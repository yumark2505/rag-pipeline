from rag_pipeline.base import BasePostRetrieval
from rag_pipeline.config import CONTEXT_MAX_CHARS
from rag_pipeline.post_retrieval import register_post_retrieval


@register_post_retrieval("basic")
class BasicFormatStrategy(BasePostRetrieval):
    """Strategy: format chunks with source labels, truncated by char budget."""

    def __init__(self, llm=None, max_chars: int = CONTEXT_MAX_CHARS):
        self.llm = llm  # unused; kept for a uniform strategy interface
        self.max_chars = max_chars

    def process(self, docs, query: str) -> str:
        formatted, total = [], 0
        for i, doc in enumerate(docs, start=1):
            content = doc.page_content
            if total + len(content) > self.max_chars:
                break
            total += len(content)
            formatted.append(f"{self._label(i, doc)}\n{content}")
        return "\n\n".join(formatted)

    @staticmethod
    def _label(index: int, doc) -> str:
        meta = doc.metadata
        source = meta.get("source", "unknown")
        page = meta.get("page", meta.get("page_number", "?"))
        return f"[{index}] (source: {source}, page: {page})"