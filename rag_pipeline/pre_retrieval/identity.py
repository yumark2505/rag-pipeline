from rag_pipeline.base import BasePreRetrieval
from rag_pipeline.pre_retrieval import register_pre_retrieval


@register_pre_retrieval("identity")
class IdentityPreRetrieval(BasePreRetrieval):
    """Strategy: no transformation; the query is used as-is."""

    def __init__(self, llm=None):
        self.llm = llm  # unused; kept for a uniform strategy interface

    def transform(self, query: str) -> list[str]:
        return [query.strip()]