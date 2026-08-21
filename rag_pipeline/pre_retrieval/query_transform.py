from rag_pipeline.base import BasePreRetrieval
from rag_pipeline.pre_retrieval import register_pre_retrieval


@register_pre_retrieval("query_transform")
class QueryTransformStrategy(BasePreRetrieval):
    """Strategy: LLM query rewriting + decomposition.

    Rewrites the question into a clear, self-contained retrieval query,
    then splits it into sub-questions covering distinct aspects. The
    pipeline retrieves once per sub-query and merges the results.
    """

    REWRITE_PROMPT = (
        "Rewrite this question into a clear, self-contained retrieval query. "
        "Keep the meaning and key terms. Output only the rewritten query.\n"
        "Question: {question}"
    )
    DECOMPOSE_PROMPT = (
        "Split this question into simpler sub-questions, one per line, "
        "each covering a distinct aspect. If it cannot be split, "
        "output only the original question.\n"
        "Question: {question}"
    )

    def __init__(self, llm, max_sub_queries: int = 3):
        self.llm = llm
        self.max_sub_queries = max_sub_queries

    def _ask(self, prompt: str) -> str:
        return self.llm.invoke(prompt).content.strip()

    def transform(self, query: str) -> list[str]:
        rewritten = self._ask(self.REWRITE_PROMPT.format(question=query))
        lines = self._ask(self.DECOMPOSE_PROMPT.format(question=rewritten)).splitlines()
        queries = [line.strip().lstrip("- ").strip() for line in lines if line.strip()]
        return queries[: self.max_sub_queries] or [rewritten]