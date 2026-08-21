import json
import operator

import ollama
from langchain_community.vectorstores.utils import DistanceStrategy

from rag.config import CHAT_MODEL, OLLAMA_HOST, RETRIEVER_K, RETRIEVER_SCORE_THRESHOLD


class PreRetrieval:
    """Stage 5: Rewrite the query, then decompose it into sub-queries."""

    def __init__(self, model: str = CHAT_MODEL, host: str = OLLAMA_HOST):
        self.client = ollama.Client(host=host)
        self.model = model

    def _ask(self, prompt: str) -> str:
        return self.client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0},
        )["message"]["content"].strip()

    def _rewrite(self, query: str) -> str:
        prompt = (
            "Rewrite this question into a clear, self-contained retrieval query. "
            "Keep the meaning and key terms. Output only the rewritten query.\n"
            f"Question: {query}"
        )
        return self._ask(prompt)

    def _decompose(self, query: str) -> list[str]:
        prompt = (
            "Split this question into simpler sub-questions, one per line, "
            "each covering a distinct aspect. If it cannot be split, "
            "output only the original question.\n"
            f"Question: {query}"
        )
        return [line.strip() for line in self._ask(prompt).splitlines() if line.strip()]

    def transform(self, query: str) -> list[str]:
        rewritten = self._rewrite(query)
        return self._decompose(rewritten)


class Retriever:
    """Stage 6: Fetch relevant chunks for each sub-query and dedupe."""

    def __init__(self, vectorstore, k: int = RETRIEVER_K, score_threshold: float = RETRIEVER_SCORE_THRESHOLD):
        self.vectorstore = vectorstore
        self.k = k
        self.score_threshold = score_threshold

    def _passes_threshold(self, score: float) -> bool:
        """True if the score is on the "relevant" side of the threshold.

        Score semantics depend on the distance strategy:
        - COSINE / EUCLIDEAN_DISTANCE return distance (lower = more similar).
        - MAX_INNER_PRODUCT / JACCARD return similarity (higher = more similar).
        """
        if self.vectorstore.distance_strategy in (
            DistanceStrategy.MAX_INNER_PRODUCT,
            DistanceStrategy.JACCARD,
        ):
            return score >= self.score_threshold
        return score <= self.score_threshold

    def retrieve(self, queries: list[str]):
        if self.vectorstore.store is None:
            raise RuntimeError("Vector store not built yet.")
        seen, docs = set(), []
        for query in queries:
            results = self.vectorstore.similarity_search(query, k=self.k)
            for doc, score in results:
                if not self._passes_threshold(score) or id(doc) in seen:
                    continue
                seen.add(id(doc))
                docs.append(doc)
        return docs


class PostRetrieval:
    """Stage 7: Rerank the chunks by relevance, then compress to fit the budget."""

    def __init__(self, model: str = CHAT_MODEL, host: str = OLLAMA_HOST, max_chars: int = 6000):
        self.client = ollama.Client(host=host)
        self.model = model
        self.max_chars = max_chars

    def rerank(self, docs, query: str):
        candidates = docs[:8]  # keep the rerank prompt small
        numbered = "\n\n".join(
            f"[{i}] {doc.page_content[:500]}" for i, doc in enumerate(candidates, start=1)
        )
        prompt = (
            "Rank these chunks by how relevant they are to the question. "
            "Return only a JSON list of indices ordered from most to least relevant, "
            'e.g. [3,1,2].\n\n'
            f"Question: {query}\n\nChunks:\n{numbered}"
        )
        ranked = []
        try:
            order = json.loads(self._ask(prompt))
            if isinstance(order, list):
                ranked = [
                    candidates[i - 1] for i in order if 1 <= i <= len(candidates)
                ]
        except (json.JSONDecodeError, ValueError):
            pass
        # Keep every doc: ranked order first, then the rest in original order.
        seen = {id(d) for d in ranked}
        return ranked + [d for d in docs if id(d) not in seen]

    def _ask(self, prompt: str) -> str:
        return self.client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0},
        )["message"]["content"].strip()

    def compress(self, docs, query: str):
        seen, kept, total = set(), [], 0
        for doc in self.rerank(docs, query):
            if id(doc) in seen:
                continue
            seen.add(id(doc))
            if total + len(doc.page_content) > self.max_chars:
                break
            total += len(doc.page_content)
            kept.append(doc)
        return kept

    def format(self, docs):
        formatted = []
        for i, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "?")
            formatted.append(f"[{i}] (source: {source}, page: {page})\n{doc.page_content}")
        return "\n\n".join(formatted)

    def process(self, docs, query: str) -> str:
        return self.format(self.compress(docs, query))