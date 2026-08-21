import ollama

from rag_pipeline.base import BaseEmbedder
from rag_pipeline.config import EMBED_BATCH_SIZE, EMBED_MODEL_OLLAMA, OLLAMA_HOST
from rag_pipeline.embedders import register_embedder


@register_embedder("ollama")
class OllamaEmbedder(BaseEmbedder):
    """Strategy: Ollama local embedding (e.g. nomic-embed-text).

    Free, local, no API key. Batches embed calls to avoid crashing the
    llama-server runner with huge single requests.
    """

    def __init__(
        self,
        model: str = EMBED_MODEL_OLLAMA,
        host: str = OLLAMA_HOST,
        batch_size: int = EMBED_BATCH_SIZE,
    ):
        self.model = model
        self.client = ollama.Client(host=host)
        self.batch_size = batch_size

    def embed_documents(self, texts):
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            result = self.client.embed(model=self.model, input=batch)
            embeddings.extend(result["embeddings"])
        return embeddings

    def embed_query(self, text):
        result = self.client.embed(model=self.model, input=[text])
        return result["embeddings"][0]