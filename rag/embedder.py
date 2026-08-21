import ollama
from langchain_core.embeddings import Embeddings

from rag.config import EMBED_BATCH_SIZE, EMBED_MODEL, OLLAMA_HOST


class OllamaEmbedder(Embeddings):
    """Stage 3: Embedding provider backed by Ollama.

    Batches embed calls to avoid crashing the llama-server runner,
    which happens when a huge list of texts is sent in one request.
    """

    def __init__(
        self,
        model: str = EMBED_MODEL,
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