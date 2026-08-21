from base import BaseEmbedder
from config import EMBED_MODEL_HF
from embedders import register_embedder


@register_embedder("huggingface")
class HuggingFaceEmbedder(BaseEmbedder):
    """Strategy: HuggingFace local sentence-transformers model.

    Local + free, no API key. Downloads the model on first use.
    Best when you want full control and no network dependency.
    """

    def __init__(
        self,
        model: str = EMBED_MODEL_HF,
        **kwargs,
    ):
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError as e:
            raise ImportError(
                "huggingface embedder needs `pip install langchain-huggingface`"
            ) from e

        self.model = model
        self.client = HuggingFaceEmbeddings(model_name=model, **kwargs)

    def embed_documents(self, texts):
        return self.client.embed_documents(texts)

    def embed_query(self, text):
        return self.client.embed_query(text)