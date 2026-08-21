from base import BaseEmbedder
from config import EMBED_MODEL_OPENAI, OPENAI_API_KEY
from embedders import register_embedder


@register_embedder("openai")
class OpenAIEmbedder(BaseEmbedder):
    """Strategy: OpenAI embedding API (e.g. text-embedding-3-small).

    High quality, hosted. Set OPENAI_API_KEY in config or environment.
    """

    def __init__(
        self,
        model: str = EMBED_MODEL_OPENAI,
        api_key: str | None = OPENAI_API_KEY,
        **kwargs,
    ):
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError as e:
            raise ImportError("openai embedder needs `pip install langchain-openai`") from e

        if api_key is None:
            import os

            api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set for the openai embedder.")

        from pydantic import SecretStr

        self.model = model
        self.client = OpenAIEmbeddings(model=model, api_key=SecretStr(api_key), **kwargs)

    def embed_documents(self, texts):
        return self.client.embed_documents(texts)

    def embed_query(self, text):
        return self.client.embed_query(text)