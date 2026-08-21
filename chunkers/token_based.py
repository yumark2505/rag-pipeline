from langchain_text_splitters import TokenTextSplitter

from base import BaseChunker
from config import CHUNK_OVERLAP, CHUNK_SIZE
from chunkers import register_chunker


@register_chunker("token")
class TokenChunker(BaseChunker):
    """Strategy: Tiktoken-based.

    Splits by token count instead of characters, giving stable chunk
    sizes for models with token-context limits (e.g. embedding models).
    Requires the `tiktoken` package (bundled with OpenAI SDK).
    """

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        encoding_name: str = "cl100k_base",
        **kwargs,
    ):
        self.splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            encoding_name=encoding_name,
            **kwargs,
        )

    def split_documents(self, docs):
        splits = self.splitter.split_documents(docs)
        return splits

    def split_text(self, text: str):
        return self.splitter.split_text(text)