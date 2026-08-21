import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter

from base import BaseChunker
from config import CHUNK_OVERLAP, CHUNK_SIZE
from chunkers import register_chunker

logger = logging.getLogger(__name__)


@register_chunker("recursive")
class RecursiveChunker(BaseChunker):
    """Strategy: RecursiveCharacterTextSplitter.

    Splits on a hierarchy of separators (headings -> blank lines ->
    newlines -> spaces). Deterministic, cheap, no model needed.
    """

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        **kwargs,
    ):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
            strip_whitespace=True,
            **kwargs,
        )

    def split_documents(self, docs):
        splits = self.splitter.split_documents(docs)
        logger.info("Split into %d chunks", len(splits))
        return splits

    def split_text(self, text: str):
        return self.splitter.split_text(text)