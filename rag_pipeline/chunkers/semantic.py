import logging

from langchain_experimental.text_splitter import SemanticChunker

from rag_pipeline.base import BaseChunker
from rag_pipeline.config import (
    BreakpointType,
    SEMANTIC_BREAKPOINT_AMOUNT,
    SEMANTIC_BREAKPOINT_TYPE,
)
from rag_pipeline.chunkers import register_chunker

logger = logging.getLogger(__name__)


@register_chunker("semantic")
class SemanticChunkerStrategy(BaseChunker):
    """Strategy: Cosine similarity breakpoints.

    Embeds sentences and cuts where consecutive embeddings are
    dissimilar. Best-quality chunks but slowest and requires an
    embedding model at split time.
    """

    def __init__(
        self,
        embeddings=None,
        breakpoint_threshold_type: BreakpointType = SEMANTIC_BREAKPOINT_TYPE,
        breakpoint_threshold_amount: float = SEMANTIC_BREAKPOINT_AMOUNT,
        **kwargs,
    ):
        if embeddings is None:
            raise ValueError(
                "SemanticChunker needs an embedding model. Pass `embeddings=`."
            )
        self.splitter = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type=breakpoint_threshold_type,
            breakpoint_threshold_amount=breakpoint_threshold_amount,
            **kwargs,
        )

    def split_documents(self, docs):
        splits = self.splitter.split_documents(docs)
        logger.info(
            "Split into %d chunks (threshold=%s)",
            len(splits),
            self.splitter.breakpoint_threshold_amount,
        )
        return splits

    def split_text(self, text: str):
        return self.splitter.split_text(text)