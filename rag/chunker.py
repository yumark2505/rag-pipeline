from langchain_experimental.text_splitter import SemanticChunker

from rag.config import BreakpointType, CHUNK_BREAKPOINT_AMOUNT, CHUNK_BREAKPOINT_TYPE


class SemanticChunkerStage:
    """Stage 2: Split documents into semantically coherent chunks."""

    def __init__(
        self,
        embeddings,
        breakpoint_threshold_type: BreakpointType = CHUNK_BREAKPOINT_TYPE,
        breakpoint_threshold_amount: float = CHUNK_BREAKPOINT_AMOUNT,
    ):
        self.splitter = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type=breakpoint_threshold_type,
            breakpoint_threshold_amount=breakpoint_threshold_amount,
        )

    def split(self, docs):
        return self.splitter.split_documents(docs)