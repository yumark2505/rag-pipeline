import logging
import os

from base import BaseVectorStore
from config import PGVECTOR_URL
from vectorstores import register_vectorstore

logger = logging.getLogger(__name__)


@register_vectorstore("pgvector")
class PGVectorStore(BaseVectorStore):
    """Strategy: PGVector (PostgreSQL). Production-grade, SQL queries.

    Good for: real deployments, metadata SQL filtering, concurrent reads.
    Requires a running PostgreSQL with the pgvector extension:
      CREATE EXTENSION IF NOT EXISTS vector;
    """

    def __init__(
        self,
        embeddings,
        connection_string: str = PGVECTOR_URL,
        collection_name: str = "documents",
    ):
        self.embeddings = embeddings
        self.connection_string = connection_string
        self.collection_name = collection_name
        self.store = None

    def build(self, chunks):
        try:
            from langchain_community.vectorstores import PGVector
        except ImportError as e:
            raise ImportError(
                "pgvector store needs: pip install pgvector psycopg2-binary "
                "langchain-community"
            ) from e

        self.store = PGVector.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            connection_string=self.connection_string,
        )
        logger.info("Built pgvector index over %d chunks", len(chunks))
        return self.store

    def similarity_search_with_score(self, query, k):
        if self.store is None:
            raise RuntimeError("Vector store not built yet. Call build() first.")
        return self.store.similarity_search_with_score(query, k=k)

    def save(self, path=None):
        # PGVector is already persistent — nothing to write locally.
        return self.connection_string

    def load(self, path=None):
        try:
            from langchain_community.vectorstores import PGVector
        except ImportError as e:
            raise ImportError("pgvector store needs: pip install pgvector psycopg2-binary") from e

        self.store = PGVector(
            embedding_function=self.embeddings,
            collection_name=self.collection_name,
            connection_string=self.connection_string,
        )
        logger.info("Connected to %s", self.connection_string)
        return self.store