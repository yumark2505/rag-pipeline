"""Base classes shared across all stage strategies."""


from langchain_core.embeddings import Embeddings


class Registry:
    """Generic name -> class factory shared by all stage registries."""

    def __init__(self, kind: str):
        self._kind = kind
        self._items = {}

    def register(self, name: str, cls):
        self._items[name] = cls
        return cls

    def get(self, name: str, **kwargs):
        if name not in self._items:
            raise ValueError(
                f"Unknown {self._kind} '{name}'. Available: {list(self._items.keys())}"
            )
        return self._items[name](**kwargs)


def make_registry(kind: str):
    """Create a (registry, register_decorator) pair for one stage kind."""
    registry = Registry(kind)

    def register(name: str):
        def decorator(cls):
            registry.register(name, cls)
            return cls

        return decorator

    return registry, register


class BaseLoader:
    """Stage 1 interface: load raw documents from disk."""

    def __init__(self, path: str, glob: str):
        self.path = path
        self.glob = glob

    def load(self):
        raise NotImplementedError


class BaseChunker:
    """Stage 2 interface: split documents into chunks."""

    def split_documents(self, docs):
        raise NotImplementedError

    def split_text(self, text: str):
        raise NotImplementedError


class BaseEmbedder(Embeddings):
    """Stage 3 interface: embed texts into vectors.

    Subclasses langchain_core.embeddings.Embeddings so LangChain vector
    stores (FAISS/Chroma/PGVector) recognize it via isinstance checks.
    """

    def embed_documents(self, texts):
        raise NotImplementedError

    def embed_query(self, text):
        raise NotImplementedError


class BaseVectorStore:
    """Stage 4 interface: persist and query vectors."""

    def build(self, chunks):
        raise NotImplementedError

    def similarity_search_with_score(self, query, k):
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError

    def load(self, path):
        raise NotImplementedError


class BaseRetriever:
    """Stage 6 interface: retrieve chunks for a query."""

    def retrieve(self, query):
        raise NotImplementedError


class BasePreRetrieval:
    """Stage 5 interface: transform a question into retrieval queries."""

    def transform(self, query: str) -> list[str]:
        raise NotImplementedError


class BasePostRetrieval:
    """Stage 7 interface: turn retrieved chunks into final context text."""

    def process(self, docs, query: str) -> str:
        raise NotImplementedError