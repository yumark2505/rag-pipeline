import os

from rag_pipeline.base import make_registry

vectorstore_registry, register_vectorstore = make_registry("vectorstore")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)