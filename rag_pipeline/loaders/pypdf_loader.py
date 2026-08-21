import logging

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from rag_pipeline.base import BaseLoader
from rag_pipeline.config import PDF_GLOB
from rag_pipeline.loaders import register_loader

import glob as _glob
import os

logger = logging.getLogger(__name__)


@register_loader("pypdf")
class PyPDFLoaderStrategy(BaseLoader):
    """Strategy: pypdf. Fast, uses the text layer, no extra deps.

    Good for: digital-born PDFs with embedded selectable text.
    Weakness: no OCR — scanned pages yield no text.
    """

    def __init__(self, path: str, glob: str = PDF_GLOB, **kwargs):
        super().__init__(path, glob)
        self.kwargs = kwargs

    def load(self):
        docs = []
        files = _glob.glob(os.path.join(self.path, self.glob), recursive=True)
        for filepath in sorted(files):
            loader = PyPDFLoader(filepath)
            page_docs = loader.load()
            for d in page_docs:
                d.metadata["loader"] = "pypdf"
            docs.extend(page_docs)
        logger.info("Loaded %d page-documents from %d files", len(docs), len(files))
        return docs