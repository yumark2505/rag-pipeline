import logging

from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader

from base import BaseLoader
from config import PDF_GLOB
from loaders import register_loader

logger = logging.getLogger(__name__)


@register_loader("unstructured")
class UnstructuredLoaderStrategy(BaseLoader):
    """Strategy: unstructured. Best for OCR + tables + images.

    Good for: scanned PDFs, tables, mixed-layout documents.
    Slower than pypdf because it may run OCR / layout analysis.
    """

    def __init__(
        self,
        path: str,
        glob: str = PDF_GLOB,
        mode: str = "elements",
        strategy: str = "auto",
        infer_table_structure: bool = True,
        use_multithreading: bool = True,
        **kwargs,
    ):
        super().__init__(path, glob)
        self.mode = mode
        self.strategy = strategy
        self.infer_table_structure = infer_table_structure
        self.use_multithreading = use_multithreading
        self.kwargs = kwargs

    def load(self):
        loader = DirectoryLoader(
            path=self.path,
            glob=self.glob,
            loader_cls=UnstructuredFileLoader,
            loader_kwargs={
                "mode": self.mode,
                "strategy": self.strategy,
            },
            show_progress=True,
            use_multithreading=self.use_multithreading,
        )
        docs = loader.load()
        for d in docs:
            d.metadata["loader"] = "unstructured"
        logger.info("Loaded %d elements (mode=%s, strategy=%s)", len(docs), self.mode, self.strategy)
        return docs