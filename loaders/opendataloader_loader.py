import glob as _glob
import logging
import os
from pathlib import Path
from typing import List, Union

from base import BaseLoader
from config import PDF_GLOB
from loaders import register_loader

logger = logging.getLogger(__name__)


@register_loader("opendataloader")
class OpenDataLoaderStrategy(BaseLoader):
    """Strategy: opendataloader. #1 benchmark (0.90), bounding box, no GPU.

    Good for: top-quality extraction incl. tables/bboxes/citations.
    Requires JVM: each call spawns a Java process, so batch all files
    in one call (slow if called repeatedly).

    Install: pip install -U langchain-opendataloader-pdf opendataloader-pdf
    """

    def __init__(
        self,
        path: str,
        glob: str = PDF_GLOB,
        format: str = "markdown",
        quiet: bool = True,
        **kwargs,
    ):
        super().__init__(path, glob)
        self.format = format
        self.quiet = quiet
        self.kwargs = kwargs

    def load(self):
        try:
            from langchain_opendataloader_pdf import OpenDataLoaderPDFLoader
        except ImportError as e:
            raise ImportError(
                "opendataloader loader needs extra packages. Run:\n"
                "  pip install -U langchain-opendataloader-pdf opendataloader-pdf"
            ) from e

        file_path = self._collect_files()
        loader = OpenDataLoaderPDFLoader(
            file_path=file_path,
            format=self.format,
            quiet=self.quiet,
            **self.kwargs,
        )
        docs = loader.load()
        for d in docs:
            d.metadata["loader"] = "opendataloader"
        logger.info("Loaded %d documents (format=%s, files=%d)", len(docs), self.format, len(file_path))
        return docs

    def _collect_files(self) -> List[Union[str, Path]]:
        files = _glob.glob(os.path.join(self.path, self.glob), recursive=True)
        if not files:
            raise FileNotFoundError(f"No PDFs found in {self.path}")
        return sorted(files)