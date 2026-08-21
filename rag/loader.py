from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader

from rag.config import DOC_GLOB, PAPERS_DIR


class PDFLoader:
    """Stage 1: Load documents (PDF, TXT, MD, ...) from a directory."""

    def __init__(
        self,
        path: str = PAPERS_DIR,
        glob: str = DOC_GLOB,
        show_progress: bool = True,
        use_multithreading: bool = True,
    ):
        self.path = path
        self.glob = glob
        self.show_progress = show_progress
        self.use_multithreading = use_multithreading

    def load(self):
        loader = DirectoryLoader(
            path=self.path,
            glob=self.glob,
            loader_cls=UnstructuredFileLoader,
            show_progress=self.show_progress,
            use_multithreading=self.use_multithreading,
        )
        docs = loader.load()
        return docs