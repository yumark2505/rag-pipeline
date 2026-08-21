import logging

from rag_pipeline.config import LOG_LEVEL

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
_configured = False


def setup_logging(level: str | None = None) -> logging.Logger:
    """Configure logging once for the whole package.

    Third-party libraries stay at WARNING so their noise doesn't drown
    out our own messages; everything under `rag_pipeline.*` logs at
    `level` (default from config.LOG_LEVEL / RAG_LOG_LEVEL env var).
    """
    global _configured
    if not _configured:
        logging.basicConfig(format=_FORMAT, level=logging.WARNING)
        _configured = True
    logger = logging.getLogger("rag_pipeline")
    logger.setLevel((level or LOG_LEVEL).upper())
    return logger