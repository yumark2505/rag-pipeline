import logging

from config import LOG_LEVEL

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
_configured = False

# Top-level logger names of this app (modules log via getLogger(__name__)).
_APP_LOGGERS = [
    "main",
    "pipeline",
    "loaders",
    "chunkers",
    "embedders",
    "vectorstores",
    "pre_retrieval",
    "retrieval",
    "post_retrieval",
    "prompts",
]


def setup_logging(level: str | None = None) -> logging.Logger:
    """Configure logging once for the whole app.

    Third-party libraries stay at WARNING so their noise doesn't drown
    out our own messages; every app logger above logs at `level`
    (default from config.LOG_LEVEL / RAG_LOG_LEVEL env var).
    """
    global _configured
    if not _configured:
        logging.basicConfig(format=_FORMAT, level=logging.WARNING)
        _configured = True
    resolved = (level or LOG_LEVEL).upper()
    for name in _APP_LOGGERS:
        logging.getLogger(name).setLevel(resolved)
    return logging.getLogger("pipeline")