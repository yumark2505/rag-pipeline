import sys

from logging_setup import setup_logging
from pipeline import RAGPipelineAdvanced


def _choose(label, options, default):
    print(f"\n[{label}] Choose one:")
    for i, opt in enumerate(options, start=1):
        marker = " (default)" if opt == default else ""
        print(f"  {i}. {opt}{marker}")
    choice = input(f"  Enter number or name [{default}]: ").strip().lower()
    if choice.isdigit():
        idx = int(choice)
        if 1 <= idx <= len(options):
            return options[idx - 1]
    if choice in options:
        return choice
    return default


def main():
    setup_logging()
    if "--help" in sys.argv or "-h" in sys.argv:
        print(
            "Interactive strategy chooser. Run `python pipeline.py --help` "
            "for the non-interactive CLI."
        )
        return

    print("=== Pluggable RAG Pipeline Setup ===")

    loader = _choose("Stage 1 - PDF Loader", ["pypdf", "unstructured", "opendataloader"], "pypdf")
    chunker = _choose("Stage 2 - Chunker", ["recursive", "token", "semantic"], "semantic")
    embedder = _choose("Stage 3 - Embedder", ["ollama", "openai", "huggingface"], "ollama")
    vectorstore = _choose("Stage 4 - Vector Store", ["faiss", "chroma", "pgvector"], "faiss")
    pre_retrieval = _choose("Stage 5 - Pre-Retrieval", ["identity", "query_transform"], "query_transform")
    retriever = _choose("Stage 6 - Retriever", ["vector", "bm25", "hybrid"], "vector")
    post_retrieval = _choose("Stage 7 - Post-Retrieval", ["basic", "rerank"], "rerank")
    prompt = _choose("Stage 8 - Prompt", ["basic", "conversational"], "basic")

    pipeline = RAGPipelineAdvanced(
        loader=loader,
        chunker=chunker,
        embedder=embedder,
        vectorstore=vectorstore,
        retriever=retriever,
        prompt=prompt,
        pre_retrieval=pre_retrieval,
        post_retrieval=post_retrieval,
    )
    pipeline.run_cli()


if __name__ == "__main__":
    main()