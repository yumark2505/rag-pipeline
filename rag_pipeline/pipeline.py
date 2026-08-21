import logging
import time

import ollama

from rag_pipeline.config import (
    CHAT_MODEL,
    CHAT_MODEL_HF,
    CHAT_MODEL_OPENAI,
    GENERATION_TEMPERATURE,
    OLLAMA_HOST,
    OPENAI_API_KEY,
)
from rag_pipeline.loaders import loader_registry
from rag_pipeline.chunkers import chunker_registry
from rag_pipeline.embedders import embedder_registry
from rag_pipeline.vectorstores import vectorstore_registry
from rag_pipeline.retrieval import retriever_registry
from rag_pipeline.prompts import prompt_registry
from rag_pipeline.logging_setup import setup_logging

logger = logging.getLogger(__name__)

# Import strategy modules so their @register_* decorators run.
from rag_pipeline.loaders import (  # noqa: F401
    opendataloader_loader,
    pypdf_loader,
    unstructured_loader,
)
from rag_pipeline.chunkers import recursive, semantic, token_based  # noqa: F401
from rag_pipeline.embedders import (  # noqa: F401
    huggingface_embedder,
    ollama_embedder,
    openai_embedder,
)
from rag_pipeline.vectorstores import (  # noqa: F401
    chroma_store,
    faiss_store,
    pgvector_store,
)
from rag_pipeline.retrieval import (  # noqa: F401
    bm25_retriever,
    hybrid_retriever,
    vector_retriever,
)
from rag_pipeline.prompts import basic, conversational  # noqa: F401
from rag_pipeline.pre_retrieval import pre_retrieval_registry
from rag_pipeline.post_retrieval import post_retrieval_registry

# Import strategy modules so their @register_* decorators run.
from rag_pipeline.pre_retrieval import (  # noqa: F401
    identity,
    query_transform,
)
from rag_pipeline.post_retrieval import (  # noqa: F401
    basic_format,
    rerank_compress,
)


class RAGPipelineAdvanced:
    """Pluggable RAG pipeline.

    Every stage picks a strategy by name at construction time:
      loader, chunker, embedder, vectorstore, retriever, prompt.

    Defaults mirror the original pipeline (ollama/pypdf/semantic/faiss/
    vector/basic) so it runs out of the box.
    """

    def __init__(
        self,
        loader: str = "pypdf",
        chunker: str = "semantic",
        embedder: str = "ollama",
        vectorstore: str = "faiss",
        retriever: str = "vector",
        prompt: str = "basic",
        pre_retrieval: str = "query_transform",
        post_retrieval: str = "rerank",
        chat_model: str = CHAT_MODEL,
        host: str = OLLAMA_HOST,
        papers_dir: str = "./papers",
        pdf_glob: str = "**/*.pdf",
        **stage_kwargs,
    ):
        self.host = host
        self.chat_model = chat_model

        self.loader_name = loader
        self.chunker_name = chunker
        self.embedder_name = embedder
        self.vectorstore_name = vectorstore
        self.retriever_name = retriever
        self.prompt_name = prompt
        self.pre_retrieval_name = pre_retrieval
        self.post_retrieval_name = post_retrieval

        # Stage 1: Loader
        self.loader = loader_registry.get(
            loader, path=papers_dir, glob=pdf_glob
        )

        # Stage 3: Embedder (needed by chunker/vectorstore)
        self.embeddings = embedder_registry.get(embedder)

        # Stage 2: Chunker
        chunker_kwargs = {}
        if chunker == "semantic":
            chunker_kwargs["embeddings"] = self.embeddings
        self.chunker = chunker_registry.get(chunker, **chunker_kwargs)

        # Stage 4: Vector store
        self.vectorstore = vectorstore_registry.get(
            vectorstore, embeddings=self.embeddings
        )

        # Stage 6: Retriever (kwargs depend on the retriever type)
        retriever_kwargs = {}
        if retriever in ("vector", "hybrid"):
            retriever_kwargs["vectorstore"] = self.vectorstore
        self.retriever = retriever_registry.get(retriever, **retriever_kwargs)

        # Stage 8: Prompt
        self.prompt_strategy = prompt_registry.get(prompt)

        # Generation LLM + output chain (built once, reused per question)
        self._init_llm()
        self._init_chain()

        # Stages 5 & 7: pre/post retrieval (share the generation LLM)
        self.pre_retrieval = pre_retrieval_registry.get(
            pre_retrieval, llm=self.llm
        )
        self.post_retrieval = post_retrieval_registry.get(
            post_retrieval, llm=self.llm
        )

    def _init_chain(self):
        from langchain_core.output_parsers import StrOutputParser

        template = self.prompt_strategy.build()
        if self.prompt_name == "conversational":
            self.chain = template | self.llm | StrOutputParser()
        else:
            from langchain_core.runnables import RunnablePassthrough

            self.chain = (
                {
                    "context": RunnablePassthrough(),
                    "question": RunnablePassthrough(),
                }
                | template
                | self.llm
                | StrOutputParser()
            )

    def _init_llm(self):
        """Pick the chat LLM to match the embedder's provider.

        ollama      -> ChatOllama   (local, same as embedder source)
        openai      -> ChatOpenAI   (hosted, same as embedder source)
        huggingface -> ChatHuggingFace wrapping a local transformers pipeline
        """
        if self.embedder_name == "ollama":
            from langchain_ollama import ChatOllama

            self.llm = ChatOllama(
                model=self.chat_model,
                base_url=self.host,
                temperature=GENERATION_TEMPERATURE,
            )
        elif self.embedder_name == "openai":
            from langchain_openai import ChatOpenAI

            llm_kwargs = {}
            if OPENAI_API_KEY:
                from pydantic import SecretStr

                llm_kwargs["api_key"] = SecretStr(OPENAI_API_KEY)
            self.llm = ChatOpenAI(
                model=CHAT_MODEL_OPENAI,
                temperature=GENERATION_TEMPERATURE,
                **llm_kwargs,
            )
        elif self.embedder_name == "huggingface":
            from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                pipeline as hf_pipeline,
            )

            import torch

            model_id = CHAT_MODEL_HF
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
            if torch.cuda.is_available():
                model.to("cuda")
            pipe = hf_pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=256,
                return_full_text=False,
            )
            hf_llm = HuggingFacePipeline(pipeline=pipe)
            self.llm = ChatHuggingFace(llm=hf_llm)
        else:
            raise ValueError(f"Unsupported embedder for LLM: {self.embedder_name}")

    def verify_ollama(self):
        client = ollama.Client(host=self.host)
        return [m.model for m in client.list().models]

    def build_index(self):
        """Runs stages 1-4, then wires BM25 for hybrid retrieval.

        Reuses a persisted index when the retriever doesn't need the raw
        corpus (bm25/hybrid must re-index chunks in memory).
        """
        if not hasattr(self.retriever, "index"):
            try:
                self.vectorstore.load()
                logger.info("Index loaded from disk")
                return self
            except FileNotFoundError:
                pass

        logger.info("Building index (loader=%s, chunker=%s, store=%s)", self.loader_name, self.chunker_name, self.vectorstore_name)
        docs = self.loader.load()                       # Stage 1
        chunks = self.chunker.split_documents(docs)     # Stage 2
        self.vectorstore.build(chunks)                  # Stages 3-4

        # Index chunks into retrieval stage (BM25/hybrid need the corpus)
        if hasattr(self.retriever, "index"):
            self.retriever.index(chunks)

        self.vectorstore.save()
        logger.info("Index ready and saved")
        return self

    def _retrieve_multi(self, queries):
        """Retrieve for each sub-query and merge, deduped by content.

        Content-based dedup because vector stores return fresh Document
        copies while bm25/hybrid return the original chunk objects.
        """
        seen, merged = set(), []
        for query in queries:
            for doc in self.retriever.retrieve(query):
                if doc.page_content in seen:
                    continue
                seen.add(doc.page_content)
                merged.append(doc)
        return merged

    def answer(self, question: str) -> str:
        """Runs stages 5-9: pre-retrieval, retrieval, context, prompt, generate."""
        start = time.perf_counter()
        q = question.strip()

        # Stage 5: pre-retrieval (rewrite + decompose into sub-queries)
        queries = self.pre_retrieval.transform(q)
        logger.info("Q: %r -> %d sub-queries", q, len(queries))
        logger.debug("Sub-queries: %s", queries)

        # Stage 6: retrieval (one pass per sub-query, merged)
        docs = self._retrieve_multi(queries)

        # Stage 7: post-retrieval (rerank + compress into context text)
        context = self.post_retrieval.process(docs, q)

        # Stages 8-9: prompt and generation
        if self.prompt_name == "conversational":
            variables = self.prompt_strategy.variables(q, context)
        else:
            variables = {"context": context, "question": q}
        answer = self.chain.invoke(variables)

        logger.info(
            "A: %d chunks -> %d chars context in %.2fs",
            len(docs), len(context), time.perf_counter() - start,
        )

        # Keep history for follow-up questions
        if self.prompt_name == "conversational":
            self.prompt_strategy.add_turn(q, answer)

        return answer

    def run_cli(self):
        if self.embedder_name == "ollama":
            self.verify_ollama()
        self.build_index()
        print(f"\nChat ready (loader={self.loader_name}, chunker={self.chunker_name}, "
              f"embedder={self.embedder_name}, vectorstore={self.vectorstore_name}, "
              f"pre={self.pre_retrieval_name}, retriever={self.retriever_name}, "
              f"post={self.post_retrieval_name}, prompt={self.prompt_name})")
        print("Type 'exit' or 'quit' to stop.\n")
        while True:
            user_input = input("Your question: ").strip()
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting...")
                break
            answer = self.answer(user_input)
            print("\nAnswer:\n", answer, "\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Pluggable RAG pipeline")
    parser.add_argument("--loader", default="pypdf", choices=["pypdf", "unstructured", "opendataloader"])
    parser.add_argument("--chunker", default="semantic", choices=["recursive", "token", "semantic"])
    parser.add_argument("--embedder", default="ollama", choices=["ollama", "openai", "huggingface"])
    parser.add_argument("--vectorstore", default="faiss", choices=["faiss", "chroma", "pgvector"])
    parser.add_argument("--retriever", default="vector", choices=["vector", "bm25", "hybrid"])
    parser.add_argument("--prompt", default="basic", choices=["basic", "conversational"])
    parser.add_argument("--pre-retrieval", default="query_transform", choices=["identity", "query_transform"])
    parser.add_argument("--post-retrieval", default="rerank", choices=["basic", "rerank"])
    parser.add_argument("--log-level", default=None, choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    setup_logging(args.log_level)
    pipeline = RAGPipelineAdvanced(
        loader=args.loader,
        chunker=args.chunker,
        embedder=args.embedder,
        vectorstore=args.vectorstore,
        retriever=args.retriever,
        prompt=args.prompt,
        pre_retrieval=args.pre_retrieval,
        post_retrieval=args.post_retrieval,
    )
    pipeline.run_cli()


if __name__ == "__main__":
    main()