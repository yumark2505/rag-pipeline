from rag.config import CHAT_MODEL, OLLAMA_HOST
from rag.embedder import OllamaEmbedder
from rag.loader import PDFLoader
from rag.chunker import SemanticChunkerStage
from rag.vectorstore import VectorStore
from rag.retriever import PreRetrieval, Retriever, PostRetrieval
from rag.prompt import PromptBuilder, Generator


class RAGPipeline:
    """Orchestrates the full 9-stage RAG flow.

    Stages: 1 Loader -> 2 Chunking -> 3 Embedding -> 4 Vector DB
            -> 5 Pre-retrieval -> 6 Retrieval -> 7 Post-retrieval
            -> 8 Prompt -> 9 Generation -> Answer
    """

    def __init__(
        self,
        embed_model: str = "nomic-embed-text",
        chat_model: str = CHAT_MODEL,
        host: str = OLLAMA_HOST,
    ):
        self.embed_model = embed_model
        self.chat_model = chat_model
        self.host = host

        self.embeddings = OllamaEmbedder(model=embed_model, host=host)
        self.loader = PDFLoader()
        self.chunker = SemanticChunkerStage(self.embeddings)
        self.vectorstore = VectorStore(self.embeddings)
        self.pre_retrieval = PreRetrieval(model=chat_model, host=host)
        self.retriever = Retriever(self.vectorstore)
        self.post_retrieval = PostRetrieval(model=chat_model, host=host)
        self.prompt_builder = PromptBuilder()
        self.generator = Generator(model=chat_model, base_url=host)

    def build_index(self):
        """Runs stages 1-4 to load docs and build the vector index."""
        if self.vectorstore.load():
            return self
        docs = self.loader.load()                     # Stage 1
        chunks = self.chunker.split(docs)             # Stage 2
        self.vectorstore.build(chunks)                # Stages 3-4
        self.vectorstore.save()                       # Persist for next run
        return self

    def answer(self, question: str) -> str:
        """Runs stages 5-9 for a single question."""
        queries = self.pre_retrieval.transform(question)  # Stage 5
        docs = self.retriever.retrieve(queries)           # Stage 6
        context = self.post_retrieval.process(docs, question)  # Stage 7
        prompt = self.prompt_builder.build()              # Stage 8
        answer = self.generator.generate(prompt, question, context)  # Stage 9
        return answer

    def run_cli(self):
        self.build_index()
        print("\nChat ready. Type 'exit' or 'quit' to stop.\n")
        while True:
            user_input = input("Your question: ").strip()
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting...")
                break
            answer = self.answer(user_input)
            print("\nAnswer:\n", answer, "\n")


def main():
    pipeline = RAGPipeline()
    pipeline.run_cli()


if __name__ == "__main__":
    main()