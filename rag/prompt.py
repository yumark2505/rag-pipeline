from langchain_core.prompts import ChatPromptTemplate

from rag.config import CHAT_MODEL, OLLAMA_HOST


PROMPT_TEMPLATE = (
    "You are a strict, citation-focused assistant for a private knowledge base.\n"
    "RULES:\n"
    "1) Use ONLY the provided context to answer.\n"
    "2) If the answer is not clearly contained in the context, say: "
    "\"I don't know based on the provided documents.\"\n"
    "3) Do NOT use outside knowledge, guessing, or web information.\n"
    "4) If applicable, cite sources as (source:page) using the metadata.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}"
)


class PromptBuilder:
    """Stage 8: Assemble the chat prompt."""

    def build(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_template(PROMPT_TEMPLATE)


class Generator:
    """Stage 9: Generate the final answer from context + question."""

    def __init__(self, model: str = CHAT_MODEL, base_url: str = OLLAMA_HOST):
        from langchain_ollama import ChatOllama

        self.llm = ChatOllama(
            model=model,
            base_url=base_url,
            temperature=0.0,
        )

    def generate(self, prompt: ChatPromptTemplate, question: str, context: str) -> str:
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.runnables import RunnablePassthrough

        chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return chain.invoke({"context": context, "question": question})