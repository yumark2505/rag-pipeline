from langchain_core.prompts import ChatPromptTemplate

from prompts import register_prompt
from prompts.base_prompt import BasePrompt


@register_prompt("basic")
class BasicPrompt(BasePrompt):
    """Strategy: single-turn, citation-focused assistant.

    Uses ONLY the provided context; no chat history involved.
    """

    TEMPLATE = (
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

    def build(self):
        return ChatPromptTemplate.from_template(self.TEMPLATE)