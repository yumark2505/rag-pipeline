from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from config import CHAT_HISTORY_K
from prompts import register_prompt
from prompts.base_prompt import BasePrompt


@register_prompt("conversational")
class ConversationalPrompt(BasePrompt):
    """Strategy: multi-turn chatbot with chat history.

    Keeps the last N exchanges so the user can ask follow-up questions
    that reference previous answers (e.g. "expand on that last point").
    """

    SYSTEM_TEMPLATE = (
        "You are a strict, citation-focused assistant for a private knowledge base.\n"
        "RULES:\n"
        "1) Use ONLY the provided context to answer.\n"
        "2) If the answer is not clearly contained in the context, say: "
        "\"I don't know based on the provided documents.\"\n"
        "3) Do NOT use outside knowledge, guessing, or web information.\n"
        "4) If applicable, cite sources as (source:page) using the metadata.\n"
        "5) Use the chat history for context on follow-up questions, "
        "but answer from the provided context."
    )

    def __init__(self, history_k: int = CHAT_HISTORY_K):
        self.history_k = history_k
        self.history = []  # list of (role, text) tuples, oldest first

    def add_turn(self, question: str, answer: str):
        self.history.append(("user", question))
        self.history.append(("assistant", answer))
        # Trim to last N messages
        self.history = self.history[-self.history_k :]

    def _format_history(self) -> str:
        lines = []
        for role, text in self.history:
            prefix = "User" if role == "user" else "Assistant"
            lines.append(f"{prefix}: {text}")
        return "\n".join(lines) if lines else "(No prior conversation)"

    def build(self):
        system = SystemMessage(content=self.SYSTEM_TEMPLATE)
        template = ChatPromptTemplate.from_messages(
            [
                system,
                (
                    "human",
                    "Chat history:\n{history}\n\n"
                    "Context:\n{context}\n\n"
                    "Question: {question}",
                ),
            ]
        )
        return template

    def variables(self, question: str, context: str) -> dict:
        return {
            "history": self._format_history(),
            "context": context,
            "question": question,
        }