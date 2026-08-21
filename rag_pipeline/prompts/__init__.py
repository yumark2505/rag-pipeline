from rag_pipeline.base import make_registry
from rag_pipeline.prompts.base_prompt import BasePrompt

prompt_registry, register_prompt = make_registry("prompt")

__all__ = ["BasePrompt", "prompt_registry", "register_prompt"]