from base import make_registry
from prompts.base_prompt import BasePrompt

prompt_registry, register_prompt = make_registry("prompt")

__all__ = ["BasePrompt", "prompt_registry", "register_prompt"]