class BasePrompt:
    """Prompt interface. Build returns (template, input_variables)."""

    def build(self):
        raise NotImplementedError