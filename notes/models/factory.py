"""
Model factory that returns the appropriate LLM handler based on CLI input.
"""
from notes.config.config_loader import ConfigLoader

from notes._exceptions import UnsupportedServiceError
# from notes.models.gemini import GeminiHandler
from notes.models.models import BaseModelHandler
# from notes.models.offline import OfflineHandler
# from notes.models.openai import OpenAIHandler


class ModelFactory:
    """Factory that returns the appropriate LLM handler based on CLI input."""

    _model_map = {
        # llms.OFFLINE.value: OfflineHandler,
        # llms.OLLAMA.value: OpenAIHandler,
        # llms.OPENAI.value: OpenAIHandler,
        # llms.GEMINI.value: GeminiHandler,
    }

    @staticmethod
    def model_handler(conf: ConfigLoader) -> BaseModelHandler:
        """Returns the appropriate LLM API handler based on CLI input."""
        llm_handler = ModelFactory._model_map.get(conf.config.llm.api)
        if llm_handler is None:
            raise UnsupportedServiceError(
                f"Unsupported LLM service provided: {conf.config.llm.api}"
            )
        return llm_handler(conf)
