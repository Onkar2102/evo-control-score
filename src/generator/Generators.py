from abc import ABC, abstractmethod
from utils.custom_logging import get_logger

logger = get_logger("Generators")


class TextGenerator(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name
        logger.debug(f"Initialized TextGenerator with model name: {model_name}")

    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass