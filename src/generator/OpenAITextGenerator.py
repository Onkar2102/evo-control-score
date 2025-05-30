from .Generators import TextGenerator
from dotenv import load_dotenv
import os
import openai

from utils.logging import get_logger

logger = get_logger("OpenAITextGenerator")

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class OpenAITextGenerator(TextGenerator):
    def __init__(self, config: dict):
        super().__init__(config["name"])
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        openai.api_key = self.api_key

        api_base = config.get("api_base")
        if api_base:
            openai.api_base = api_base

        self.model_name = config["name"]
        self.generation_args = config.get("generation_args", {})

        logger.debug(f"[OpenAITextGenerator] Initialized with model: {self.model_name}")
        if api_base:
            logger.debug(f"[OpenAITextGenerator] Using custom API base: {api_base}")
        logger.debug(f"[OpenAITextGenerator] Generation arguments: {self.generation_args}")

    def generate(self, prompt: str) -> str:
        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                **self.generation_args
            )
            content = response["choices"][0]["message"]["content"].strip()
            logger.info(f"Generated response using model: {self.model_name}")
            logger.debug(f"[OpenAITextGenerator] Prompt: {prompt}")
            logger.debug(f"[OpenAITextGenerator] Response: {content}")
            return content
        except openai.error.OpenAIError as e:
            logger.error(f"[OpenAITextGenerator] OpenAI API error: {str(e)}")
            raise RuntimeError(f"OpenAI API error: {str(e)}")
        except Exception as ex:
            logger.error(f"[OpenAITextGenerator] Unexpected error during generation: {str(ex)}")
            raise RuntimeError(f"Unexpected error during OpenAI generation: {str(ex)}")