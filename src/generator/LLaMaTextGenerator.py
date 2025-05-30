import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from generator.Generators import TextGenerator
from utils.logging import get_logger

load_dotenv()
LOGGER = get_logger("LLaMaTextGenerator")

class LlamaTextGenerator(TextGenerator):
    def __init__(self, config: dict):
        super().__init__(config["name"])
        self.model_name = config["name"]
        self.args = config.get("generation_args", {})

        self.tokenizer, self.model = self._load_model_and_tokenizer(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.to("cuda" if torch.cuda.is_available() else "cpu")

    def _load_model_and_tokenizer(self, model_name):
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HF_API_TOKEN")
        LOGGER.debug(f"Using HuggingFace token: {'FOUND' if hf_token else 'NOT FOUND'}")

        LOGGER.debug(f"Loading tokenizer for model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, legacy=False, token=hf_token)

        LOGGER.debug(f"Loading model for: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token)

        return tokenizer, model

    def generate(self, prompt: str) -> str:
        try:
            LOGGER.info(f"Generating text with model {self.model_name}")
            LOGGER.debug(f"Prompt: {prompt}")
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
            with torch.no_grad():
                output = self.model.generate(**inputs, **self.args)
            LOGGER.debug(f"Raw output tokens: {output}")
            response = self.tokenizer.decode(output[0], skip_special_tokens=True).strip()
            LOGGER.debug(f"Decoded response: {response}")
            return response
        except Exception as e:
            LOGGER.error(f"Text generation failed with model {self.model_name}: {e}")
            raise RuntimeError(f"Generation failed: {e}") from e